import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from make_prompt import system, render
from stream import load, eval_minutes, window, score, print_report, save_results


def auth():
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "none"
    return {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}


def ask(base, model, user, temperature, think, reasoning=None, sys_text=None):
    msgs = [{"role": "system", "content": sys_text or system()}, {"role": "user", "content": user}]
    if base.rstrip("/").endswith("/v1"):
        body = {"model": model, "temperature": temperature, "messages": msgs, "response_format": {"type": "json_object"}}
        if reasoning:
            body["reasoning"] = {"enabled": False} if reasoning == "off" else {"effort": reasoning}
        url, pick = f"{base.rstrip('/')}/chat/completions", lambda r: r["choices"][0]["message"]["content"]
    else:
        body = {"model": model, "stream": False, "think": think, "format": "json",
                "options": {"temperature": temperature}, "messages": msgs}
        url, pick = f"{base.rstrip('/')}/api/chat", lambda r: r["message"]["content"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=auth())
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return pick(json.load(r))
        except urllib.error.HTTPError as e:
            if e.code in (429, 402, 500, 502, 503, 524, 529) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")


def parse(text):
    mt = re.search(r'"decision"\s*:\s*"?(PAGE|NO_PAGE)', text, re.I)
    if mt:
        return mt.group(1).upper()
    up = text.upper()
    if "NO_PAGE" in up or "NO PAGE" in up:
        return "NO_PAGE"
    if "PAGE" in up:
        return "PAGE"
    return "UNPARSED"


def reason_of(text):
    mt = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
    return mt.group(1) if mt else text.strip()[:200]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:11434", help="Ollama native API; a base ending in /v1 uses the OpenAI chat format")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", help="results file name, defaults to model name")
    ap.add_argument("--every", type=int, default=1, help="thin samples: 4 = one per minute")
    ap.add_argument("--fmt", choices=["csv", "lines"], default="csv")
    ap.add_argument("--agg", choices=["max", "mean"], default="max", help="how to merge samples when thinning")
    ap.add_argument("--think", action="store_true", help="let thinking models think (slow)")
    ap.add_argument("--stride", type=int, default=1, help="evaluate every N minutes")
    ap.add_argument("--stop-on-page", action="store_true", help="skip the rest of a series after the first PAGE")
    ap.add_argument("--only", help="comma separated series ids")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--reasoning", help="OpenAI format only: off, minimal, low, medium or high")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests per series")
    ap.add_argument("--hold", type=int, default=3, help="the for duration the prompt tells the model about")
    a = ap.parse_args()
    sys_text = system(a.hold, 15 * a.every)

    manifest, series = load()
    if a.only:
        keep = set(a.only.split(","))
        manifest = [i for i in manifest if i["id"] in keep]
    name = a.name or re.sub(r"[^a-zA-Z0-9_.-]", "_", a.model)
    decisions = {}
    for item in manifest:
        out = []
        t0 = time.time()
        def one(m):
            try:
                raw = ask(a.base, a.model, render(window(series, item["id"], m), a.every, a.fmt, a.agg), a.temperature, a.think, a.reasoning, sys_text)
            except Exception as e:
                raw = f"ERROR {e}"
            d = parse(raw)
            return {"m": m, "page": d == "PAGE", "reason": reason_of(raw), "raw_decision": d}
        minutes = eval_minutes(series, a.stride)
        if a.workers > 1:
            with ThreadPoolExecutor(a.workers) as ex:
                out = list(ex.map(one, minutes))
        else:
            for m in minutes:
                out.append(one(m))
                if a.stop_on_page and out[-1]["page"]:
                    break
        errors = sum(1 for d in out if d["raw_decision"] == "UNPARSED")
        decisions[item["id"]] = out
        first = next((d["m"] for d in out if d["page"]), None)
        print(f"{item['id']} {item['name']:<40} expected={item['label']:<8} first_page={first}  unparsed={errors}  {round(time.time()-t0)}s", flush=True)
        via = "OpenRouter" if "openrouter" in a.base else "Ollama"
        thinking = (", reasoning " + a.reasoning) if a.reasoning else (", thinking on" if a.think else (", thinking off" if via == "Ollama" else ""))
        save_results(name, "model", f"Chat model {a.model} via {via}{thinking}. Gets the last 60 min as text ({'per-minute ' + a.agg if a.every == 4 else str(15 * a.every) + ' s'} samples) with an SRE on-call prompt that explains the {a.hold} min hold, and answers PAGE or NO_PAGE as JSON.", decisions,
                     {"model": a.model, "base": a.base, "every": a.every, "fmt": a.fmt, "agg": a.agg, "stride": a.stride, "think": a.think, "hold": a.hold, "reasoning": a.reasoning}, merge=bool(a.only))
    print_report(name, score(manifest, decisions))


if __name__ == "__main__":
    main()
