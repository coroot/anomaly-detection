import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from make_prompt import thin
from stream import load, eval_minutes, window, score, print_report, save_results

def question(hold=3):
    return (
        "You are an SRE on call at night. The state holds the last 60 minutes of p99 latency of a production web "
        "service, oldest first, the last value is now. You are asked this question every minute, and the on-call "
        f"engineer is paged only if you answer yes for {hold} minutes in a row, like a `for` clause in Prometheus. "
        "Judge this minute on its own: is latency right now at a level that would be a real problem for users if it "
        "continued? Answer no if the service looks healthy right now, even if there was a spike earlier that has since "
        "recovered. Do not wait for confirmation yourself, the hold does that. Noise and regular repeating patterns "
        "are not a problem."
    )


def ask(base, model, values, every, agg, hold=3):
    vals = thin(values, every, agg)
    body = {
        "model": model,
        "state": {
            "metric": "p99 latency in milliseconds of a production web service",
            "sampling": f"one value every {15 * every} seconds, oldest first, the last value is now, null means no data",
            "p99_latency_ms": ",".join("null" if v is None else str(int(round(v))) for v in vals),
        },
        "questions": {"page": {"type": "noul", "instructions": question(hold)}},
    }
    key = os.environ.get("OPENROUTER_API_KEY", "none")
    req = urllib.request.Request(f"{base.rstrip('/')}/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (429, 402, 500, 502, 503, 524, 529) and attempt < 3:
                time.sleep(2 * (attempt + 1))
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--name")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--agg", choices=["max", "mean"], default="max")
    ap.add_argument("--threshold", type=float, default=0.5, help="page when p(page) is at or above this")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--only")
    ap.add_argument("--workers", type=int, default=1, help="parallel requests per series")
    ap.add_argument("--hold", type=int, default=3, help="the for duration the question tells the model about")
    a = ap.parse_args()

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
                r = ask(a.base, a.model, window(series, item["id"], m), a.every, a.agg, a.hold)
                p = float(r["answers"]["page"]["noul"])
                return {"m": m, "page": p >= a.threshold, "p": round(p, 3), "reason": f"p(page) = {p:.2f}"}
            except Exception as e:
                return {"m": m, "page": False, "p": None, "reason": f"error: {e}"}
        minutes = eval_minutes(series, a.stride)
        if a.workers > 1:
            with ThreadPoolExecutor(a.workers) as ex:
                out = list(ex.map(one, minutes))
        else:
            out = [one(m) for m in minutes]
        decisions[item["id"]] = out
        first = next((d["m"] for d in out if d["page"]), None)
        pmax = max((d["p"] for d in out if d["p"] is not None), default=None)
        print(f"{item['id']} {item['name']:<40} expected={item['label']:<8} first_page={first}  max p={pmax}  {round(time.time()-t0)}s", flush=True)
        via = "OpenRouter" if "openrouter" in a.base else "Ollama"
        save_results(name, "model", f"Jev decision model {a.model} via {via}. Gets the last 60 min ({'per-minute ' + a.agg if a.every == 4 else str(15 * a.every) + ' s'} samples) and one yes or no question that explains the {a.hold} min hold, returns a probability of page. "
                     f"Pages when p is at or above {a.threshold}.", decisions,
                     {"model": a.model, "base": a.base, "every": a.every, "agg": a.agg, "stride": a.stride, "threshold": a.threshold, "api": "systemone", "hold": a.hold}, merge=bool(a.only))
    print_report(name, score(manifest, decisions))


if __name__ == "__main__":
    main()
