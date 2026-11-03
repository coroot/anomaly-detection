import json
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load():
    manifest = json.load(open(DATA / "manifest.json"))
    series = json.load(open(DATA / "series.json"))
    return manifest, series


def ppm(series):
    return 60 // series["step_seconds"]


def eval_minutes(series, stride=1):
    return list(range(series["eval_from_minute"], series["eval_to_minute"] + 1, stride))


def window(series, sid, minute):
    p = ppm(series)
    w = series["window_minutes"]
    return series["series"][sid][(minute - w) * p: minute * p]


def classify(item, first_page):
    paged = first_page is not None
    if item["label"] == "page":
        return "paged" if paged else "missed"
    return "false_page" if paged else "silent"


def apply_for(ds, for_minutes):
    if for_minutes <= 1 or not ds:
        return ds
    ds = sorted(ds, key=lambda d: d["m"])
    stride = min((b["m"] - a["m"] for a, b in zip(ds, ds[1:])), default=1)
    need = -(-for_minutes // stride)
    out = []
    for i, d in enumerate(ds):
        held = i + 1 >= need and all(x["page"] for x in ds[i + 1 - need:i + 1])
        out.append({**d, "page": bool(held)})
    return out


def score(manifest, decisions, for_minutes=1):
    rows = []
    for item in manifest:
        ds = apply_for(decisions.get(item["id"], []), for_minutes)
        paged = [d["m"] for d in ds if d["page"]]
        first = min(paged) if paged else None
        rows.append(dict(id=item["id"], name=item["name"], label=item["label"], ambiguous=item["ambiguous"],
                         first_page=first, outcome=classify(item, first),
                         correct=classify(item, first) in ("paged", "silent")))
    return rows


def summary(rows):
    clear = [r for r in rows if not r["ambiguous"]]
    amb = [r for r in rows if r["ambiguous"]]
    return {
        "correct": f"{sum(r['correct'] for r in clear)}/{len(clear)}",
        "false_page": [r["id"] for r in clear if r["outcome"] == "false_page"],
        "missed": [r["id"] for r in clear if r["outcome"] == "missed"],
        "ambiguous_correct": f"{sum(r['correct'] for r in amb)}/{len(amb)}",
    }


def print_report(name, rows):
    s = summary(rows)
    print(f"\n== {name}")
    print(f"correct {s['correct']}  ambiguous {s['ambiguous_correct']}")
    print(f"false pages {s['false_page']}")
    print(f"missed      {s['missed']}")


def save_results(name, kind, description, decisions, extra=None, merge=False):
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    if merge and path.exists():
        merged = json.load(open(path))["decisions"]
        merged.update(decisions)
        decisions = merged
    doc = {"name": name, "kind": kind, "description": description, "decisions": decisions}
    if extra:
        doc.update(extra)
    json.dump(doc, open(path, "w"))
    return path
