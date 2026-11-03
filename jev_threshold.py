import argparse
import json
import re
from stream import RESULTS, load, score, summary


def apply(doc, thr):
    for ds in doc["decisions"].values():
        for d in ds:
            if d.get("p") is not None:
                d["page"] = d["p"] >= thr
    doc["threshold"] = thr
    doc["description"] = re.sub(r"at or above [0-9.]+\.?", f"at or above {thr}.", doc["description"])
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--apply", type=float, help="rewrite decisions at this threshold")
    ap.add_argument("--as", dest="as_name", help="with --apply: write a copy under this result name instead of overwriting")
    ap.add_argument("--for", dest="for_minutes", type=int, default=1, help="score the sweep with this hold duration")
    a = ap.parse_args()
    manifest, _ = load()
    path = RESULTS / f"{a.name}.json"
    doc = json.load(open(path))
    ps = sorted({d["p"] for ds in doc["decisions"].values() for d in ds if d.get("p") is not None})
    print(f"p(page) range {ps[0]:.3f} to {ps[-1]:.3f} over {len(ps)} distinct values")
    for item in manifest:
        pm = max((d["p"] for d in doc["decisions"].get(item["id"], []) if d.get("p") is not None), default=None)
        print(f"  {item['id']} {item['label']:<8}{'amb ' if item['ambiguous'] else '    '} max p = {pm}")
    print("\nthreshold  correct  false_pages  missed")
    for thr in [round(x, 2) for x in [i / 100 for i in range(int(ps[0] * 100), int(ps[-1] * 100) + 2)]]:
        s = summary(score(manifest, apply(json.loads(json.dumps(doc)), thr)["decisions"], for_minutes=a.for_minutes))
        print(f"{thr:<10} {s['correct']:<8} {len(s['false_page']):<12} {len(s['missed'])}")
    if a.apply is not None:
        doc = apply(doc, a.apply)
        if a.as_name:
            doc["name"] = a.as_name
            doc["description"] += f" That threshold was tuned on this dataset{' with a ' + str(a.for_minutes) + ' min hold' if a.for_minutes > 1 else ''}."
            path = RESULTS / f"{a.as_name}.json"
        json.dump(doc, open(path, "w"))
        print(f"\nwrote {path} at threshold {a.apply}")


if __name__ == "__main__":
    main()
