import argparse
import json
from stream import RESULTS, load, score, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--for", dest="for_minutes", type=int, default=1, help="condition must hold this many consecutive minutes")
    a = ap.parse_args()
    manifest, _ = load()
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        doc = json.load(open(path))
        s = summary(score(manifest, doc["decisions"], a.for_minutes))
        rows.append((doc["name"], s))
    rows.sort(key=lambda r: -int(r[1]["correct"].split("/")[0]))
    print(f"for = {a.for_minutes} min")
    print(f"{'detector':<20} {'correct':<8} false pages / missed")
    for name, s in rows:
        print(f"{name:<20} {s['correct']:<8} {s['false_page']} / {s['missed']}")


if __name__ == "__main__":
    main()
