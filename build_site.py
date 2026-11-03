import json
from pathlib import Path
from stream import ROOT, RESULTS, load
from make_prompt import system
from run_jev import question

manifest, series = load()
results = {}
for p in sorted(RESULTS.glob("*.json")):
    doc = json.load(open(p))
    results[doc["name"]] = doc
data = {"manifest": manifest, "series": series, "results": results, "prompts": {"chat": system(3, 60), "jev": question(3)}}
tpl = (ROOT / "site" / "template.html").read_text()
out = ROOT / "site" / "index.html"
out.write_text(tpl.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":"))))
print(out, f"{out.stat().st_size // 1024} KB, result sets: {list(results)}")
