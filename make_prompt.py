import argparse
from stream import load, window

def system(hold=3, step=15):
    return (
        "You are an experienced SRE on call at night. You will be shown the last 60 minutes of p99 latency "
        f"for a production web service, one value every {step} seconds, ending right now. "
        "You are asked this same question every minute. Your answer feeds an alerting rule that pages the on-call "
        f"engineer only if you answer PAGE for {hold} minutes in a row, like a `for` clause in Prometheus. "
        "So judge this minute on its own. Answer PAGE if latency right now is at a level that would be a real problem "
        "for users if it continued. Answer NO_PAGE if the service looks healthy right now, even if there was a spike "
        "earlier that has since recovered. Do not wait for confirmation yourself, the hold does that. "
        "Noise and regular repeating patterns are not a problem. "
        "Answer with a JSON object only: {\"decision\": \"PAGE\" or \"NO_PAGE\", \"reason\": \"one sentence\"}."
    )


SYSTEM = system()


def thin(values, every, agg="max"):
    if every <= 1:
        return list(values)
    out = []
    for i in range(0, len(values), every):
        chunk = [v for v in values[i:i + every] if v is not None]
        out.append(None if not chunk else (max(chunk) if agg == "max" else sum(chunk) / len(chunk)))
    return out


def render(values, every=1, fmt="csv", agg="max"):
    vals = thin(values, every, agg)
    step = 15 * every
    if fmt == "csv":
        body = ",".join("null" if v is None else str(int(round(v))) for v in vals)
        return f"p99_latency_ms (oldest first, {step}s apart, {len(vals)} points, last value is now):\n{body}"
    lines = []
    for i, v in enumerate(vals):
        ago = (len(vals) - i) * step
        lines.append(f"-{ago//60:02d}:{ago%60:02d} {'null' if v is None else int(round(v))}")
    return "time_before_now p99_latency_ms\n" + "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("id")
    ap.add_argument("--minute", type=int, default=150)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--fmt", choices=["csv", "lines"], default="csv")
    ap.add_argument("--agg", choices=["max", "mean"], default="max")
    a = ap.parse_args()
    _, series = load()
    print(f"[system]\n{system(3, 15 * a.every)}\n[user]\n{render(window(series, a.id, a.minute), a.every, a.fmt, a.agg)}")
