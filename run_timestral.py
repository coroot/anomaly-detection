import numpy as np
import torch
from timestral import TimestralForecaster

from stream import eval_minutes, load, print_report, save_results, score, window

MODEL = "mistralai/Timestral-28M"
REVISION = "49b26409ac87b5935539a5d8ffe7f397b720db47"
NAME = "timestral_28m_q99_30m"
DESCRIPTION = (
    "Timestral-28M forecasts the second half of the trailing hour from the first half. "
    "PAGE when every observed sample in the last minute exceeds its forecast 99th percentile."
)


def detect(forecaster, values):
    values = np.asarray(values, dtype=float)
    if values.shape != (240,) or np.isinf(values).any():
        raise ValueError("Expected 240 latency samples (NaN for missing data)")
    context, target = values[:120], values[-4:]
    observed = np.isfinite(target)
    if not observed.any():
        return False, "No observed samples in the last minute."
    if not np.isfinite(context).any():
        return False, "No observed context samples."

    forecast = forecaster.forecast(context, prediction_length=120,
                                   symmetric_averaging=True, clip_nonnegative=True)
    bounds = np.asarray(forecast.quantile(0.99))
    if bounds.shape != (1, 1, 120) or not np.isfinite(bounds).all():
        raise ValueError("Expected 120 finite forecast bounds for one series")
    bounds = bounds[0, 0, -4:]
    count = int((target[observed] > bounds[observed]).sum())
    total = int(observed.sum())
    reason = (f"{count}/{total} observed samples exceed their forecast q99 "
              f"(bounds {bounds.min():.1f} to {bounds.max():.1f} ms).")
    return count == total, reason


def main():
    manifest, series = load()
    if series["step_seconds"] != 15 or series["window_minutes"] != 60:
        raise ValueError("Expected a 60-minute window with samples every 15 seconds")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = TimestralForecaster.from_pretrained(MODEL, revision=REVISION, device="cpu")
    decisions = {}
    for item in manifest:
        sid = item["id"]
        rows = []
        for minute in eval_minutes(series):
            page, reason = detect(model, window(series, sid, minute))
            rows.append({"m": minute, "page": page, "reason": reason})
        decisions[sid] = rows
    save_results(NAME, "model", DESCRIPTION, decisions, {"model": MODEL, "revision": REVISION})
    print_report(NAME, score(manifest, decisions, for_minutes=3))


if __name__ == "__main__":
    main()
