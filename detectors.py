import numpy as np
from scipy import stats
from stream import load, eval_minutes, window, ppm, score, print_report, save_results

RECENT_MIN = 5
SUSTAIN = 0.8
SLO_MS = 1000


def clean(y):
    return y[~np.isnan(y)]


def frac_over(recent, thr):
    ok = clean(recent)
    return (ok > thr).mean() if len(ok) else 0.0


def parts(w, p):
    return w[-RECENT_MIN * p:], w[: 30 * p], w[: -RECENT_MIN * p]


def mean_sd(y):
    y = clean(y)
    return y.mean(), y.std(ddof=1)


def robust(y):
    y = clean(y)
    med = np.median(y)
    return med, np.median(np.abs(y - med)) * 1.4826


def sigma_rule(baseline, stat, min_frac):
    def rule(w, p):
        recent, first30, prev = parts(w, p)
        mu, sd = stat(first30 if baseline == "first30" else prev)
        thr = mu + 3 * sd
        f = frac_over(recent, thr)
        return (f >= min_frac if min_frac > 0 else f > 0), thr
    return rule


def static_threshold(w, p):
    recent = w[-RECENT_MIN * p:]
    return frac_over(recent, SLO_MS) >= SUSTAIN, SLO_MS


def burn_rate(w, p):
    budget = 0.01
    fast = 14.4 * budget
    short = frac_over(w[-RECENT_MIN * p:], SLO_MS)
    long = frac_over(w, SLO_MS)
    return short > fast and long > fast, SLO_MS


def ratio_to_previous(w, p):
    recent, _, prev = parts(w, p)
    thr = 2.0 * np.nanmean(prev)
    r = clean(recent)
    return len(r) > 0 and r.mean() > thr, thr


def iqr_fence(w, p):
    recent, first30, _ = parts(w, p)
    q1, q3 = np.nanpercentile(first30, [25, 75])
    thr = q3 + 3 * (q3 - q1)
    return frac_over(recent, thr) >= SUSTAIN, thr


def percentile_history(w, p):
    recent, _, prev = parts(w, p)
    thr = np.nanpercentile(prev, 99)
    return frac_over(recent, thr) >= SUSTAIN, thr


def ewma_control(w, p, alpha=0.05):
    recent, _, prev = parts(w, p)
    y = clean(prev)
    m, v = y[0], 0.0
    for x in y[1:]:
        d = x - m
        m += alpha * d
        v = (1 - alpha) * (v + alpha * d * d)
    thr = m + 3 * np.sqrt(v)
    return frac_over(recent, thr) >= SUSTAIN, thr


def cusum(w, p, k_sd=0.5, h_sd=5.0):
    _, first30, _ = parts(w, p)
    mu, sd = mean_sd(first30)
    if sd == 0:
        return False, None
    s = 0.0
    for x in clean(w[30 * p:]):
        s = max(0.0, s + (x - mu - k_sd * sd))
    return s > h_sd * sd, None


def generalized_esd(w, p, alpha=0.05, max_frac=0.2):
    idx = np.where(~np.isnan(w))[0]
    y = w[idx].astype(float)
    n = len(y)
    if n < 10:
        return False, None
    r = int(max_frac * n)
    mask = np.ones(n, dtype=bool)
    removed, best = [], 0
    for i in range(1, r + 1):
        cur = y[mask]
        sd = cur.std(ddof=1)
        if sd == 0:
            break
        dev = np.abs(cur - cur.mean()) / sd
        j_local = int(np.argmax(dev))
        j = np.where(mask)[0][j_local]
        R = dev[j_local]
        m = n - i + 1
        t = stats.t.ppf(1 - alpha / (2 * m), m - 2)
        lam = (m - 1) * t / np.sqrt((m - 2 + t * t) * m)
        removed.append(j)
        mask[j] = False
        if R > lam:
            best = i
    flagged = set(idx[removed[:best]])
    recent_idx = [i for i in range(len(w) - RECENT_MIN * p, len(w)) if not np.isnan(w[i])]
    if not recent_idx:
        return False, None
    f = sum(1 for i in recent_idx if i in flagged and w[i] > np.nanmedian(w)) / len(recent_idx)
    return f >= SUSTAIN, None


def trend_slope(w, p, lookback_min=30, r2_min=0.5):
    _, first30, _ = parts(w, p)
    base = np.nanmean(first30)
    seg = w[-lookback_min * p:]
    t = np.arange(len(seg)) / p
    ok = ~np.isnan(seg)
    if ok.sum() < 10:
        return False, None
    res = stats.linregress(t[ok], seg[ok])
    projected = res.slope * lookback_min
    return projected > base and res.rvalue ** 2 > r2_min, None


DETECTORS = {
    "3sigma_any_point": ("Page when any single sample in the last 5 min is above normal plus 3 standard deviations. Normal is the first 30 min of the window.", sigma_rule("first30", mean_sd, 0)),
    "3sigma_sustained": ("Page when 80% of the last 5 min is above normal plus 3 standard deviations. Normal is the first 30 min of the window.", sigma_rule("first30", mean_sd, SUSTAIN)),
    "3sigma_trailing": ("Page when 80% of the last 5 min is above normal plus 3 standard deviations. Normal is everything before the last 5 min, so it follows slow drifts.", sigma_rule("trailing", mean_sd, SUSTAIN)),
    "3sigma_robust": ("Page when 80% of the last 5 min is above the median plus 3 MAD, a 3 sigma rule that ignores outliers. Normal is the first 30 min of the window.", sigma_rule("first30", robust, SUSTAIN)),
    "static_threshold": (f"Page when 80% of the last 5 min is above a fixed SLO of {SLO_MS} ms. No baseline, no statistics.", static_threshold),
    "burn_rate": (f"SLO burn rate, as in the Google SRE workbook: page when more than 14.4% of samples are above {SLO_MS} ms in both the last 5 min and the last 60 min. That is burning a 1% error budget 14.4 times too fast.", burn_rate),
    "ratio_2x": ("Page when the average of the last 5 min is more than twice the average of the preceding 55 min.", ratio_to_previous),
    "iqr_fence": ("Page when 80% of the last 5 min is above the box plot outlier fence, Q3 plus 3 IQR, of the first 30 min of the window.", iqr_fence),
    "percentile_p99": ("Page when 80% of the last 5 min is above the p99 of the preceding 55 min, so the current values would have been the worst 1% of the previous hour.", percentile_history),
    "ewma_control": ("Control chart: page when 80% of the last 5 min is above an exponentially weighted moving average plus 3 EWMA standard deviations of the preceding 55 min, alpha 0.05.", ewma_control),
    "cusum": ("CUSUM change detection over the last 30 min against the first 30 min: page when small deviations above normal, over 0.5 sigma, accumulate past 5 sigma. Textbook settings.", cusum),
    "cusum_tuned": ("Same CUSUM with deliberately insensitive settings: deviations over 1 sigma must accumulate past 20 sigma.", lambda w, p: cusum(w, p, 1.0, 20.0)),
    "generalized_esd": ("Generalized ESD outlier test, alpha 0.05, up to 20% of the window may be outliers: page when 80% of the last 5 min is flagged as a high outlier.", generalized_esd),
    "trend_slope": ("Trend: fit a straight line to the last 30 min. Page when the fit is clean, R squared above 0.5, and it projects latency rising by a whole baseline mean within the next 30 min.", trend_slope),
}


def main(only=None):
    manifest, series = load()
    p = ppm(series)
    for name, (desc, rule) in DETECTORS.items():
        if only and name not in only:
            continue
        decisions = {}
        for item in manifest:
            out = []
            for m in eval_minutes(series):
                w = np.array([np.nan if v is None else v for v in window(series, item["id"], m)], dtype=float)
                page, thr = rule(w, p)
                out.append({"m": m, "page": bool(page), "thr": None if thr is None else round(float(thr), 1)})
            decisions[item["id"]] = out
        save_results(name, "rule", desc, decisions)
        print_report(f"{name}: {desc}", score(manifest, decisions))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
