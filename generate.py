import json
import numpy as np
import pandas as pd
from pathlib import Path

TOTAL_MIN = 150
WINDOW_MIN = 60
STEP_S = 15
PPM = 60 // STEP_S
N = TOTAL_MIN * PPM
START = pd.Timestamp("2026-11-02 00:00:00")
TS = pd.date_range(START, periods=N, freq=f"{STEP_S}s")
OUT = Path(__file__).parent / "data"
TIMEOUT_MS = 30000


def t_idx(minute):
    return int(round(minute * PPM))


def baseline(rng, base, sigma=0.08, rho=0.6):
    e = np.zeros(N)
    e[0] = rng.normal(0, sigma)
    for i in range(1, N):
        e[i] = rho * e[i - 1] + rng.normal(0, sigma * np.sqrt(1 - rho**2))
    return base * np.exp(e)


def spike(y, m_start, m_dur, factor, shape="box"):
    a, b = t_idx(m_start), t_idx(m_start + m_dur)
    b = min(max(b, a + 1), N)
    k = b - a
    if shape == "box":
        y[a:b] *= factor
    elif shape == "tri":
        y[a:b] *= 1 + (factor - 1) * (1 - np.abs(np.linspace(-1, 1, k)))
    elif shape == "decay":
        y[a:b] *= 1 + (factor - 1) * np.exp(-np.linspace(0, 4, k))
    return y


def ramp(y, m_start, m_end, f_start, f_end, curve="lin"):
    a, b = t_idx(m_start), t_idx(m_end)
    u = np.linspace(0, 1, b - a)
    prof = np.linspace(f_start, f_end, b - a) if curve == "lin" else f_start * (f_end / f_start) ** (u**2)
    y[a:b] *= prof
    y[b:] *= f_end
    return y


def gap(y, m_start, m_dur):
    y[t_idx(m_start):t_idx(m_start + m_dur)] = np.nan
    return y


def timeout(y, m_start, m_dur):
    y[t_idx(m_start):t_idx(m_start + m_dur)] = TIMEOUT_MS
    return y


def periodic(y, period, dur, f, start=2):
    m = start
    while m < TOTAL_MIN:
        spike(y, m, dur, f)
        m += period
    return y


def noisy_spikes(y, r):
    for m in [4, 13, 27, 38, 51, 66, 79, 93, 108, 121, 137, 144]:
        y[t_idx(m)] *= r.uniform(1.8, 2.6)
    return y


def sparse_slow(y, r, frac=0.25):
    last = -2
    for i in range(N):
        if i - last > 1 and r.random() < frac:
            y[i] *= r.uniform(5.8, 8.5)
            last = i
    return y


def cluster(y):
    spike(y, 80, 0.5, 3.5)
    spike(y, 82.5, 0.75, 4.0)
    spike(y, 84.5, 0.5, 3.0)
    return y


def flap(y):
    m = 100
    while m < TOTAL_MIN:
        spike(y, m, 2.5, 8)
        m += 5
    return y


def growing(y):
    for i, m in enumerate(range(68, TOTAL_MIN, 6)):
        spike(y, m, 0.25 + i * 0.1, 2 + i * 0.9)
    return y


def sawtooth(y, amp_fn):
    p = t_idx(10)
    i = np.arange(N)
    cyc = i // p
    phase = (i % p) / p
    return y * (1 + amp_fn(cyc) * phase)


def bimodal(y, m_start):
    idx = np.arange(t_idx(m_start), N)
    y[idx[idx % 2 == 1]] *= 10
    return y


S = []


def add(sid, name, label, category, difficulty, ambiguous, rationale, base, build,
        sigma=0.08, onset=None, page_from=None, page_by=None, resolved=None):
    S.append(dict(id=sid, name=name, label=label, category=category, difficulty=difficulty,
                  ambiguous=ambiguous, rationale=rationale, base_ms=base, build=build, sigma=sigma,
                  onset=onset, page_from=page_from, page_by=page_by, resolved=resolved))


# ---------------- never page ----------------
add("s01", "flat_normal", "no_page", "normal", "easy", False,
    "Steady latency with ordinary noise for the whole run. Nothing to act on.",
    240, lambda y, r: y)
add("s02", "noisy_normal_with_single_point_spikes", "no_page", "normal", "easy", False,
    "Higher variance and isolated single-sample spikes every 10 to 15 minutes, throughout. A noisy p99 at low traffic. No trend, no clustering.",
    180, noisy_spikes, sigma=0.14)
add("s03", "single_spike_recovered", "no_page", "transient", "easy", False,
    "One 30 second spike to 5x at minute 82, normal before and after. A transient.",
    300, lambda y, r: spike(y, 82, 0.5, 5), onset=82, resolved=83)
add("s04", "short_blip_recovered", "no_page", "transient", "easy", False,
    "Two minutes at up to 3x around minute 78, then back to normal. Review in the morning.",
    120, lambda y, r: spike(y, 78, 2, 3, "tri"), onset=78, resolved=80)
add("s05", "mild_drift_up", "no_page", "trend", "medium", False,
    "Latency drifts up about 15 percent between minute 60 and 150. Too small and too slow to page. A ticket at most.",
    210, lambda y, r: ramp(y, 60, 150, 1.0, 1.15), onset=60)
add("s06", "step_down_improvement", "no_page", "change", "easy", False,
    "A deploy at minute 90 cuts latency by a third and it stays there. A change, but an improvement.",
    350, lambda y, r: ramp(y, 90, 90.25, 1.0, 0.65), onset=90)
add("s07", "periodic_small_bumps", "no_page", "pattern", "medium", False,
    "A 30 second bump of 1.5x every 5 minutes, identical for the whole run. A cron job or GC cycle, not an incident.",
    160, lambda y, r: periodic(y, 5, 0.5, 1.5))
add("s08", "low_traffic_jitter", "no_page", "normal", "medium", False,
    "Wide random jitter between roughly 0.6x and 2x with no structure. A percentile over very few requests at night.",
    200, lambda y, r: y * r.lognormal(0, 0.28, N), sigma=0.05)
add("s09", "rise_then_full_recovery", "no_page", "transient", "medium", False,
    "Latency climbs from 260 to about 420 ms over 14 minutes, peaks at minute 94, and is back to baseline by minute 108. A bump, never a problem.",
    260, lambda y, r: spike(y, 80, 28, 1.6, "tri"), onset=80, resolved=108)
add("s10", "deploy_warmup", "no_page", "transient", "medium", False,
    "At minute 100 latency jumps to 2.5x and decays back to normal within 3 minutes. Cold start after a rollout.",
    140, lambda y, r: spike(y, 100, 3, 2.5, "decay"), onset=100, resolved=103)
add("s11", "past_data_gap", "no_page", "data_quality", "medium", False,
    "Three minutes of missing data at minute 85, normal latency before and after. A scrape hiccup.",
    230, lambda y, r: gap(y, 85, 3), onset=85, resolved=88)
add("s12", "spike_cluster_recovered", "no_page", "transient", "medium", False,
    "Three spikes of 3x to 4x between minute 80 and 85, then clean baseline. Worth a look tomorrow.",
    190, lambda y, r: cluster(y), onset=80, resolved=85)
add("s13", "stable_sawtooth", "no_page", "pattern", "medium", False,
    "A sawtooth with a 10 minute period and constant 1.6x amplitude, repeating identically. Periodic cache refill.",
    150, lambda y, r: sawtooth(y, lambda c: 0.6), sigma=0.04)
add("s14", "mean_reverting_wander", "no_page", "normal", "medium", False,
    "Slow wandering within about plus or minus 20 percent, always returning to the middle.",
    280, lambda y, r: y, sigma=0.12)
add("s15", "one_minute_outage_recovered", "no_page", "transient", "medium", False,
    "One minute pinned at the timeout ceiling at minute 74, then completely normal. Over before anyone could react. This is what a for duration exists for.",
    220, lambda y, r: timeout(y, 74, 1), onset=74, resolved=75)

# ---------------- page ----------------
add("s16", "step_up_sustained", "page", "step", "easy", False,
    "Latency jumps 5x at minute 95 and stays there. Five minutes of a sustained 5x is enough to page.",
    250, lambda y, r: ramp(y, 95, 95.25, 1.0, 5.0), onset=95, page_from=100, page_by=105)
add("s17", "ramp_still_climbing", "page", "trend", "easy", False,
    "From minute 90 latency climbs linearly from 200 ms, passing 1 s around minute 117 and reaching 2 s by the end. Page once it is clearly past 1 s and still climbing.",
    200, lambda y, r: ramp(y, 90, 150, 1.0, 10.0), onset=97, page_from=115, page_by=125)
add("s18", "timeout_plateau", "page", "step", "easy", False,
    "From minute 105 every sample sits at the 30 second timeout ceiling. Two minutes of that is enough.",
    180, lambda y, r: timeout(y, 105, 45), onset=105, page_from=107, page_by=110)
add("s19", "exponential_blowup", "page", "trend", "easy", False,
    "Slow rise from minute 80 that keeps accelerating: 1.9x at minute 115, 3.4x at minute 130, 12x at the end. Page when the curve is clearly bending up.",
    300, lambda y, r: ramp(y, 80, 150, 1.0, 12.0, "exp"), onset=108, page_from=125, page_by=135)
add("s20", "flapping", "page", "instability", "medium", False,
    "From minute 100 latency alternates between normal and 8x every 5 minutes. After the second flap it is clearly unstable.",
    160, lambda y, r: flap(y), onset=100, page_from=108, page_by=115)
add("s21", "post_deploy_regression", "page", "step", "easy", False,
    "A step from 200 ms to 1.2 s at minute 75 that never comes back. Five to fifteen minutes in, it needs a rollback.",
    200, lambda y, r: ramp(y, 75, 75.25, 1.0, 6.0), onset=75, page_from=80, page_by=90)
add("s22", "steady_degradation", "page", "trend", "medium", False,
    "Monotonic climb from 220 ms at minute 60 to about 2.2 s at minute 150, no plateau. Passes 1 s around minute 95. Page once it is clearly past the point of being a nuisance.",
    220, lambda y, r: ramp(y, 60, 150, 1.0, 10.0), onset=65, page_from=95, page_by=110)
add("s23", "growing_spikes", "page", "instability", "medium", False,
    "From minute 68 a spike every 6 minutes, each taller and longer than the last: 2x at first, 6.5x by minute 98, over 10x later. Page once the escalation is clear.",
    240, lambda y, r: growing(y), onset=68, page_from=98, page_by=116)
add("s24", "step_to_1200ms_sustained", "page", "step", "easy", False,
    "A step from 300 ms to 1.2 s at minute 80, sustained for the rest of the run. Ten minutes of that breaches any reasonable SLO.",
    300, lambda y, r: ramp(y, 80, 80.25, 1.0, 4.0), onset=80, page_from=85, page_by=95)
add("s25", "sawtooth_growing_amplitude", "page", "instability", "medium", False,
    "A normal sawtooth between 250 and 380 ms whose peaks start growing after minute 60: 600 ms, 830 ms, 1.05 s, 1.3 s and on. Each cycle ends worse than the last.",
    250, lambda y, r: sawtooth(y, lambda c: 0.5 + 0.9 * np.maximum(0, c - 6)), onset=75, page_from=100, page_by=115)
add("s26", "spike_then_sustained_break", "page", "step", "easy", False,
    "A one minute 6x spike at minute 80 that fully recovers, then a 5x jump at minute 110 that holds until the end. The first one is noise, the second one is the page.",
    240, lambda y, r: ramp(spike(y, 80, 1, 6.0), 110, 110.25, 1.0, 5.0), onset=110, page_from=115, page_by=122)
add("s27", "bimodal_half_fleet", "page", "instability", "hard", False,
    "From minute 100 samples alternate between normal and 10x. Half the requests are fine and half are not: one bad instance behind the balancer.",
    150, lambda y, r: bimodal(y, 100), onset=100, page_from=105, page_by=112)

add("s28", "two_min_spike_recovered", "no_page", "transient", "medium", False,
    "Two minutes at 1.7 s from minute 100, then back to 210 ms for good. Big but short, over before a for duration would expire.",
    210, lambda y, r: spike(y, 100, 2, 8), onset=100, resolved=102)
add("s29", "slow_ramp_to_340ms", "no_page", "trend", "medium", False,
    "A smooth climb from 190 to 340 ms between minute 60 and 150. Abnormal, not harmful. A morning ticket, not a page.",
    190, lambda y, r: ramp(y, 60, 150, 1.0, 1.8), onset=60)
add("s30", "two_stage_degradation", "page", "step", "easy", False,
    "A step from 230 to 460 ms at minute 85, then a second step to 1.4 s at minute 110, both sustained. The first stage is a ticket, the second is a page.",
    230, lambda y, r: ramp(ramp(y, 85, 85.25, 1.0, 2.0), 110, 110.25, 1.0, 3.0), onset=85, page_from=115, page_by=125)
add("s31", "low_traffic_sparse_slow_requests", "no_page", "normal", "medium", False,
    "Night traffic: a few requests per 15 s bucket, so p99 is the single slowest request. One sample in four lands between 1 and 1.5 s all night long, never two in a row, while the typical value stays at 180 ms. Nothing changed and nobody is affected.",
    180, sparse_slow, sigma=0.06)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "csv").mkdir(exist_ok=True)
    manifest, combined = [], {}
    for i, s in enumerate(S):
        rng = np.random.default_rng(1000 + i)
        y = baseline(rng, s["base_ms"], sigma=s["sigma"])
        y = s["build"](y, rng)
        y = np.round(np.minimum(y, TIMEOUT_MS), 1)
        pd.DataFrame({"timestamp": TS, "latency_ms": y}).to_csv(OUT / "csv" / f'{s["id"]}_{s["name"]}.csv', index=False)
        combined[s["id"]] = [None if np.isnan(v) else float(v) for v in y]
        manifest.append({k: s[k] for k in ("id", "name", "label", "category", "difficulty", "ambiguous",
                                           "rationale", "base_ms", "onset", "page_from", "page_by", "resolved")})
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    json.dump({"start": str(START), "step_seconds": STEP_S, "points": N, "total_minutes": TOTAL_MIN,
               "window_minutes": WINDOW_MIN, "eval_from_minute": WINDOW_MIN, "eval_to_minute": TOTAL_MIN,
               "series": combined}, open(OUT / "series.json", "w"))
    print(f"{len(S)} series x {N} points written to {OUT}")


if __name__ == "__main__":
    main()
