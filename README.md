# Anomaly detection for alerting: statistics, LLMs and decision models compared

Live site: https://anomaly-detection.coroot.com

Synthetic p99 latency scenarios for a web service, replayed minute by minute. At every minute a detector sees
only the trailing 60 minutes (240 samples at 15 s) and answers one question: page the on-call engineer now, or not.
Detectors are scored against human labels: did it page when a human would have, and stay silent when a human would not have.

## Dataset

31 series, 150 minutes each at 15 s resolution (600 points). The first 60 minutes are clean baseline, the scenario
unfolds afterwards. Evaluation runs every minute from minute 60 to 150.

Labels per series in `data/manifest.json`:

- `label`: `page` or `no_page`
- `onset`: first abnormal minute
- `page_from`, `page_by`: when a human would page, shown for context, not scored
- `resolved`: when a transient was over
- `rationale`: the human reasoning in one or two sentences

Mix: 13 page, 18 no page. Baselines vary from 120 to 350 ms. Every scenario is meant to be obvious to a human: page scenarios end above 1.3 s, no-page scenarios never exceed about 400 ms except for short spikes that recover and the low traffic series, where one sample in four sits between 1 and 1.5 s all night.

## Scoring

Binary: did the detector page at any point of the run, and should it have. A `page` series is correct if the
detector paged, a `no_page` series is correct if it stayed silent. The first PAGE minute is kept for inspection
but not scored.

A `for` duration can be applied on top of any detector, as in Prometheus: the detector must say page for N consecutive
minutes before a page counts. `report.py --for 3` prints the comparison table with that layer; the site has a For selector.
Short spikes are meant to be handled by this layer, not by the detector.

## Files

- `generate.py`: deterministic generator, edit scenarios and rerun
- `stream.py`: windowing, scoring, results format
- `detectors.py`: registry of statistical detectors (3σ variants, static SLO threshold, burn rate, ratio to previous hour, IQR fence, historical p99, EWMA control chart, CUSUM, generalized ESD, trend slope), writes `results/<name>.json`
- `make_prompt.py`: renders one window as model input
- `run_eval.py`: runs a chat model through the replay (Ollama native API with thinking off and JSON output, or any OpenAI compatible endpoint when the base ends in `/v1`), writes `results/<name>.json`
- `run_jev.py`: runs a Jev-style decision model (Ollama 0.35+, `/v1/systemone`, models `nimble`, `tev1`, `tev1:0.8b`) with one `noul` question, pages when p(page) is at or above a threshold
- `report.py`: comparison table for all result sets, optional `--for N`
- `build_site.py`: embeds data and all result sets into `site/index.html`
- `site/template.html`: the replay site source

## Results format

`results/<name>.json`: `{"name", "kind": "rule" | "model", "description", "decisions": {"s01": [{"m": 60, "page": false, "thr": 312.4, "reason": "..."}]}}`.
`thr` and `reason` are optional. Minutes that were not evaluated are simply absent.

## Usage

```
.venv/bin/python generate.py
.venv/bin/python detectors.py            # or: detectors.py cusum ratio_2x
.venv/bin/python run_eval.py --model gemma4:e4b-mlx --name gemma4_e4b --every 4   # Ollama on localhost:11434
.venv/bin/python run_eval.py --model <name> --base http://host:8000/v1 --stride 2 --only s16,s20
.venv/bin/python run_jev.py --model nimble --every 4
.venv/bin/python run_jev.py --model tev1:0.8b --threshold 0.7
export OPENROUTER_API_KEY=...                                                 # never in the repo
.venv/bin/python run_eval.py --base https://openrouter.ai/api/v1 --model anthropic/claude-opus-5.5 --name claude_opus_5.5 --every 4 --stride 2 --workers 6 --reasoning off
.venv/bin/python run_jev.py --base https://openrouter.ai/api --model typesafe/jev-1.13 --name jev_1.13 --every 4 --workers 6
.venv/bin/python build_site.py && open site/index.html
```

Hosted models go through OpenRouter: chat models via the OpenAI format (`--reasoning off|minimal|low|...` maps to OpenRouter's reasoning parameter), Jev via the same `/v1/systemone` endpoint as Ollama. The API key is read from `OPENROUTER_API_KEY`. `--workers N` sends N requests in parallel per series. With `--stride 2` the `for` layer counts consecutive evaluations, so `for 3` means two evaluations four minutes apart.

A full model run is 31 series x 91 minutes = 2821 prompts of about 800 tokens at full resolution, or about
250 tokens with `--every 4`. `--every 4` merges each minute into its max (or `--agg mean`). `--stop-on-page` stops a series after its first PAGE, which cuts the page scenarios short.
`--stride 2` halves the number of evaluations.

## Results so far (2026-11-03, 31 scenarios, for = 3 min, hold-aware prompts)

Model prompts explain the hold: the model is asked every minute, a page needs 3 PAGE answers in a row, so it should judge the current minute on its own and leave patience to the hold (`--hold N` on both runners).

| detector | correct | false pages | missed |
|---|---|---|---|
| burn_rate (1 s SLO, 5 min and 60 min windows) | 30 | low traffic | |
| jev_1.13_tuned (OpenRouter, p ≥ 0.59, tuned with the hold) | 29 | spike cluster | growing spikes |
| deepseek_v4_pro, gpt_5.5, tev1_4b (p ≥ 0.5) | 28 | 2 to 3 recovered transients | 0 to 1 |
| static_threshold (1 s, 5 min), trend_slope | 28 | | flapping, growing spikes, bimodal |
| claude_opus_5.5 (first prompt, not rerun), gemini_3.1_pro, jev_1.13 (p ≥ 0.5), nimble_9b_tuned (p ≥ 0.95) | 27 | | |
| iqr_fence, ratio_2x | 27 | 1 to 2 | 2 to 3 |
| 3sigma variants, generalized_esd | 26 | | |
| gemma4_e4b, cusum_tuned | 23 | 8 | 0 |
| tev1_0.8b, nimble_9b, cusum, 3sigma_any_point, ewma_control, percentile_p99 | 15 to 20 | | |

Explaining the hold helped the 4B decision model most (tev1 at its default threshold went from 23 to 28) and the frontier chat models hardly at all (GPT-5.5 stayed at 28, Gemini dropped by one). gemma4 e4b got worse: told to judge the minute alone, it pages on every transient. Models never miss a page except growing spikes, where each spike is short and a model judging one minute at a time correctly sees a healthy minute. Jev thresholds must be tuned with the hold (`jev_threshold.py NAME --for 3`). The two OpenRouter passes cost about 30 dollars, of which Jev 1.13 was 8 cents.

Scenarios are built to be obvious to a human: page scenarios go above 1.3 s, no-page scenarios stay under about
400 ms except for short spikes that recover. That is why the fixed 1 s SLO rules do so well here.
Models never miss a page but keep saying page for minutes after a spike has recovered, so `for` does not save
them. Rules with a 5 min lookback (ratio_2x) fire for 5 minutes after a 2 min spike for the same reason.
Tuned thresholds (tev1_4b_tuned at 0.72, nimble_9b_tuned at 0.94) were chosen on this dataset.

## Timestral-28M forecast baseline

Each minute, Timestral uses the first 30 minutes of the trailing hour to forecast the next 30 on CPU.
It flags the latest minute if every observed latency sample exceeds its own forecast 99th percentile
(four samples when complete). An empty context or target minute gives no signal.

With the shared three-minute hold, the baseline scores **27/31**, with zero false pages and four missed
incidents. The model size, window split and decision rule were selected after inspecting these scenarios.

```sh
.venv/bin/python -m pip install -r requirements-timestral.txt
.venv/bin/python run_timestral.py
```

## Hosting

The site is the single file `site/index.html`, built by `build_site.py` and committed. GitHub Pages deploys the `site` folder from `main` through `.github/workflows/pages.yml`, with the custom domain `anomaly-detection.coroot.com`.

## License

Apache License 2.0, see `LICENSE`.
