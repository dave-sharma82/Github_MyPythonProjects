# Australian Powerball Analytics Lab

A local interactive Streamlit application for the supplied 1,592-draw Australian Powerball CSV. It automatically restricts 7-from-35 analyses to draws from the current-rule era (2018 onward).

## Windows installation (VS Code or Windows Terminal)

1. Extract the ZIP into a folder, e.g. `C:\\PowerballLab`.
2. Open that folder in VS Code (File > Open Folder), then open Terminal > New Terminal.
3. Run:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Or, with a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Use the sidebar to upload an updated CSV, choose an analysis window and generation style, and export ten games as CSV.

## Features

- Data quality and rule-change detection (1996–2013: 5/45, 2013–2018: 6/40, 2018–present: 7/35)
- Historical counts vs expected, hot/cold and rolling trend plots
- Gap/overdue analysis, pair co-occurrence heatmap, scatter and statistical null simulation
- Draw sums, odd/even ratios, low/high split, consecutive patterns
- Ridge regression with strictly chronological holdout and comparison with the 20% baseline per number
- Ten distinct 7-from-35 game combinations with overlap control, reproducible RNG seed, CSV export
- Separate baseline, blended, experimental model-style weights. None of them predicts fair lottery outcomes.

## Statistical notes

Each fair 7-from-35 draw gives an individual number probability of 7/35 = 0.20. Within a draw, number indicators are negatively dependent. An ordinary chi-square frequency-test p-value that assumes independently sampled number occurrences is therefore not valid; the application uses Monte Carlo simulations drawing 7 distinct numbers each trial to calibrate its frequency statistic. The regression experiment is *diagnostic*, not a proven forecasting method; compare its held-out Brier error to the constant probability baseline (0.16 expected).

Each particular seven-number ticket has probability `1 / C(35, 7)` of matching all seven main numbers. Covering all twenty Powerball values for the same combination removes uncertainty about the separate Powerball number but does not change seven-main-number odds. If buying 10 distinct main-number sets and all 20 Powerball numbers per set, there are 200 standard-equivalent combinations, before considering retailer game/pricing rules. Never interpret the scoring heuristics as increased jackpot probability; spend only a fixed entertainment budget.
