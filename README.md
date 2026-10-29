# PD Scorecard Studio


Build a classic PD scorecard on 30,000 public credit-card accounts: monotonic binning, WoE/IV analysis, logistic regression, points scaling, calibration and Gini/KS validation. A Claude agent answers risk questions by calling validated scorecard tools (bad rates, IV tables, performance, score breakdowns) and explains results in validator-friendly language.

Ask questions in plain English; a Claude agent answers by calling validated Python tools over
the real **UCI_Credit_Card** dataset (cc-by-4.0). Every number in an answer comes from a tool
result, and the app shows which tools ran, with their inputs and outputs.

## Example questions

- Which characteristics have the highest Information Value, and which would you drop?
- What is the scorecard's Gini on the test set, and is it well calibrated by decile?
- How does default rate change with the most recent repayment status PAY_0?
- Why did account 1234 get its score? Which characteristics cost it the most points?

## How it works

| Tool | Operation | What it does |
|---|---|---|
| `bad_rate_by_segment` | aggregate | Default rate and account count of default.payment.next.month grouped by a categorical driver (e.g. most recent repayment status PAY_0, EDUCATION). |
| `woe_iv_table` | woe_iv | Monotonic/quantile binning of one driver with per-bin counts, bad rate, WoE and total Information Value. |
| `scorecard_performance` | scorecard_performance | Fit (cached) WoE logistic scorecard on a seeded train split; return Gini, KS, AUC on train/test and a calibration table by score decile (predicted vs observed PD). |
| `score_breakdown` | score_breakdown | Points contribution per characteristic for one account ID or a what-if input, with the resulting score and PD. |
| `filter_accounts` | filter_rows | Return accounts matching filters on repayment status and education, for drill-down. |

- The model (`claude-opus-5-5`) only chooses tools and arguments. Arguments are validated against
  the real data (allowed metrics, filter values, date ranges, feature bounds); it never writes
  or runs code.
- Each question is limited to 6 model turns and 8 tool calls.
- All scorecard tools share one WoE logistic-regression scorecard, fitted on a stratified 70%
  training split (fixed seed) and validated on the remaining 30%. Points are scaled to 600 at
  50:1 good:bad odds, with 20 points to double the odds.

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` elsewhere),
then:

```bash
pip install -e ".[dev]"
```

```bash
python scripts/fetch_data.py
```

```bash
streamlit run app.py
```

For the chat tab, set `ANTHROPIC_API_KEY` in your environment or in
`.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`). The tool runner, evaluation
and data tabs work without a key.

## Tests and evaluation

```bash
pytest
```

Tests run offline on SYNTHETIC fixtures with a fake Anthropic client. After downloading the
data, the real-data evals run too.

```bash
python scripts/pin_expectations.py
```

```bash
python scripts/run_eval.py
```

`pin_expectations.py` recomputes the expected answers from the raw data with pyarrow and plain
Python, independently of the tool code. `run_eval.py --agent` also runs agent-level evals
(calls the API, costs tokens).

## Data and attribution

UCI_Credit_Card, https://zenodo.org/records/12524822 (licence: cc-by-4.0), describing credit-card
clients in Taiwan from April to September 2005 (University of California, Irvine). The data is
downloaded at setup time and is not committed to this repository.

Limitations:

- The data is a Taiwanese 2005 card portfolio, so results illustrate the method, not any
  current lender's portfolio.
- It has one observation window, so stability over time (PSI) cannot be measured.
- SEX, MARRIAGE and AGE are not model inputs.
- The repayment-status codes (PAY_0 to PAY_6) are binned one bin per value, so rare codes have
  unstable WoE.

## Deployment (not performed)

The app is ready for Streamlit Community Cloud: `requirements.txt` installs the package, and
the API key goes into the app's Secrets settings. The data is downloaded on first run from the
sidebar. Deployment and publishing to GitHub have **not** been done.
