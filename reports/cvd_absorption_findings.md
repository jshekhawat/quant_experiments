# H1 — Price Absorption: Findings & Recommendation

**Status:** complete — negative result.
**Design:** `reports/cvd_absorption_study.md` (frozen before results).
**Machine outputs:** `experiments/reports/absorption/{summary,by_symbol,leave_one_out,robustness}.json`, `panel.parquet`.
**Reproduce:** `python scripts/absorption_study.py`.

---

## 1. Verdict

**NO EDGE (H0).** The preregistered criteria P1–P7 **all fail**. The impact
residual (absorption) does not predict forward returns in the hypothesised
direction; if anything it is mildly *inverted* out-of-sample and unstable.

## 2. What was tested

Rolling price-impact residual of aggressive perp taker flow, oriented forward
returns, strong-flow events, matched controls, and a control regression — on the
crypto-only top-10 perps, 1h bars, 4h primary horizon, 2026 (design Jan–Apr,
test May–Sep). 1m klines only; no tick data.

## 3. Results

| quantity | design | test |
|---|---|---|
| pooled rank IC (A vs oriented return) | +0.0049 (n=23,830) | **−0.0187** (n=36,670) |
| test 95% block-bootstrap CI | — | [−0.0379, +0.0007] |
| strong-flow-only test IC | — | −0.0047 |

- **Horizon IC (test):** 1h −0.0171 · 4h −0.0187 · 24h +0.0014 · 3d −0.0035.
- **Matched controls:** absorbed − control = **+17.2 bps**, but CI [−7.6, +37.0]
  **includes zero** (n=3,657 events) → not significant, and inconsistent with the
  negative headline IC.
- **Control regression:** A coefficient ≈ **0.00001**, CI [−0.00039, +0.00042] →
  indistinguishable from zero after controls.
- **Per-symbol test IC:** negative for **8 of 10** (HYPEUSDT +0.0203, BNBUSDT
  +0.0060 are the exceptions; OGNUSDT −0.0479, ETHUSDT −0.0379 weakest).
- **Quintiles (strong-flow):** non-monotone (−0.00018, −0.00079, −0.00057,
  +0.00008, +0.00049).

## 4. Robustness sweep (test IC)

18 cells (timeframe ∈ {15m,1h,4h} × window ∈ {15,30,60}d × event q ∈ {0.90,0.95}):

- **Test IC is negative in every single cell** (−0.0099 … −0.0233).
- **Design IC is positive in most cells** → a systematic design→test **sign flip**.
- Strong-flow-only IC is mixed and small.

This is the signature of a noise-driven in-sample artefact, not an edge.

## 5. Interpretation

The absorption story was mechanically plausible and the estimator was sound (a
rolling-regression residual, not an unstable ratio). The data still reject it:
any weak positive signal in the design window reverses out-of-sample and is
indistinguishable from zero after controls and matching. Combined with the
earlier spot–perp divergence negative, two independently motivated order-flow
hypotheses now fail on the same 2026 sample.

## 6. Limitations (why the negative is not the final word)

- **Single regime, in-sample 2026.** Only BTCUSDT has multi-year history; the
  universe is 9 months. No true out-of-sample across regimes — the same
  limitation that sank the divergence study.
- **Taker-flow proxy.** CVD cannot see maker intent, hidden liquidity, or
  cancellations; absorption can be a venue artefact.
- **n = 10 cross-section; 1m→1h resolution.** Sub-minute impact dynamics are
  invisible without tick data.

## 7. Recommendation

1. **Do not re-tune H1.** Changing windows/quantiles to manufacture a positive
   result is not science; the sweep already shows the sign is stable-negative OOS.
2. **Highest-leverage next step: build 2024–2025 perp 1m history** (~2 h, ~3 GB;
   the downloader already supports it). This converts *any* flow hypothesis from
   exploratory to genuinely out-of-sample and should precede new order-flow bets.
3. **Then test H3 (flow-confirmed breakout continuation)** rather than H2. H3 has
   a clean price-only baseline and a natural tradeable framing, and does not
   depend on untested swing detection. H2 remains attractive if the interest is
   reversal structure, but it first requires building (and validating) causal
   swing-high/low detection, which does not exist in the repo.
4. **H4/H5 as a market-state model, not a directional signal.** Directional
   price-impact asymmetry (H5) may still be useful as an execution-cost /
   slippage model even where it has no alpha; keep it as infrastructure.
5. **Do not acquire tick/aggTrades yet.** The binding constraint is regime
   breadth and effect existence, not resolution. Reserve tick data for execution
   realism or if a 2024–2026 bar-level test of a specific hypothesis is
   promising.

**Bottom line: abandon H1 as a directional signal; invest next in out-of-sample
history (2024–2025), then H3.**
