# Hyperliquid Market-Data Acquisition — Proposal

**Status:** proposal for review — nothing implemented.
**Goal:** obtain, for the **entire Hyperliquid universe**, the same data we have for
Binance from **2025-01-01 onward**: 1-minute OHLCV **with a taker-buy split (for
CVD)**, funding rates, and open interest — usable by the existing `quant_lab`
pipeline (`data/sources/*`, canonical schemas, resumable backfill).
**Probe date:** 2026-10-10. All numbers below were measured live or quoted from the
official docs.

---

## 0b. Vendor validated — Hydromancer Reservoir (recommended path)

The vendor `https://docs.hydromancer.xyz/reservoir/hyperliquid` provides exactly the
data we need, **already normalized to Parquet**, on `s3://hydromancer-reservoir`
(requester-pays, region **ap-northeast-1**). Access confirmed via the local scoped
AWS profile `hydromancer`.

**Coverage:** **2025-07-28 → present** (complete from that date; not all of 2025).
So the OOS design window would be ~2025-08 → 2026-01 (≈5 months) + 2026 test.

**Exact dataset sizes** (whole datasets, measured with `aws s3 ls --recursive --summarize`):

| dataset | path | size | objects |
|---|---|---|---|
| Perp fills (main dex, all) | `by_dex/hyperliquid/fills/perp/all/` | **138.2 GiB** | 438 |
| **All fills (all dexes, perp+spot)** | `global/fills/raw/` | **218.9 GiB** | 438 |
| Spot fills | `global/fills/spot/all/` | 16.2 GiB | 438 |
| Perp candles 1s | `by_dex/hyperliquid/candles/1s/` | 11.5 GiB | 435 |
| Spot candles 1s | `global/candles/1s/` | 15.0 GiB | 435 |
| Perp position snapshots | `by_dex/hyperliquid/snapshots/perp/` | 5.5 GiB | 386 |
| Account-value snapshots | `global/snapshots/account_values/` | 17.6 GiB | 386 |
| Spot-holdings snapshots | `global/snapshots/spot/` | 32.0 GiB | 386 |
| Perp liquidations | `by_dex/hyperliquid/fills/perp/liquidations/` | 0.32 GiB | 438 |
| Perp builder / TWAP fills | `.../builder_fills/`, `.../twap_fills/` | 12.6 / 5.4 GiB | 438 / 433 |
| Order book 1m (perp) | `by_dex/hyperliquid/orderbook/1m/perps/` | 13.5 GiB | 54,689 |

**Schemas (verified from a live sample, one day, 2026-03-22):**

- **Fills** — 27-col Parquet, one file/day (~211 MiB/day perp): `coin, dex,
  asset_class, base_symbol, quote_symbol, price, size, side (buy/sell),
  timestamp, direction, realized_pnl, tx_hash, order_id, trade_id, fee,
  fee_token, address, crossed (taker flag), start_position, …, is_liquidation,
  liquidation_mark_px, liquidation_method`.
  - **⚠ Each trade appears twice**: one `crossed=true` (taker) row and one
    `crossed=false` (maker) row; side counts are exactly equal (verified: 1,734,578
    each). Build bars/volume from **one side only** (`crossed=true`):
    `taker_buy_volume = Σsize where crossed & side==buy`.
- **Candles 1s** — ~19 MiB/day: `open/high/low/close/volume/volume_quote/trade_count`
  per active second, **no taker split** → good for price cross-checks, useless for CVD.
- **Snapshots** — daily per-user positions → **OI by summation** (positions), plus
  account values. (Binance gave 5-min OI; here OI is daily-reconstructable.)

**Funding:** not published by the vendor → pull free from the Hyperliquid API
`fundingHistory` (full hourly history).

**Cost (egress $0.09/GB, 100 GB/month free tier):**

| scope | size | one-shot $ | paced (free tier) |
|---|---|---|---|
| Candles only (perp+spot) | 26.5 GiB | **$0** | $0 |
| Main-dex perp fills + candles + snapshots | ~155 GiB | **~$5** | $0 (2 months) |
| **All fills (`raw`) + snapshots** (recommended) | ~224 GiB | **~$11** | $0 (3 months) |
| Everything incl. snapshots + orderbook | ~500 GiB | ~$36 | $0 (5 months) |

Request costs are negligible (~438 objects/dataset → a few hundred GETs; orderbook
54k objects ≈ $0.02). Transfer speed measured ~3–6 MiB/s.

**Verdict: use the vendor.** It is cheaper, cleaner, finer (`1s` candles + true
fills for CVD), and removes the entire "reconstruct from raw node fills" R&D cost.
The self-hosted `node_fills` path (§2–3, Option A) is no longer needed.

---

## 0. Bottom line

**Feasible, but materially harder and more expensive than Binance.** Binance hands
you a free bulk archive of 1-minute klines that already contain the taker-buy
split. Hyperliquid does **not**:

- The candle API serves **only the most recent 5000 candles** (verified: `0` rows
  for 2025-01-01; `5016` for the last 8 days at 1m, i.e. capped). **No deep candle
  history exists via the API.**
- The official S3 archive explicitly provides **"no candles or spot asset data"** —
  only L2 book snapshots (`market_data`), daily asset contexts (`asset_ctxs`), and
  raw **fills** (`hl-mainnet-node-data/node_fills*`).
- Therefore the only way to reconstruct 1-minute bars + taker flow back to 2025 is
  **from raw fills** (`node_fills_by_block`), which are **requester-pays** (we pay
  S3 egress) and come with a "timely updates not guaranteed / data may be missing"
  caveat.

So the recommendation is a **hybrid**: reconstruct bars/CVD from archived fills,
pull funding from the API, pull OI from daily `asset_ctxs`, and run a **live
WebSocket capture** to stay current and fill gaps.

---

## 1. Hyperliquid data surface (what actually exists)

### 1.1 Live APIs (`POST https://api.hyperliquid.xyz/info`)

| need | endpoint | depth / notes |
|---|---|---|
| universe (perps) | `meta`, `allPerpMetas`, `perpDexs` | full, incl. `isDelisted` |
| universe (spot) | `spotMeta` | full |
| candles | `candleSnapshot` | **only last 5000** — useless for backfill |
| funding | `fundingHistory` | **full history**, paginated 500/req, **hourly** |
| open interest | `metaAndAssetCtxs`, `activeAssetCtx` | **current only** |
| trades | *(WS only)* | recent/live only; no REST history |

Measured universe (2026-10-10):

| segment | count |
|---|---|
| main perp dex coins | **234** (of which **56 `isDelisted`**) |
| HIP-3 builder perp dexs | **10** (`xyz, flx, vntl, hyna, km, abcd, cash, para, mkts, io`) |
| spot pairs | **330** (over 503 tokens) |

### 1.2 Official S3 archives (requester-pays)

| bucket / prefix | contents | useful for |
|---|---|---|
| `hyperliquid-archive/market_data/[date]/[hour]/[datatype]/[coin].lz4` | L2 book snapshots (+ asset ctx) | microstructure (optional) |
| `hyperliquid-archive/asset_ctxs/[date].csv.lz4` | **daily** per-coin ctx (OI, funding, mark, volume) | **OI + funding history** |
| `hl-mainnet-node-data/node_fills_by_block` | **fills**, API format, per block | **bars + CVD (primary source)** |
| `hl-mainnet-node-data/node_fills` | fills, older format (API-compatible) | pre-migration history |
| `hl-mainnet-node-data/node_trades` | trades, **different** format | fallback |
| `hl-mainnet-node-data/misc_events_by_block` | funding, transfers, staking | funding cross-check |
| `hl-mainnet-node-data/explorer_blocks`, `replica_cmds` | L1 blocks / txs | deep reconstruction |

Notes: uploads are **~monthly** and "**data may be missing**"; downloads require an
AWS account and the requester pays transfer. Anonymous listing is denied (verified
empty), so **AWS credentials are required** to even enumerate.

---

## 2. Can we get "the same data"? — feasibility

| Binance dataset | Hyperliquid equivalent | Feasible 2025+? |
|---|---|---|
| 1m klines (OHLCV) | **reconstruct from `node_fills_by_block`** | ✅ (fills archive) |
| taker-buy volume (→ CVD) | derive from fills' **taker side** (`crossed` + `side`) | ✅ **only** via fills |
| trades / aggTrades | fills (px, sz, side, time, taker flag) | ✅ |
| funding (8h) | `fundingHistory` API (**hourly**) + `misc_events` | ✅ cheap |
| open interest (5m) | `asset_ctxs` **daily** + live snapshots | ⚠️ daily only for history |
| mark/index/premium | asset ctx `markPx`/`oraclePx`/`premium` | ⚠️ daily for history |

**CVD is the crux.** Hyperliquid candles carry only `v` (base volume) and `n`
(trade count) — **no buy/sell split**. CVD must be built from fills, where each
fill records the crossing order's `side` and a `crossed` (taker) flag. This is
arguably *better* than Binance (true tick), but it is a full reconstruction job,
not a download.

---

## 3. Options

### Option A — Self-hosted reconstruction from raw fills (recommended core)
Reconstruct 1m bars + taker-buy volume + trades for **every coin** (perp + spot)
from `node_fills_by_block`, 2025→present, into the canonical `quant_lab` schemas.
- **Pros:** complete, all coins incl. delisted & spot, true tick, no vendor, same
  schema as Binance → all existing analytics work unchanged.
- **Cons:** requester-pays egress cost; large volume; format-era handling;
  monthly lag + "may be missing" → needs live top-up.

### Option B — Hybrid (recommended overall) = A + API + live capture
- **Bars/CVD/trades:** Option A (fills).
- **Funding:** `fundingHistory` API (cheap, complete, hourly).
- **OI:** `asset_ctxs` daily for history; **live `activeAssetCtx` snapshots** (e.g.
  1/min per coin) going forward for finer OI.
- **Freshness/gap-fill:** WS daemon (`trades`, `candle`, `activeAssetCtx`) writes
  bars + OI/funding continuously, backfilling the S3 monthly lag.

### Option C — Third-party vendor
Tardis.dev (HL trades/book/1m), CoinGlass/Amberdata (OI/funding/liquidations).
- **Pros:** fast, clean, no S3 plumbing.
- **Cons:** paid + licensing; may not cover the full HIP-3/spot/long tail; lock-in.

### Option D — Live-only from now
WS capture everything going forward.
- **Pros:** cheap, clean, immediate.
- **Cons:** **no 2025 history** — fails the requirement; only viable as the
  freshness layer of Option B.

**Recommendation: Option B.** Option A alone is sufficient for history but has no
freshness; B adds the API + live layer cheaply.

---

## 4. Proposed architecture (mapping to `quant_lab`)

```
hyperliquid/
├─ sources/hyperliquid.py      # REST info client + S3 (boto3, requester-pays)
├─ hl_fills/                   # node_fills_by_block -> normalized fills parquet
│    └─ aggregate to 1m bars (taker split) + trades
├─ funding / open_interest     # API + asset_ctxs + live snapshots
└─ live_capture/               # WS daemon (trades, candle, activeAssetCtx)
```

**Schema mapping (existing canonical schemas — no new schema needed):**

| canonical `BAR_SCHEMA` | from fills (per coin, per minute) |
|---|---|
| `open/high/low/close` | first/max/min/last `px` |
| `volume` | Σ `sz` |
| `quote_volume` | Σ `px·sz` |
| `trade_count` | count(fills) |
| **`taker_buy_volume`** | Σ `sz` where taker side = buy (`crossed` & buy) |
| **`taker_buy_quote_volume`** | Σ `px·sz` where taker side = buy |

- **Trade/aggTrade:** fields map directly (`side` = taker side).
- **Funding:** `{timestamp, symbol, funding_rate, funding_interval_hours=1}`.
- **OpenInterest:** `{timestamp, symbol, open_interest, open_interest_value}`.
- **Universe:** enumerate `allPerpMetas` (all dexs, incl. `isDelisted`) + `spotMeta`;
  preserve delisted coins to avoid survivorship.

**Reuse:** the existing `DatasetKey`/`write_partition`/catalog, the resumable
`scripts/backfill_history.py` pattern (state file + retries + cut-resilience), and
all `analytics/*`. The HL adapter slots in beside `sources/binance.py`.

**Processing plan:** fills are block-streamed → per-block parquet → partitioned by
`{coin}/{day}` → aggregated to 1m/1h bars in DuckDB/Polars. Deterministic and
re-runnable per day (idempotent checkpoints), so power/internet cuts just resume.

---

## 5. Cost & effort

**Egress is the dominant cost, and it is avoidable — not just reducible.**

Pricing (confirmed): requester-pays = **standard AWS data-transfer-out to the
internet, $0.09/GB**, with **100 GB/month free tier** shared across the account.
The requester also pays for the S3 requests (GET/LIST), which are negligible here.

Let `T` = total compressed bytes of `node_fills_by_block` for 2025→present (and any
auxiliary prefixes). Single-month one-shot egress cost:

```
cost ≈ max(0, T_GB − 100) × $0.09
```

Three ways to pay ≈ $0 for the same data:

| strategy | egress cost | wall-clock | notes |
|---|---|---|---|
| **Pace ≤ 100 GB/month** (local) | **$0** | T/100 months | free tier covers it each month; ideal given the cut-prone setup |
| **Process in-region on EC2** | **~$0 egress** (+ EC2/S3-request) | days | S3→EC2 same-region is not internet DTO; download only the tiny aggregated 1m bars |
| One-shot from local | `max(0, T−100)×0.09` | days | e.g. ~$18 at 300 GB, ~$83 at 1 TB (single month) |

**Volume `T` is unknown until Phase 0** (we cannot list the bucket anonymously).
Rough bracket from HL activity: **~100–500 GB compressed** for fills over
2025→present (order-of-magnitude; must be measured by summing object sizes for a
representative week). Local storage: similar order, ~100–500 GB.

**Recommended:** **two-stage** — run the fills→bars aggregation **in-region**
(Option B's fast path, near-zero egress, download only the aggregated bars), or if
staying fully local, **pace the pull at ≤100 GB/month to stay in the free tier**
(which also suits unreliable power/internet — the checkpointed job just resumes).

Other costs:
- **Funding API:** trivial (~250 coins × ~8,760 hourly rows/yr ⇒ ~4k paginated requests/yr).
- **asset_ctxs:** ~1 small file/day → negligible.
- **Compute:** Polars/DuckDB aggregation; CPU-bound, parallelizable; fits the local lab (or the in-region EC2).
- **Engineering:** new source adapter + fills aggregator + live daemon + backfill
  wiring + tests. **~1–2 weeks** of focused work for a first complete pass, most of
  it in fills normalization and format-era handling.

---

## 6. Risks & limitations

1. **No free archive; egress costs money** and requires AWS credentials.
2. **Completeness caveat:** official docs say updates are ~monthly and "data may be
   missing" → live capture is mandatory for reliable recent data.
3. **Format eras:** `node_fills_by_block` vs older `node_fills` vs `node_trades` —
   need version-aware parsing.
4. **Naming/remapping:** spot uses `@index` (e.g. `@107`); some perps are remapped
   on the UI (`BTC/USDC` ↔ `UBTC`); HIP-3 coins carry `dex:` prefixes. A symbol
   mapping table is required.
5. **HIP-3 proliferation:** builder-deployed dexs add coins continuously; the
   universe is a moving target (handle via `allPerpMetas` snapshots + delistings).
6. **Survivorship:** 56/234 main-dex coins are already delisted — must retain them.
7. **Taker semantics:** verify `crossed`/`side` conventions against a sample before
   trusting CVD.
8. **Rate limits / WS disconnects:** the live daemon needs reconnect + gap backfill.
9. **Daily-only historical OI** (Binance gives 5-min): document the resolution gap.

---

## 7. First steps (Phase 0, read-only, before any build)

1. **Sample the archive:** with AWS read creds, list and download one hour of
   `node_fills_by_block` for one coin and one `asset_ctxs/[date].csv.lz4`; confirm
   schema, taker-side fields, and the fill→bar mapping.
2. **Measure volume:** sum S3 object sizes for a representative week to pin the
   egress budget for 2025→present.
3. **Confirm the universe:** snapshot `allPerpMetas` + `spotMeta` and build the
   full symbol map (incl. delisted, `@index`, HIP-3 prefixes).
4. **Then** implement the adapter + aggregator + resumable backfill (Option B).

---

## 8. Decisions needed

1. **Budget / pacing:** with a 100 GB/month free tier at $0.09/GB, egress can be
   **≈$0** if we either (a) **pace the pull ≤100 GB/month** or (b) **aggregate
   in-region on EC2** and download only the small bars. Which do you prefer — paced
   local pull (free, slow, cut-friendly) or in-region EC2 (fast, near-free, more
   setup)? Falling back to **Option C (vendor)** or **Option D (live-only, no 2025
   history)** only if neither is acceptable.
2. **Scope of "entire universe":** main perp dex only, or also **HIP-3 dexs** and
   **spot** (recommended: all three, with delisted retained)?
3. **Resolution:** is **daily** historical OI acceptable (Binance gave 5-min), or do
   we start the live OI snapshot now and accept daily before that?
4. **Do we also capture L2 book** (`market_data`) for order-flow microstructure, or
   is fills-derived CVD enough?
5. **Ownership of the live daemon:** run continuously on this machine, or capture
   only when online (accepting gaps filled later from S3)?
