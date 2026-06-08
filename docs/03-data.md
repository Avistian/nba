# 3. Data & Data Preparation

To train an NBA bandit, your data must be structured as a sequence of **logged bandit
feedback** events — not a flat label table. Each event is a tuple:

$$\big(\;x \;(\text{context}),\;\; a\;(\text{action}),\;\; r\;(\text{reward}),\;\; p\;(\text{propensity})\;\big)$$

The system needs three data families to assemble these tuples:

1. **Prospect / property data** — estimate the latent quality and purchasing power of each
   household or business (feeds the *context* and the *reward model*).
2. **Environment data** — time, weather, neighborhood density, recent neighbor conversions
   (feeds the *context*).
3. **Geospatial / routing data** — coordinates and road networks for the TSP-P/VRP solver.

> ⚠️ **The Boston Housing dataset is not good enough.** It is tiny (506 rows), ethically
> problematic (it contains a race-derived feature and has been formally deprecated in
> scikit-learn), and has no actions, rewards, or propensities. Use it at most as a 30-minute
> regression warm-up. The datasets in **§3.2** are the real foundation.

---

## 3.1 The Context–Action–Reward–Propensity schema

This is the canonical training schema for D2D NBA. Every door knock (or skip) becomes one row:

| Category | Symbol | Example variables | Purpose |
|----------|--------|-------------------|---------|
| **Prospect context** | $x$ | Property age, roof age, estimated income, tenure, past interactions | State of the target entity. |
| **Environment context** | $x$ | Time of day, day of week, weather, neighborhood density, recent neighbor conversion | External variables that shift conversion. |
| **Action space** | $a$ | Knock Now, Leave Flyer, Skip Door, Pitch Solar, Pitch Security | The discrete choices the algorithm can recommend. |
| **Reward** | $r$ | `0.0` slammed door · `0.2` appointment set · `1.0` closed deal | The numerical optimization target. |
| **Propensity** | $p$ | `0.25` (probability the *logging* system chose this action) | Debiases historical data during **OPE** — **mandatory**. |

### Designing the reward function

Rewards encode business priorities. Keep them simple, monotonic, and stable:

```python
REWARD = {
    "slammed_door":      0.0,
    "not_home":          0.0,
    "info_collected":    0.1,
    "appointment_set":   0.2,
    "closed_deal":       1.0,
}
```

Optionally subtract a small **action cost** (a knock consumes minutes + walking; a flyer is
cheap) so the bandit prefers cheap actions when rewards tie — see
[05-implementation-steps.md](05-implementation-steps.md).

### Why propensity must be logged *now*

The propensity $p$ — the probability the **current** system/rep chose the action that was taken
— is the denominator in Inverse Propensity Scoring. Without it you can never run unbiased OPE,
and **you cannot reconstruct it after the fact**. Log it at decision time from day one
(even if the "policy" is just a rep's habit modeled as a uniform/heuristic distribution).

---

## 3.2 Real datasets to build and practice with

No single public dataset is a ready-made D2D NBA set, so you **compose** one from these
building blocks. Each is real, public, and mapped to the role it plays.

### A. Property / lead-scoring data (the reward-model substrate)

| Dataset | Size / scope | Why it's good | Role |
|---------|--------------|---------------|------|
| **Ames Housing** (Kaggle "House Prices") | 1,460 homes, **79 features** | Far richer than Boston: `YearBuilt`, `RoofStyle`/`RoofMatl`, `OverallQual`, `Neighborhood`, lot size, remodel date — directly relevant to solar/roofing/home-improvement targeting. | Train property-value & roof-age proxies for context. |
| **US Census ACS** (American Community Survey) | National, block-group granularity | Income, home age, tenure, household composition, ownership — the demographic backbone of `estimated_income` and `tenure`. Free, authoritative. | Enrich context for every address. |
| **Microsoft US Building Footprints** | ~130M building polygons | Free geocoded building shapes → roof area, lot density, walkability of a block. | Spatial features + node generation. |
| **OpenAddresses** | Hundreds of millions of addresses | Free address points to materialize the universe of "doors." | Build the prospect universe. |
| **NREL solar datasets** (e.g., PVWatts / rooftop solar potential) | National | Roof solar suitability — a direct reward driver for solar D2D. | Vertical-specific reward signal. |

> Ames replaces Boston as the property warm-up; ACS + Building Footprints + OpenAddresses turn
> it into a realistic, nationwide prospect universe.

### B. Logged-bandit / NBA datasets (learn bandits + OPE for real)

These are the crown jewels — they contain **actions, rewards, AND propensities**, so you can
practice contextual bandits and OPE end-to-end before you ever have D2D logs.

| Dataset | What it is | Why it matters for NBA |
|---------|-----------|------------------------|
| **Open Bandit Dataset (OBD)** + **Open Bandit Pipeline (OBP)** — ZOZO | Real-world logged bandit feedback from a fashion e-commerce platform; a 7-day experiment across 3 campaigns, logged under **Uniform Random** *and* **Bernoulli Thompson Sampling** policies, **with propensities**. | The reference sandbox for **OPE**: OBP ships IPS, DM, and DR estimators and bandit policies. Mirror its `(context, action, reward, pscore)` schema in your D2D pipeline. |
| **Yahoo! Front Page Today Module (R6A/R6B)** — Webscope | Click logs collected under a **uniform-random** policy on news articles. | Uniform-random logging makes **unbiased replay/OPE** straightforward — the classic contextual-bandit benchmark. |
| **Criteo Uplift / Display Advertising** | Large-scale ad logs with treatment/exposure info. | Practice uplift modeling and propensity-weighted evaluation at scale. |
| **RecoGym / RecSim (simulators)** | Configurable RL/bandit environments. | Generate non-stationary and cold-start scenarios on demand to stress-test exploration strategies safely. |

### C. Geospatial / routing data (the TSP-P substrate)

| Dataset / tool | Role |
|----------------|------|
| **OpenStreetMap (OSM)** via **OSMnx** | Real street networks → walk/drive graphs for routing. |
| **OSRM / Valhalla / GraphHopper** | Self-hostable routing engines → real drive-time/distance matrices (avoid Euclidean). |
| **Large synthetic geo-demand sets** (e.g., 10M-row Kaggle geographic demand) | Practice clustering millions of points into walkable territories at scale. |

---

## 3.3 Worked example — assembling one training table

```python
# Conceptual join: build (x, a, r, p) rows from the building blocks
import pandas as pd

doors      = load_openaddresses(city="Austin")          # the door universe
doors      = enrich_with_acs(doors)                      # income, tenure, home age
doors      = enrich_with_footprints(doors)               # roof area, block density
doors      = enrich_with_ames_model(doors)               # predicted property value / roof age

events = (
    spotio_logs                                          # historical knocks
    .merge(doors, on="address_id")
    .assign(
        hour=lambda d: d["timestamp"].dt.hour,
        dow=lambda d: d["timestamp"].dt.dayofweek,
        neighbor_recent_conversion=compute_neighbor_signal,
    )
)

# canonical bandit-feedback columns
events = events[[
    # context (x)
    "property_value", "roof_age", "estimated_income", "tenure", "prior_interactions",
    "hour", "dow", "weather", "block_density", "neighbor_recent_conversion",
    # action (a), reward (r), propensity (p)
    "action", "reward", "propensity",
]]
```

---

## 3.4 (Optional) Boston / Ames as a regression warm-up — and how to clean it

If you use a housing set purely to practice **property-value regression** (the reward-model
substrate), prefer **Ames** over Boston. The cleaning principles below apply to either; the
Boston-specific figures are retained only for the classic exercise.

- Collected by the U.S. Census Service, published **1978**.
- **506 observations** of housing values in Boston suburbs.
- **14 variables**; target = **`MEDV`** (median value of owner-occupied homes, in thousands
  of 1970s dollars).

By predicting `MEDV`, an organization can **synthetically score neighborhoods nationwide** and
direct canvassing teams toward high-value, high-propensity areas.

### Features that map to field-sales targeting

| Feature | Definition | Strategic relevance |
|---------|------------|---------------------|
| **CRIM** | Per-capita crime rate by town | Safety/liability risk for reps; correlates with lower discretionary income. |
| **ZN** | Proportion of residential land zoned for lots > 25,000 sq ft | Larger lots → wealthier, more surface area → great for solar, landscaping, roofing. |
| **AGE** | Proportion of units built before 1940 | Premier indicator for home improvement: HVAC, windows, structural renovations. |
| **DIS** | Weighted distance to 5 Boston employment centers | Proxy for suburban sprawl; drives routing/drive-time parameters. |
| **LSTAT** | % lower-status population | Strong **negative** correlation with target; signals lack of discretionary income. |
| **RM** | Average rooms per dwelling | Larger homes → higher value; contains outliers (see below). |
| **PTRATIO** | Pupil–teacher ratio | Neighborhood quality proxy. |
| **NOX** | Nitric oxide concentration | Environmental/industrial proximity signal. |
| **CHAS** | Charles River dummy (1 if tract bounds river) | Binary; treated as outlier by IQR detection. |
| **TAX** | Property tax rate (per $10,000) | Scales up to ~711; needs normalization. |
| **MEDV** | **Target** — median home value ($000s) | Proxy for homeowner purchasing power. |

### Advanced cleaning & outlier treatment

Raw ingestion **will bias the model**. Required preparation steps:

1. **Outlier detection (IQR method).** Skewed columns and outlier rates:
   - `CRIM` ≈ **13.04%** outliers
   - `ZN` ≈ **13.44%** outliers
   - `RM` ≈ **5.93%** outliers
   - `PTRATIO` ≈ **2.96%** outliers
   - `CHAS` — flagged entirely (binary variable).

2. **Remove the censored target ceiling.** `MEDV` is artificially capped at **50.0**
   (= $50,000). Training on censored data teaches a false upper limit and ruins scoring of
   hyper-premium neighborhoods. **Filter out `MEDV >= 50.0`**, reducing **506 → 490 rows** and
   restoring a normal target distribution.

3. **Normalize features.** Variables span wildly different scales (single-digit `NOX` vs. tax
   rates up to 711). Apply **Min-Max Scaling** so magnitude doesn't bias the model.

```python
# Sketch of the cleaning pipeline (illustrative)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = load_boston_housing()              # 506 rows, 14 cols
df = df[df["MEDV"] < 50.0]              # remove censored ceiling -> 490 rows

features = df.drop(columns=["MEDV"])
target = df["MEDV"]

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
y = target.reset_index(drop=True)
```

> Note: the Boston Housing dataset is used here purely as a **didactic stand-in** for
> property-value scoring. In production, replace it with real, ethically sourced property and
> demographic data, and avoid using protected attributes as targeting features.

---

## 3.5 Geospatial & routing data

To practice the **routing / TSP-P** components, use real street networks (OSM via OSMnx) plus a
routing engine (OSRM/Valhalla) for drive-time matrices, and a large spatial set (e.g., a
synthetic **Geographic Product Demand** dataset with ~**10 million records**) to practice
clustering at scale.

Essential fields for routing optimization:

| Field | Use |
|-------|-----|
| **Latitude / Longitude** | Distance-matrix calculations and clustering. |
| **Expected reward / revenue** | Node *profit* for the TSP-P objective. |
| **Order date timestamp** | Seasonal patterns and **recency** features. |
| **Customer type** | New prospect vs. returning client → decides "cold knock" vs. "account check-in". |

### Preprocessing massive geospatial data

Do **not** feed raw global coordinates into a routing engine. First:

1. **Isolate lat/long columns.**
2. **Apply spatial clustering** (**DBSCAN** or **K-Means**) to segment the national/global
   dataset into **localized, walkable territories**.
3. This ensures OR-Tools only evaluates **feasible, geographically bound** paths — instead of
   wasting compute on impossible cross-continental drive times.

```python
# Sketch: segment prospects into territories before routing
from sklearn.cluster import KMeans

coords = transactions[["latitude", "longitude"]].to_numpy()
territories = KMeans(n_clusters=250, random_state=42).fit_predict(coords)
transactions["territory_id"] = territories
# Route each territory independently downstream (see 05-implementation-steps.md)
```

---

## 3.6 Feature families (the context vector $x$)

The implementation builds a per-prospect **context vector** $x$ from three families, which —
together with the chosen **action** $a$ — feeds the reward model and bandit
(detailed in [05-implementation-steps.md](05-implementation-steps.md)):

| Family | Examples | Source |
|--------|----------|--------|
| **Prospect** | Property value estimate, roof age, estimated income, tenure, prior interactions | Ames-style model + ACS + footprints + CRM |
| **Environment** | Time of day, day of week, weather, block density, recent neighbor conversion | Calendar + weather API + geo index |
| **Spatial** | Walking distance from rep's GPS, density of nearby high-reward doors | Live GPS + geo index |

The action $a$, reward $r$, and propensity $p$ complete each logged tuple (see §3.1).

---

## 3.7 Data quality checklist

- [ ] **Log a propensity $p$ for every action** at decision time (non-negotiable for OPE).
- [ ] Define a stable, monotonic reward function; version it.
- [ ] Calibrate reward-model scores (Platt / isotonic).
- [ ] Detect and treat outliers (IQR) on skewed numeric columns.
- [ ] Validate lat/long ranges and drop invalid coordinates.
- [ ] Cluster coordinates into walkable territories before routing.
- [ ] Engineer recency/frequency features from timestamps.
- [ ] **Exclude protected/discriminatory attributes** (race, etc.) from targeting logic; the
      deprecated Boston `B` feature is a cautionary example.
- [ ] Watch for distribution shift (non-stationarity) and schedule retrains.
