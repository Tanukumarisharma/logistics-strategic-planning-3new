"""
Week 3 Task - Advanced Data Analysis and Visualization in Logistics
Simulates a hypothetical logistics dataset, performs EDA and creates visualizations.
Run:  python logistics_analysis.py
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="deep")
OUT = "figs/"

# ------------------------------------------------------------------
# 1. DATA SIMULATION
# ------------------------------------------------------------------
N = 1500
warehouses = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Bengaluru"]
regions = ["North", "South", "East", "West", "Central"]
modes = ["Road", "Rail", "Air"]
priorities = ["Standard", "Express"]

df = pd.DataFrame({
    "shipment_id": [f"SH{1000+i}" for i in range(N)],
    "ship_date": pd.to_datetime("2025-01-01") + pd.to_timedelta(np.random.randint(0, 365, N), unit="D"),
    "warehouse": np.random.choice(warehouses, N, p=[0.24, 0.26, 0.16, 0.18, 0.16]),
    "dest_region": np.random.choice(regions, N),
    "transport_mode": np.random.choice(modes, N, p=[0.55, 0.30, 0.15]),
    "priority": np.random.choice(priorities, N, p=[0.7, 0.3]),
})

# distance depends on mode (air used for longer trips)
base_dist = np.random.gamma(shape=4, scale=220, size=N)           # ~880 km average
df["distance_km"] = np.where(df.transport_mode == "Air", base_dist * 1.6,
                    np.where(df.transport_mode == "Rail", base_dist * 1.2, base_dist)).round(0).clip(50, 3500)

# shipment volume (kg): right-skewed
df["shipment_kg"] = np.random.lognormal(mean=5.3, sigma=0.7, size=N).round(1).clip(10, 5000)

# warehouse processing time (hours) -- Mumbai & Delhi congested
proc_mean = {"Delhi": 7.5, "Mumbai": 9.0, "Kolkata": 5.5, "Chennai": 5.0, "Bengaluru": 4.5}
df["processing_hrs"] = [max(1, np.random.normal(proc_mean[w], 1.8)) for w in df.warehouse]
df["processing_hrs"] = df.processing_hrs.round(1)

# speed by mode (km/h effective)
speed = {"Road": 38, "Rail": 30, "Air": 450}
df["transit_hrs"] = df.apply(lambda r: r.distance_km / speed[r.transport_mode]
                             + np.random.normal(0, 2), axis=1)
# seasonal monsoon delay (Jul-Sep) for road & rail
df["month"] = df.ship_date.dt.month
monsoon = df.month.isin([7, 8, 9]) & df.transport_mode.isin(["Road", "Rail"])
df["monsoon_factor"] = 1.0
df.loc[monsoon, "monsoon_factor"] = np.random.uniform(1.15, 1.40, monsoon.sum())
df["transit_hrs"] *= df["monsoon_factor"]
df["is_monsoon"] = monsoon.astype(int)
# festive season peak load (Oct-Dec) adds handling delay
festive = df.month.isin([10, 11, 12])
df.loc[festive, "processing_hrs"] *= np.random.uniform(1.10, 1.30, festive.sum())
# Monday backlog: weekend orders pile up and are processed on Monday
df["day_of_week"] = df.ship_date.dt.day_name()
df.loc[df.day_of_week == "Monday", "processing_hrs"] += np.random.uniform(1.0, 2.5, (df.day_of_week == "Monday").sum())
df["transit_hrs"] = df.transit_hrs.clip(lower=1).round(1)
df["processing_hrs"] = df.processing_hrs.round(1)

df["delivery_time_hrs"] = (df.processing_hrs + df.transit_hrs).round(1)

# cost (INR): distance, weight, mode, priority
rate = {"Road": 11, "Rail": 7, "Air": 45}          # INR per km base
kg_rate = {"Road": 2.2, "Rail": 1.5, "Air": 9.0}   # INR per kg
df["transport_cost_inr"] = (
    df.distance_km * df.transport_mode.map(rate) * 0.35 +
    df.shipment_kg * df.transport_mode.map(kg_rate) * 0.6 +
    np.random.normal(0, 400, N)
)
df.loc[df.priority == "Express", "transport_cost_inr"] *= 1.25
df["transport_cost_inr"] = df.transport_cost_inr.clip(lower=300).round(0)
df["cost_per_kg"] = (df.transport_cost_inr / df.shipment_kg).round(2)

# promised SLA = (8 hr handling budget + distance / planned speed) x service factor
planned_speed = {"Road": 40, "Rail": 32, "Air": 500}
df["sla_hrs"] = ((8 + df.distance_km / df.transport_mode.map(planned_speed)) *
                 np.where(df.priority == "Express", 1.00, 1.25)).round(1)
df["delay_hrs"] = (df.delivery_time_hrs - df.sla_hrs).clip(lower=0).round(1)
df["on_time"] = (df.delay_hrs == 0).astype(int)

df.to_csv("logistics_dataset.csv", index=False)

# ------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------
num_cols = ["distance_km", "shipment_kg", "processing_hrs", "transit_hrs",
            "delivery_time_hrs", "transport_cost_inr", "cost_per_kg", "delay_hrs"]
desc = df[num_cols].describe().T
desc["median"] = df[num_cols].median()
desc["skew"] = df[num_cols].skew()
def modal_bin(x, bins=30):
    c, e = np.histogram(x, bins=bins); i = c.argmax(); return (e[i] + e[i+1]) / 2
desc["mode"] = [modal_bin(df[c]) for c in num_cols]   # midpoint of the tallest histogram bin
desc = desc[["count", "mean", "median", "mode", "std", "min", "max", "skew"]].round(2)
desc.to_csv("descriptive_stats.csv")

corr = df[["distance_km", "shipment_kg", "processing_hrs", "transit_hrs",
           "delivery_time_hrs", "transport_cost_inr", "delay_hrs"]].corr().round(2)

by_mode = df.groupby("transport_mode").agg(
    shipments=("shipment_id", "count"), avg_delivery=("delivery_time_hrs", "mean"),
    avg_cost=("transport_cost_inr", "mean"), avg_cost_kg=("cost_per_kg", "mean"),
    on_time=("on_time", "mean")).round(2)
by_wh = df.groupby("warehouse").agg(
    shipments=("shipment_id", "count"), avg_proc=("processing_hrs", "mean"),
    avg_delivery=("delivery_time_hrs", "mean"), on_time=("on_time", "mean")).round(2)
by_region = df.groupby("dest_region").agg(
    avg_delivery=("delivery_time_hrs", "mean"), on_time=("on_time", "mean")).round(2)
monthly = df.groupby("month").agg(
    shipments=("shipment_id", "count"), avg_delivery=("delivery_time_hrs", "mean"),
    avg_proc=("processing_hrs", "mean"), on_time=("on_time", "mean"),
    avg_cost=("transport_cost_inr", "mean")).round(2)
by_pri = df.groupby("priority").agg(on_time=("on_time", "mean"),
                                    avg_delivery=("delivery_time_hrs", "mean"),
                                    avg_cost=("transport_cost_inr", "mean")).round(2)

# outliers via IQR on cost
q1, q3 = df.transport_cost_inr.quantile([.25, .75]); iqr = q3 - q1
n_out = int(((df.transport_cost_inr < q1 - 1.5*iqr) | (df.transport_cost_inr > q3 + 1.5*iqr)).sum())

# ------------------------------------------------------------------
# 3. VISUALIZATIONS
# ------------------------------------------------------------------
# Fig 1: distribution of delivery time
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.histplot(df.delivery_time_hrs, bins=40, kde=True, ax=ax[0], color="#2E75B6")
ax[0].axvline(df.delivery_time_hrs.mean(), color="red", ls="--", label=f"Mean {df.delivery_time_hrs.mean():.1f}")
ax[0].axvline(df.delivery_time_hrs.median(), color="green", ls=":", label=f"Median {df.delivery_time_hrs.median():.1f}")
ax[0].set(title="Distribution of Delivery Time", xlabel="Delivery time (hours)", ylabel="Shipments"); ax[0].legend()
sns.histplot(df.transport_cost_inr, bins=40, kde=True, ax=ax[1], color="#C55A11")
ax[1].axvline(df.transport_cost_inr.mean(), color="red", ls="--", label=f"Mean ₹{df.transport_cost_inr.mean():,.0f}")
ax[1].axvline(df.transport_cost_inr.median(), color="green", ls=":", label=f"Median ₹{df.transport_cost_inr.median():,.0f}")
ax[1].set(title="Distribution of Transportation Cost", xlabel="Cost (INR)", ylabel="Shipments"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT + "fig1_distributions.png", dpi=150); plt.close()

# Fig 2: delivery time & cost by mode (boxplots)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.boxplot(data=df, x="transport_mode", y="delivery_time_hrs", order=modes, ax=ax[0], hue="transport_mode", legend=False)
ax[0].set(title="Delivery Time by Transport Mode", xlabel="Mode", ylabel="Hours")
sns.boxplot(data=df, x="transport_mode", y="transport_cost_inr", order=modes, ax=ax[1], hue="transport_mode", legend=False)
ax[1].set(title="Transportation Cost by Transport Mode", xlabel="Mode", ylabel="Cost (INR)")
plt.tight_layout(); plt.savefig(OUT + "fig2_mode_boxplots.png", dpi=150); plt.close()

# Fig 3: correlation heatmap
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix of Key Logistics Variables")
plt.tight_layout(); plt.savefig(OUT + "fig3_correlation.png", dpi=150); plt.close()

# Fig 4: distance vs cost scatter by mode
plt.figure(figsize=(8.5, 5))
sns.scatterplot(data=df, x="distance_km", y="transport_cost_inr", hue="transport_mode", size="shipment_kg",
                sizes=(10, 120), alpha=.6)
plt.title("Distance vs Transportation Cost (bubble size = shipment weight)")
plt.xlabel("Distance (km)"); plt.ylabel("Cost (INR)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout(); plt.savefig(OUT + "fig4_distance_cost.png", dpi=150); plt.close()

# Fig 5: monthly trend (dual axis)
fig, ax1 = plt.subplots(figsize=(9.5, 4.5))
ax1.plot(monthly.index, monthly.avg_delivery, marker="o", color="#2E75B6", label="Avg delivery time (hrs)")
ax1.set_xlabel("Month"); ax1.set_ylabel("Avg delivery time (hrs)", color="#2E75B6"); ax1.set_xticks(range(1, 13))
ax2 = ax1.twinx()
ax2.bar(monthly.index, monthly.on_time * 100, alpha=.25, color="#548235", label="On-time %")
ax2.set_ylabel("On-time delivery (%)", color="#548235"); ax2.set_ylim(0, 100); ax2.grid(False)
plt.title("Monthly Trend: Delivery Time vs On-Time Rate")
fig.tight_layout(); plt.savefig(OUT + "fig5_monthly_trend.png", dpi=150); plt.close()

# Fig 6: warehouse bottleneck - processing time
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
order = by_wh.sort_values("avg_proc", ascending=False).index
sns.boxplot(data=df, x="warehouse", y="processing_hrs", order=order, ax=ax[0], hue="warehouse", legend=False)
ax[0].set(title="Warehouse Processing Time", xlabel="Warehouse", ylabel="Hours")
sns.barplot(data=df, x="warehouse", y=df.on_time * 100, order=order, ax=ax[1], errorbar=None, hue="warehouse", legend=False)
ax[1].set(title="On-Time Delivery Rate by Warehouse", xlabel="Warehouse", ylabel="On-time (%)")
plt.tight_layout(); plt.savefig(OUT + "fig6_warehouse.png", dpi=150); plt.close()

# Fig 7: heatmap - on-time % by warehouse x mode
pivot = df.pivot_table(index="warehouse", columns="transport_mode", values="on_time", aggfunc="mean") * 100
plt.figure(figsize=(6.5, 4.5))
sns.heatmap(pivot[modes], annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100, linewidths=.5)
plt.title("On-Time Delivery % — Warehouse × Mode")
plt.tight_layout(); plt.savefig(OUT + "fig7_heatmap_ontime.png", dpi=150); plt.close()

# Fig 8: cost per kg by mode & weight band
df["weight_band"] = pd.cut(df.shipment_kg, [0, 100, 250, 500, 1000, 5000],
                           labels=["<100", "100-250", "250-500", "500-1000", ">1000"])
plt.figure(figsize=(8.5, 4.5))
sns.barplot(data=df, x="weight_band", y="cost_per_kg", hue="transport_mode", hue_order=modes, errorbar=None)
plt.title("Cost per kg by Weight Band and Mode (economies of scale)")
plt.xlabel("Shipment weight band (kg)"); plt.ylabel("Cost per kg (INR)")
plt.tight_layout(); plt.savefig(OUT + "fig8_cost_per_kg.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 3b. ADVANCED ANALYSIS: tests, regression, Pareto, scenarios
# ------------------------------------------------------------------
from scipy import stats

adv = {}
# (a) Is Mumbai's processing time really higher than Bengaluru's?  (Welch t-test)
t, p = stats.ttest_ind(df[df.warehouse == "Mumbai"].processing_hrs,
                       df[df.warehouse == "Bengaluru"].processing_hrs, equal_var=False)
adv["ttest_mum_blr"] = [round(t, 2), float(p)]
# (b) Do warehouses differ overall? (one-way ANOVA)
F, pa = stats.f_oneway(*[g.processing_hrs.values for _, g in df.groupby("warehouse")])
adv["anova_wh"] = [round(F, 1), float(pa)]
# (c) Monsoon effect on Road delivery time (Welch t-test)
road = df[df.transport_mode == "Road"]
t2, p2 = stats.ttest_ind(road[road.is_monsoon == 1].delivery_time_hrs,
                         road[road.is_monsoon == 0].delivery_time_hrs, equal_var=False)
adv["ttest_monsoon"] = [round(t2, 2), float(p2),
                        round(road[road.is_monsoon == 1].delivery_time_hrs.mean(), 1),
                        round(road[road.is_monsoon == 0].delivery_time_hrs.mean(), 1)]
# (d) Express vs Standard on-time (chi-square)
ct = pd.crosstab(df.priority, df.on_time)
chi, pc, _, _ = stats.chi2_contingency(ct)
adv["chi_priority"] = [round(chi, 1), float(pc)]
# (e) Spearman correlation (robust to skew): distance vs cost, within each mode
adv["spearman_dist_cost"] = {m: round(stats.spearmanr(g.distance_km, g.transport_cost_inr)[0], 2)
                             for m, g in df.groupby("transport_mode")}
adv["pearson_dist_cost_within"] = {m: round(g[["distance_km", "transport_cost_inr"]].corr().iloc[0, 1], 2)
                                   for m, g in df.groupby("transport_mode")}

# (f) OLS regression: cost drivers
X = pd.DataFrame({"intercept": 1.0, "distance_km": df.distance_km, "shipment_kg": df.shipment_kg,
                  "mode_Rail": (df.transport_mode == "Rail").astype(float),
                  "mode_Air": (df.transport_mode == "Air").astype(float),
                  "express": (df.priority == "Express").astype(float)})
y = df.transport_cost_inr.values
beta, *_ = np.linalg.lstsq(X.values, y, rcond=None)
pred = X.values @ beta
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
n, k = X.shape
sigma2 = ((y - pred) ** 2).sum() / (n - k)
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.values.T @ X.values)))
adv["ols"] = [[c, round(b, 2), round(e, 2), round(b / e, 1)] for c, b, e in zip(X.columns, beta, se)]
adv["ols_r2"] = round(r2, 3)

# (g) Delivery-time decomposition by mode (processing vs transit)
decomp = df.groupby("transport_mode")[["processing_hrs", "transit_hrs"]].mean().round(1)
decomp["proc_share_pct"] = (decomp.processing_hrs / (decomp.processing_hrs + decomp.transit_hrs) * 100).round(1)
adv["decomp"] = decomp.reset_index().values.tolist()

# (h) Delay Pareto: which segments generate the most delay hours?
df["segment"] = df.transport_mode + " | " + df.warehouse
seg = df.groupby("segment").agg(ships=("shipment_id", "count"), delay_total=("delay_hrs", "sum"),
                                late=("on_time", lambda x: (1 - x).sum())).sort_values("delay_total", ascending=False)
seg["share_pct"] = (seg.delay_total / seg.delay_total.sum() * 100).round(1)
seg["cum_pct"] = seg.share_pct.cumsum().round(1)
adv["pareto_top"] = seg.head(6).reset_index().round(1).values.tolist()
adv["pareto_n_for_80"] = int((seg.cum_pct < 80).sum() + 1)
adv["pareto_n_segments"] = len(seg)
adv["total_delay_hrs"] = round(df.delay_hrs.sum(), 0)
adv["late_n"] = int((1 - df.on_time).sum())
mode_delay = df.groupby("transport_mode").agg(share_ships=("shipment_id", "count"), delay=("delay_hrs", "sum"))
mode_delay["ship_pct"] = (mode_delay.share_ships / N * 100).round(1)
mode_delay["delay_pct"] = (mode_delay.delay / mode_delay.delay.sum() * 100).round(1)
adv["mode_delay"] = mode_delay.reset_index().values.tolist()

# (i) Day-of-week effect
dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
dow = df.groupby("day_of_week").agg(n=("shipment_id", "count"), proc=("processing_hrs", "mean"),
                                    on_time=("on_time", "mean")).reindex(dow_order).round(2)
adv["dow"] = dow.reset_index().values.tolist()

# (j) Monsoon / festive season summary
season = pd.cut(df.month, [0, 6, 9, 12], labels=["Jan-Jun (normal)", "Jul-Sep (monsoon)", "Oct-Dec (festive)"])
sea = df.groupby(season, observed=True).agg(n=("shipment_id", "count"), deliv=("delivery_time_hrs", "mean"),
                                           proc=("processing_hrs", "mean"), on_time=("on_time", "mean")).round(2)
adv["season"] = sea.reset_index().values.tolist()

# (k) What-if scenarios (counterfactual, same shipments)
def on_time_rate(d):  # d must have delivery_time_hrs and sla_hrs
    return round(((d.delivery_time_hrs <= d.sla_hrs).mean()) * 100, 1)
base_ot = on_time_rate(df)
# S1: bring Mumbai & Delhi processing down to the average of the other three warehouses
target = df[~df.warehouse.isin(["Mumbai", "Delhi"])].processing_hrs.mean()
s1 = df.copy(); m = s1.warehouse.isin(["Mumbai", "Delhi"])
s1.loc[m, "processing_hrs"] = (s1.loc[m, "processing_hrs"] - (s1.loc[m].groupby("warehouse").processing_hrs.transform("mean") - target)).clip(lower=1)
s1["delivery_time_hrs"] = s1.processing_hrs + s1.transit_hrs
adv["s1"] = [round(target, 2), base_ot, on_time_rate(s1),
             round(df.loc[m, "processing_hrs"].mean(), 2), round(s1.loc[m, "processing_hrs"].mean(), 2)]
# S2: shift Rail shipments under 900 km to Road (faster, cost difference estimated by OLS betas)
s2 = df.copy(); r = (s2.transport_mode == "Rail") & (s2.distance_km < 900)
road_t = s2.loc[r, "distance_km"] / 38 * s2.loc[r, "monsoon_factor"]
s2.loc[r, "transit_hrs"] = road_t
s2.loc[r, "delivery_time_hrs"] = s2.loc[r, "processing_hrs"] + road_t
s2.loc[r, "sla_hrs"] = ((8 + s2.loc[r, "distance_km"] / 40) * np.where(s2.loc[r, "priority"] == "Express", 1.00, 1.25))
d_cost_per_ship = (0.35 * (11 - 7) * df.loc[r, "distance_km"] + 0.6 * (2.2 - 1.5) * df.loc[r, "shipment_kg"]).mean()
adv["s2"] = [int(r.sum()), base_ot, on_time_rate(s2), round(df.loc[r, "delivery_time_hrs"].mean(), 1),
             round(s2.loc[r, "delivery_time_hrs"].mean(), 1), round(d_cost_per_ship, 0)]
# S3: Monday backlog removal (extra staffing on Monday)
s3 = df.copy(); mon = s3.day_of_week == "Monday"
gap = df[mon].processing_hrs.mean() - df[~mon].processing_hrs.mean()
s3.loc[mon, "processing_hrs"] = (s3.loc[mon, "processing_hrs"] - gap).clip(lower=1)
s3["delivery_time_hrs"] = s3.processing_hrs + s3.transit_hrs
adv["s3"] = [round(gap, 2), base_ot, on_time_rate(s3)]
# S4: both S1 + S3 combined
s4 = s1.copy(); mon4 = s4.day_of_week == "Monday"
s4.loc[mon4, "processing_hrs"] = (s4.loc[mon4, "processing_hrs"] - gap).clip(lower=1)
s4["delivery_time_hrs"] = s4.processing_hrs + s4.transit_hrs
adv["s4"] = [base_ot, on_time_rate(s4)]

# worked example row: pick a Rail shipment from Mumbai in monsoon
ex = df[(df.warehouse == "Mumbai") & (df.transport_mode == "Rail") & (df.is_monsoon == 1) & (df.on_time == 0)].iloc[0]
adv["example"] = {k: (str(v) if k == "ship_date" else v) for k, v in
                  ex[["shipment_id", "ship_date", "warehouse", "dest_region", "transport_mode", "priority",
                      "distance_km", "shipment_kg", "processing_hrs", "transit_hrs", "monsoon_factor",
                      "delivery_time_hrs", "sla_hrs", "delay_hrs", "transport_cost_inr"]].to_dict().items()}
adv["example"] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in adv["example"].items()}

# ---- Extra figures ----
# Fig 9: Pareto of delay hours by Mode|Warehouse segment
top = seg.head(10)
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.bar(range(len(top)), top.delay_total, color="#2E75B6")
ax.set_xticks(range(len(top))); ax.set_xticklabels(top.index, rotation=35, ha="right", fontsize=8)
ax.set_ylabel("Total delay hours")
ax2 = ax.twinx(); ax2.plot(range(len(top)), top.cum_pct, color="#C00000", marker="o"); ax2.set_ylim(0, 105)
ax2.set_ylabel("Cumulative % of all delay hours"); ax2.grid(False)
ax2.axhline(80, color="grey", ls="--", lw=.8)
ax.set_title("Pareto Chart: Top 10 Mode | Warehouse Segments by Delay Hours")
plt.tight_layout(); plt.savefig(OUT + "fig9_pareto.png", dpi=150); plt.close()

# Fig 10: stacked bar processing vs transit by mode
fig, ax = plt.subplots(figsize=(7.5, 4.3))
dd = decomp.loc[modes]
ax.bar(dd.index, dd.processing_hrs, label="Warehouse processing", color="#C55A11")
ax.bar(dd.index, dd.transit_hrs, bottom=dd.processing_hrs, label="Transit", color="#2E75B6")
for i, (p_, t_, sh) in enumerate(zip(dd.processing_hrs, dd.transit_hrs, dd.proc_share_pct)):
    ax.text(i, p_ + t_ + 0.8, f"{sh:.0f}% processing", ha="center", fontsize=9)
ax.set(title="Where Does Delivery Time Go? Processing vs Transit", ylabel="Average hours"); ax.legend()
plt.tight_layout(); plt.savefig(OUT + "fig10_decomposition.png", dpi=150); plt.close()

# Fig 11: Day of week processing time & on-time rate
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.barplot(x=dow.index, y=dow.proc, ax=ax[0], color="#2E75B6", order=dow_order)
ax[0].set(title="Avg Processing Time by Dispatch Day", xlabel="", ylabel="Hours"); ax[0].tick_params(axis="x", rotation=30)
sns.barplot(x=dow.index, y=dow.on_time * 100, ax=ax[1], color="#548235", order=dow_order)
ax[1].set(title="On-Time Rate by Dispatch Day", xlabel="", ylabel="On-time (%)", ylim=(50, 100)); ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig(OUT + "fig11_dayofweek.png", dpi=150); plt.close()

# ------------------------------------------------------------------
# 4. SAVE RESULTS FOR REPORT
# ------------------------------------------------------------------
res = {
    "n": N,
    "desc": desc.reset_index().values.tolist(),
    "corr": corr.reset_index().values.tolist(),
    "by_mode": by_mode.reset_index().values.tolist(),
    "by_wh": by_wh.reset_index().values.tolist(),
    "by_region": by_region.reset_index().values.tolist(),
    "by_pri": by_pri.reset_index().values.tolist(),
    "monthly": monthly.reset_index().values.tolist(),
    "overall_on_time": round(df.on_time.mean() * 100, 1),
    "n_cost_outliers": n_out,
    "adv": adv,
    "pivot": pivot.round(1).reset_index().values.tolist(),
    "head": df.head(5)[["shipment_id","ship_date","warehouse","dest_region","transport_mode","priority",
                        "distance_km","shipment_kg","processing_hrs","transit_hrs","delivery_time_hrs",
                        "transport_cost_inr","on_time"]].astype(str).values.tolist(),
}
json.dump(res, open("results.json", "w"), default=str, indent=1)
print(desc.to_string()); print(); print(corr.to_string()); print()
print(by_mode.to_string()); print(); print(by_wh.to_string()); print(); print(by_region.to_string()); print()
print(by_pri.to_string()); print(); print(monthly.to_string()); print(); print(pivot.round(1).to_string())
print(json.dumps(adv, indent=1, default=str)); print("Overall on-time:", res["overall_on_time"], "| cost outliers:", n_out)
print(df.isna().sum().sum(), "missing values")
