"""One-command driver for Phase 6 (T6.1-T6.8): SHAP explanation of the final model.

    python scripts/pipeline/run_phase6.py               # fit, compute SHAP, tables, figures
    python scripts/pipeline/run_phase6.py --plots-only  # reuse the saved SHAP values

The explained model is LightGBM with the tuned parameters, refitted on every
land row of all three years (D-035). SHAP is interventional TreeSHAP against a
1,000-row background (D-036), computed for every land row (for the per-cell
map) in parallel worker processes; the summaries use a 10,000-row subset
stratified by year x urban/rest. XGBoost is explained on the same 10,000 rows
as a robustness check. Interaction values use exact path-dependent TreeSHAP,
because shap has no interventional interaction algorithm (D-036). Units are °C
of LST_anom throughout. Language is associational: SHAP describes the fitted
model's response, not a causal effect.
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import pearsonr, spearmanr

from mirageai import config as c
from mirageai import cv
from mirageai import explain as X
from mirageai import models as M
from mirageai import plotting as P
from mirageai import preprocess as pp

ap = argparse.ArgumentParser()
ap.add_argument("--plots-only", action="store_true")
args = ap.parse_args()

t_start = time.perf_counter()
c.ensure_dirs()
df = pp.load_model_table()
cv.require_land_only(df)
FEATS = M.features_for("lgbm")
assert FEATS == M.features_for("xgb")
PARAMS = {n: v["best_params"] for n, v in
          json.loads((c.MODELS / "best_params.json").read_text(encoding="utf-8")).items()}

LABELS = {
    "NDVI": "NDVI", "NDBI": "NDBI (dryness)", "MNDWI": "MNDWI (moisture)",
    "frac_tree": "Tree cover fraction", "frac_shrub": "Shrub fraction",
    "frac_grass": "Grass fraction", "frac_crop": "Cropland fraction",
    "frac_built": "Built-up fraction", "frac_bare": "Bare fraction",
    "frac_water": "Water fraction", "elevation": "Elevation (m)",
    "slope": "Slope (°)", "dist_water_m": "Distance to water (m)",
    X.SPECTRAL_GROUP: "Spectral block (NDVI+NDBI+MNDWI)",
}
GROUP_LABELS = {
    "spectral": "Spectral (NDVI + dryness + moisture)",
    "dry cover": "Dry cover (grass, crop, bare, shrub)",
    "tree cover": "Tree cover", "built": "Built-up",
    "water proximity": "Water proximity", "terrain": "Terrain (elevation, slope)",
}
# Chart tokens (validated categorical order; slots 1-3 pass all-pairs CVD checks).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
SEQ_ORD = LinearSegmentedColormap.from_list("seq_blue_ord", ["#86b6ef", "#2a78d6", "#0d366b"])


def style_axes(ax, grid_axis="y") -> None:
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.6)


# ---------------------------------------------------------------------------
# Rows: explained sample, background, interaction subsample (all deterministic)
# ---------------------------------------------------------------------------
explain_idx = X.stratified_sample(df, X.N_EXPLAIN)
bg_idx = X.stratified_sample(df, X.N_BACKGROUND, seed=c.SEED + 1, exclude=explain_idx)
inter_idx = X.stratified_sample(df.loc[explain_idx], X.N_INTERACTION, seed=c.SEED + 2)
Xe = X.float32_exact(df.loc[explain_idx, FEATS])
Xb = X.float32_exact(df.loc[bg_idx, FEATS])
meta = df.loc[explain_idx, ["cell_id", "year", "is_urban"]]
urban = meta["is_urban"].to_numpy()
shap_path = {n: c.TABLES / f"phase6_shap_{n}.parquet" for n in ("lgbm", "xgb")}
inter_path = c.TABLES / "phase6_interaction_values_lgbm.npz"
cells_path = c.TABLES / "phase6_shap_cells_lgbm.parquet"

# ---------------------------------------------------------------------------
# T6.1 Final models and SHAP values
# ---------------------------------------------------------------------------
if not args.plots_only:
    resources = []
    for name in ("lgbm", "xgb"):
        with M.PeakMemory() as mem:
            t0 = time.perf_counter()
            model = X.fit_final(df, name, PARAMS[name])
            fit_s = time.perf_counter() - t0
            joblib.dump({"model": model, "features": FEATS, "params": PARAMS[name],
                         "target": c.TARGET_ANOM, "fitted_on": "all_years_land",
                         "years": list(c.YEARS), "water_excluded": True,
                         "features_rounded": "float32"},
                        c.MODELS / f"{name}_all.joblib", compress=3)
            t0 = time.perf_counter()
            if name == "lgbm":
                # Every land row, for the per-cell map; the explained sample is a
                # subset of the same run, so the two cannot disagree.
                Xall = X.float32_exact(df[FEATS])
                all_vals, base = X.interventional_shap(model, Xall, Xb, n_jobs=X.SHAP_N_JOBS)
                vals = all_vals[df.index.get_indexer(explain_idx)]
            else:
                vals, base = X.interventional_shap(model, Xe, Xb, n_jobs=X.SHAP_N_JOBS)
            shap_s = time.perf_counter() - t0
        n_rows = len(df) if name == "lgbm" else len(Xe)
        out = meta.copy()
        out.insert(0, "row_id", explain_idx.to_numpy())
        for j, f in enumerate(FEATS):
            out[f"shap_{f}"] = vals[:, j].astype("float32")
        out["base_value"] = base
        out["prediction"] = model.predict(Xe).astype("float32")
        out.to_parquet(shap_path[name], index=False)
        resources.append({"stage": "fit+interventional", "model": name, "fit_s": round(fit_s, 1),
                          "shap_s": round(shap_s, 1), "peak_rss_mb": round(mem.peak_mb)})
        print(f"{name}: fit {fit_s:.0f}s, interventional SHAP {n_rows:,} rows x "
              f"{X.N_BACKGROUND:,} background in {shap_s / 60:.1f} min, base {base:+.3f}, "
              f"peak {mem.peak_mb:.0f} MB (main process)", flush=True)

        if name == "lgbm":
            cells = pd.DataFrame(all_vals.astype("float32"), columns=[f"shap_{f}" for f in FEATS])
            cells.insert(0, "row_id", df.index.to_numpy())
            cells["cell_id"] = df["cell_id"].to_numpy()
            cells["x_utm"], cells["y_utm"] = df["x_utm"].to_numpy(), df["y_utm"].to_numpy()
            cells["is_urban"] = df["is_urban"].to_numpy()
            cells.to_parquet(cells_path, index=False)
            del all_vals, cells

            Xi = X.float32_exact(df.loc[inter_idx, FEATS])
            with M.PeakMemory() as mem:
                t0 = time.perf_counter()
                inter, _ = X.interaction_values(model, Xi)
                el = time.perf_counter() - t0
            np.savez_compressed(inter_path, values=inter.astype("float32"), row_id=inter_idx.to_numpy())
            resources.append({"stage": "interactions", "model": name, "fit_s": 0,
                              "shap_s": round(el, 1), "peak_rss_mb": round(mem.peak_mb)})
            print(f"  interactions {len(inter_idx):,} rows in {el / 60:.1f} min", flush=True)

            # Diagnostic only: how much does the perturbation choice move the
            # attribution? Path-dependent SHAP on the explained rows.
            pd_vals, _ = X.path_dependent_shap(model, Xe)
            np.save(c.TABLES / "phase6_shap_lgbm_pathdep.npy", pd_vals.astype("float32"))
        del model

    # Linear regression on the same rows, for RQ2: standardised coefficients.
    lin_feats = M.features_for("linear")
    lin = M.make_model("linear").fit(df[lin_feats], df[c.TARGET_ANOM])
    pd.DataFrame({"feature": lin_feats,
                  "coef_per_sd": lin.named_steps["model"].coef_,
                  "feature_mean": lin.named_steps["scaler"].mean_,
                  "feature_sd": lin.named_steps["scaler"].scale_}).to_csv(
        c.table("phase6_linear_coefs"), index=False)
    pd.DataFrame(resources).to_csv(c.table("phase6_runtimes"), index=False)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
S = {n: pd.read_parquet(shap_path[n]) for n in ("lgbm", "xgb")}
for n in S:
    assert (S[n]["row_id"].to_numpy() == explain_idx.to_numpy()).all(), "sample changed"
V = {n: S[n][[f"shap_{f}" for f in FEATS]].to_numpy("float64") for n in S}
lin = pd.read_csv(c.table("phase6_linear_coefs")).set_index("feature")
iz = np.load(inter_path)
inter, inter_rows = iz["values"].astype("float64"), iz["row_id"]
xv = Xe.to_numpy()

# ---------------------------------------------------------------------------
# T6.2 / T6.7 Global importance, all cells vs urban; T6.1 robustness vs XGBoost
# ---------------------------------------------------------------------------
imp = pd.DataFrame({
    "lgbm_all": X.importance(V["lgbm"], FEATS),
    "lgbm_urban": X.importance(V["lgbm"][urban], FEATS),
    "xgb_all": X.importance(V["xgb"], FEATS),
    "xgb_urban": X.importance(V["xgb"][urban], FEATS),
    "lgbm_mean_signed": pd.Series(V["lgbm"].mean(axis=0), index=FEATS),
}).sort_values("lgbm_all", ascending=False)
for col in ("lgbm_all", "lgbm_urban", "xgb_all", "xgb_urban"):
    imp[f"rank_{col}"] = imp[col].rank(ascending=False).astype(int)
imp["share_lgbm_all"] = imp["lgbm_all"] / imp["lgbm_all"].sum()
imp.index.name = "feature"
imp.to_csv(c.table("phase6_shap_importance"))
TOP4 = list(imp.index[:4])

agree = pd.DataFrame([
    {"comparison": "lgbm vs xgb, all cells", **X.rank_agreement(imp.lgbm_all, imp.xgb_all)},
    {"comparison": "lgbm vs xgb, urban", **X.rank_agreement(imp.lgbm_urban, imp.xgb_urban)},
    {"comparison": "lgbm all vs lgbm urban", **X.rank_agreement(imp.lgbm_all, imp.lgbm_urban)},
])
agree.to_csv(c.table("phase6_rank_agreement"), index=False)

print(f"\nT6.2/T6.7 mean |SHAP| (°C), n = {len(Xe):,} ({urban.sum():,} urban)")
print(imp[["lgbm_all", "rank_lgbm_all", "lgbm_urban", "rank_lgbm_urban", "xgb_all",
           "rank_xgb_all", "lgbm_mean_signed"]].round(3).to_string())
print("\nRank agreement\n" + agree.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# T6.8 Grouped spectral SHAP
# ---------------------------------------------------------------------------
g_rows = []
for n in ("lgbm", "xgb"):
    for subset, mask in (("all", np.ones(len(Xe), bool)), ("urban", urban)):
        v = V[n][mask]
        g = X.grouped(v, FEATS)
        members = {f: float(np.abs(v[:, FEATS.index(f)]).mean()) for f in c.SPECTRAL_FEATURES}
        others = {f: float(np.abs(v[:, j]).mean()) for j, f in enumerate(FEATS)
                  if f not in c.SPECTRAL_FEATURES}
        g_abs = float(np.abs(g).mean())
        g_rows.append({"model": n, "subset": subset, "group_mean_abs": g_abs,
                       "sum_of_member_mean_abs": sum(members.values()),
                       "cancellation_share": 1 - g_abs / sum(members.values()),
                       **{f"{f}_mean_abs": m for f, m in members.items()},
                       "group_rank_vs_other_features": 1 + sum(o > g_abs for o in others.values()),
                       "top_other_feature": max(others, key=others.get),
                       "top_other_mean_abs": max(others.values())})
grouped_tab = pd.DataFrame(g_rows)
grouped_tab.to_csv(c.table("phase6_grouped_spectral"), index=False)
print("\nT6.8 grouped spectral SHAP\n" + grouped_tab.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# T6.4 Interactions (path-dependent, 2,000-row stratified subsample)
# ---------------------------------------------------------------------------
imat = X.interaction_matrix(inter, FEATS)
imat.to_csv(c.table("phase6_interaction_matrix"))
pairs = X.top_interactions(inter, FEATS, k=0)
pairs.to_csv(c.table("phase6_interactions"), index=False)
TOP_PAIRS = pairs.head(3)
print("\nT6.4 top interacting pairs (mean |phi_ij + phi_ji|, °C)")
print(pairs.head(6).round(4).to_string(index=False))
off = imat.to_numpy().copy()
np.fill_diagonal(off, -np.inf)
PARTNER = {f: FEATS[int(np.argmax(off[FEATS.index(f)]))] for f in FEATS}

# ---------------------------------------------------------------------------
# T6.3 Thresholds and saturation; RQ2 comparison with linear regression
# ---------------------------------------------------------------------------
# Fractions are zero-inflated: a knot at 1% tree cover is the step from none to
# some, not a threshold. Where at least 20% of cells are exactly zero, the
# step is reported separately and the breakpoint is refitted on positive values.
ZERO_INFLATED = 0.2
th_rows = []
for j, f in enumerate(FEATS):
    x, s = xv[:, j], V["lgbm"][:, j]
    bp = X.breakpoint(x, s)
    sd = float(df[f].std())
    slope_lin = float(np.polyfit(x, s, 1)[0])
    zero_share = float((x == 0).mean())
    zi = {"zero_share": zero_share}
    if zero_share >= ZERO_INFLATED:
        pos = x > 0
        low_edge = float(np.quantile(x[pos], 0.2))
        zi["low_positive_upper"] = low_edge
        zi["step_zero_to_low"] = float(s[pos & (x <= low_edge)].mean() - s[~pos].mean())
        zi.update({f"pos_{k}": v for k, v in X.breakpoint(x[pos], s[pos]).items()})
    th_rows.append({"feature": f, **bp, **zi,
                    "slope_below_per_sd": bp["slope_below"] * sd,
                    "slope_above_per_sd": bp["slope_above"] * sd,
                    "tree_linearised_per_sd": slope_lin * sd,
                    "linear_coef_per_sd": float(lin.coef_per_sd.get(f, np.nan)),
                    "shap_range_p1_p99": float(np.quantile(s, 0.99) - np.quantile(s, 0.01)),
                    "feature_sd": sd, "top_partner": PARTNER[f]})
thresholds = pd.DataFrame(th_rows).set_index("feature").loc[imp.index]
thresholds.to_csv(c.table("phase6_thresholds"))
print("\nT6.3 segmented fits of SHAP on feature value (knot with 95% bootstrap CI)")
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(thresholds[["knot", "knot_lo", "knot_hi", "slope_below", "slope_above", "r2_linear",
                      "r2_segmented", "shape", "tree_linearised_per_sd", "linear_coef_per_sd",
                      "top_partner"]].round(4).to_string())
    zcols = [k for k in ("zero_share", "low_positive_upper", "step_zero_to_low", "pos_knot",
                         "pos_knot_lo", "pos_knot_hi", "pos_slope_below", "pos_slope_above",
                         "pos_r2_linear", "pos_r2_segmented", "pos_shape") if k in thresholds]
    print("\nZero-inflated features: step at zero, then breakpoint on positive values")
    print(thresholds.loc[thresholds.zero_share >= ZERO_INFLATED, zcols].round(4).to_string())

# ---------------------------------------------------------------------------
# RQ5 frac_built: raw association vs the model's conditional response
# ---------------------------------------------------------------------------
jb = FEATS.index("frac_built")
fb_x, fb_s = xv[:, jb], V["lgbm"][:, jb]
rq5 = []
for subset, d_mask, s_mask in (("all", np.ones(len(df), bool), np.ones(len(Xe), bool)),
                               ("urban", df["is_urban"].to_numpy(), urban)):
    d = df[d_mask]
    rq5.append({
        "subset": subset, "n_rows": int(d_mask.sum()), "n_sample": int(s_mask.sum()),
        "raw_pearson": float(pearsonr(d.frac_built, d[c.TARGET_ANOM]).statistic),
        "raw_spearman": float(spearmanr(d.frac_built, d[c.TARGET_ANOM]).statistic),
        "raw_slope_per_0.1": float(np.polyfit(d.frac_built, d[c.TARGET_ANOM], 1)[0] / 10),
        "shap_spearman": float(spearmanr(fb_x[s_mask], fb_s[s_mask]).statistic),
        "shap_slope_per_0.1": float(np.polyfit(fb_x[s_mask], fb_s[s_mask], 1)[0] / 10),
        "shap_mean_abs": float(np.abs(fb_s[s_mask]).mean()),
        "shap_mean_signed": float(fb_s[s_mask].mean()),
        "shap_rank": int(X.importance(V["lgbm"][s_mask], FEATS).index.get_loc("frac_built") + 1),
    })
bins = [0, 0.1, 0.25, 0.5, 0.75, 1.0001]
fb_bins = (pd.DataFrame({"frac_built": fb_x, "shap": fb_s, "urban": urban})
           .assign(bin=lambda t: pd.cut(t.frac_built, bins, right=False))
           .groupby("bin", observed=True)["shap"].agg(["size", "mean", "std"]))
fb_bins.to_csv(c.table("phase6_rq5_frac_built_bins"))
rq5 = pd.DataFrame(rq5)
rq5.to_csv(c.table("phase6_rq5_frac_built"), index=False)
print("\nRQ5 frac_built: raw association vs SHAP response\n" + rq5.round(4).to_string(index=False))
print(fb_bins.round(3).to_string())

# ---------------------------------------------------------------------------
# T6.5 Dominant driver per cell (interventional SHAP, mean over the 3 years)
# ---------------------------------------------------------------------------
cells = pd.read_parquet(cells_path)
scols = [f"shap_{f}" for f in FEATS]
assert np.allclose(cells.set_index("row_id").loc[explain_idx, scols].to_numpy(), V["lgbm"])
# Diagnostic: how often would path-dependent SHAP pick the same dominant driver?
pd_rows = np.load(c.TABLES / "phase6_shap_lgbm_pathdep.npy").astype("float64")
dom_int = np.array(FEATS)[np.abs(V["lgbm"]).argmax(axis=1)]
dom_pd = np.array(FEATS)[np.abs(pd_rows).argmax(axis=1)]
agree_dom = float((dom_int == dom_pd).mean())
pd_imp = X.importance(pd_rows, FEATS)
pd_vs_int = X.rank_agreement(imp.lgbm_all, pd_imp)

per_cell = cells.groupby("cell_id").agg({**{s: "mean" for s in scols}, "x_utm": "first",
                                         "y_utm": "first", "is_urban": "first"})
cell_vals = per_cell[scols].to_numpy()
GROUPS = list(X.DRIVER_GROUPS)


def dominance(vals: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Dominant driver per cell, and the share of cells each one dominates."""
    k = np.abs(vals).argmax(axis=1)
    frame = pd.DataFrame({"dominant": np.array(names)[k],
                          "sign": np.sign(vals[np.arange(len(vals)), k]),
                          "is_urban": per_cell["is_urban"].to_numpy()})
    out = (frame.groupby("dominant")
           .agg(cells=("dominant", "size"), urban_cells=("is_urban", "sum"),
                share_warming=("sign", lambda s: float((s > 0).mean())))
           .sort_values("cells", ascending=False))
    out["share"] = out.cells / out.cells.sum()
    out["share_of_urban"] = out.urban_cells / out.urban_cells.sum()
    return out, frame["dominant"].to_numpy()


# Headline: driver groups. Supplementary: individual features.
dom_g, per_cell["dominant_group"] = dominance(X.group_values(cell_vals, FEATS), GROUPS)
dom, per_cell["dominant"] = dominance(cell_vals, FEATS)
dom_g.to_csv(c.table("phase6_dominant_driver_groups"))
dom.to_csv(c.table("phase6_dominant_driver"))

# Stability of the dominant driver across models and SHAP methods, at both levels.
g_int, g_xgb, g_pd = (X.group_values(v, FEATS) for v in (V["lgbm"], V["xgb"], pd_rows))
argmax_name = lambda v, names: np.array(names)[np.abs(v).argmax(axis=1)]  # noqa: E731
stability = pd.DataFrame([
    {"level": "feature", "lgbm_vs_xgb": float((argmax_name(V["lgbm"], FEATS) == argmax_name(V["xgb"], FEATS)).mean()),
     "interventional_vs_path_dependent": agree_dom},
    {"level": "group", "lgbm_vs_xgb": float((argmax_name(g_int, GROUPS) == argmax_name(g_xgb, GROUPS)).mean()),
     "interventional_vs_path_dependent": float((argmax_name(g_int, GROUPS) == argmax_name(g_pd, GROUPS)).mean())},
])
stability.to_csv(c.table("phase6_dominant_stability"), index=False)
group_imp = pd.DataFrame({
    "lgbm_all": np.abs(g_int).mean(axis=0), "lgbm_urban": np.abs(g_int[urban]).mean(axis=0),
    "xgb_all": np.abs(g_xgb).mean(axis=0), "xgb_urban": np.abs(g_xgb[urban]).mean(axis=0),
    "lgbm_path_dependent": np.abs(g_pd).mean(axis=0)}, index=pd.Index(GROUPS, name="group"),
).sort_values("lgbm_all", ascending=False)
group_imp.to_csv(c.table("phase6_group_importance"))
pd.DataFrame([{"rows": len(explain_idx), "dominant_agreement": agree_dom, **pd_vs_int}]).to_csv(
    c.table("phase6_path_vs_interventional"), index=False)
print(f"\nT6.5 dominant driver GROUP per cell ({len(per_cell):,} cells)\n{dom_g.round(3).to_string()}")
print(f"\n  supplementary, per feature:\n{dom.round(3).to_string()}")
print(f"\n  dominant-driver stability on the explained rows\n{stability.round(3).to_string(index=False)}")
print(f"\n  group importance (mean |grouped SHAP|, °C)\n{group_imp.round(3).to_string()}")
print("  path-dependent importance:", pd_imp.round(3).to_dict())

# ===========================================================================
# Figures
# ===========================================================================
import shap  # noqa: E402

# T6.2 beeswarm
expl = shap.Explanation(values=V["lgbm"], base_values=S["lgbm"]["base_value"].to_numpy(),
                        data=xv, feature_names=[LABELS[f] for f in FEATS])
plt.figure()
np.random.seed(c.SEED)  # beeswarm jitters points with the global RNG
shap.plots.beeswarm(expl, max_display=len(FEATS), show=False, color=SEQ_ORD,
                    plot_size=(8.5, 6.2), s=6, alpha=0.6)
fig = plt.gcf()
ax = fig.axes[0]
ax.set_xlabel("SHAP value: contribution to predicted LST anomaly (°C)", fontsize=9, color=INK2)
ax.set_title(f"LightGBM drivers, all land cells (n = {len(Xe):,}, interventional SHAP)",
             fontsize=10, color=INK, loc="left")
ax.tick_params(labelsize=8, colors=INK2)
fig.text(0.01, -0.02, "Each dot is one cell-year. Colour: feature value, light = low, dark = high. "
         "Associational: the fitted model's response, not a causal effect.",
         fontsize=7.5, color=MUTED)
P.save_figure(fig, c.figure(6, "shap_beeswarm"))
plt.close(fig)

# T6.2 / T6.7 bar: all cells vs urban, LightGBM, with XGBoost ranks marked
order = list(imp.index)[::-1]
fig, ax = plt.subplots(figsize=(8, 5.6))
y = np.arange(len(order))
h = 0.38
ax.barh(y + h / 2, imp.loc[order, "lgbm_all"], height=h, color=BLUE, label="All land cells",
        edgecolor="white", linewidth=1)
ax.barh(y - h / 2, imp.loc[order, "lgbm_urban"], height=h, color=ORANGE,
        label="Urban cells (frac_built ≥ 0.5)", edgecolor="white", linewidth=1)
for yi, f in zip(y, order):
    ax.text(imp.loc[f, "lgbm_all"] + 0.01, yi + h / 2, f"{imp.loc[f, 'lgbm_all']:.2f}",
            va="center", fontsize=7, color=INK2)
    ax.text(imp.loc[f, "lgbm_urban"] + 0.01, yi - h / 2, f"{imp.loc[f, 'lgbm_urban']:.2f}",
            va="center", fontsize=7, color=INK2)
ax.set_yticks(y, [f"{LABELS[f]}  (#{imp.loc[f, 'rank_lgbm_all']} / #{imp.loc[f, 'rank_lgbm_urban']})"
                  for f in order], fontsize=8)
ax.set_xlabel("Mean |SHAP| (°C of LST anomaly)", fontsize=9, color=INK2)
ax.set_title("Driver importance, all cells vs urban (LightGBM)", fontsize=10, color=INK, loc="left")
ax.legend(frameon=False, fontsize=8, loc="lower right")
style_axes(ax, grid_axis="x")
fig.text(0.01, -0.02, f"Labels: rank all / rank urban. XGBoost on the same rows: Spearman "
         f"{agree.spearman.iloc[0]:.2f} (all), {agree.spearman.iloc[1]:.2f} (urban).",
         fontsize=7.5, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(6, "shap_bar"))
plt.close(fig)

# T6.8 grouped spectral
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
for ax, subset in zip(axes, ("all", "urban")):
    r = grouped_tab[(grouped_tab.model == "lgbm") & (grouped_tab.subset == subset)].iloc[0]
    names = [LABELS["MNDWI"], LABELS["NDBI"], "NDVI", "sum of the three\n(separately)",
             "spectral block\n(summed per cell)"]
    vals_ = [r.MNDWI_mean_abs, r.NDBI_mean_abs, r.NDVI_mean_abs, r.sum_of_member_mean_abs, r.group_mean_abs]
    cols = [MUTED, MUTED, MUTED, "#c3c2b7", BLUE]
    ax.barh(range(5), vals_, color=cols, height=0.6, edgecolor="white")
    for i, v in enumerate(vals_):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(5), names, fontsize=8)
    ax.set_title(f"{'All land cells' if subset == 'all' else 'Urban cells'}: "
                 f"{r.cancellation_share:.0%} cancels within the block", fontsize=9, color=INK, loc="left")
    ax.set_xlabel("Mean |SHAP| (°C)", fontsize=8.5, color=INK2)
    style_axes(ax, grid_axis="x")
fig.tight_layout()
P.save_figure(fig, c.figure(6, "shap_grouped_spectral"))
plt.close(fig)

# T6.3 dependence, top 4
fig, axes = plt.subplots(2, 2, figsize=(10.5, 8))
for ax, f in zip(axes.ravel(), TOP4):
    j = FEATS.index(f)
    x, s = xv[:, j], V["lgbm"][:, j]
    partner = PARTNER[f]
    pv = xv[:, FEATS.index(partner)]
    lo_, hi_ = np.quantile(pv, [0.02, 0.98])
    sc = ax.scatter(x, s, c=np.clip(pv, lo_, hi_), cmap=SEQ_ORD, s=4, alpha=0.5, linewidths=0,
                    rasterized=True)
    # Binned mean of SHAP (20 quantile bins)
    q = pd.qcut(x, 20, duplicates="drop")
    bm = pd.DataFrame({"x": x, "s": s, "q": q}).groupby("q", observed=True).mean()
    ax.plot(bm.x, bm.s, color=INK, linewidth=1.6, label="binned mean (20 quantiles)")
    t = thresholds.loc[f]
    pre = "pos_" if t.zero_share >= ZERO_INFLATED else ""
    if t[f"{pre}shape"] != "near-linear":
        ax.axvspan(t[f"{pre}knot_lo"], t[f"{pre}knot_hi"], color=ORANGE, alpha=0.15, linewidth=0)
        ax.axvline(t[f"{pre}knot"], color=ORANGE, linewidth=1.2)
        ax.text(t[f"{pre}knot"], ax.get_ylim()[1], f" {t[f'{pre}shape']} at {t[f'{pre}knot']:.3g}"
                + (" (cells > 0)" if pre else ""), color=INK, fontsize=7.5, va="top")
    if f in lin.index:
        xs = np.linspace(np.quantile(x, 0.01), np.quantile(x, 0.99), 50)
        r = lin.loc[f]
        ax.plot(xs, r.coef_per_sd * (xs - r.feature_mean) / r.feature_sd, color=INK2,
                linestyle="--", linewidth=1.1, label="linear regression (same scale)")
    else:
        ax.text(0.98, 0.04, "linear model: reference category (D-023)", transform=ax.transAxes,
                ha="right", fontsize=7, color=MUTED)
    ax.axhline(0, color="#c3c2b7", linewidth=0.6)
    ax.set_xlabel(LABELS[f], fontsize=9, color=INK2)
    ax.set_ylabel("SHAP (°C)", fontsize=9, color=INK2)
    ax.set_title(f"{LABELS[f]}  (mean |SHAP| {imp.loc[f, 'lgbm_all']:.2f} °C)", fontsize=9.5,
                 color=INK, loc="left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(f"{LABELS[partner]} (top interaction)", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    style_axes(ax, grid_axis="both")
axes[0, 0].legend(frameon=False, fontsize=7.5, loc="lower left")
fig.text(0.01, -0.01, "Orange line and band: segmented-regression breakpoint with 95% bootstrap CI. "
         "Dashed: linear regression's contribution on the same centring (°C).",
         fontsize=7.5, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(6, "shap_dependence"))
plt.close(fig)

# T6.4 interactions: matrix + top 3 pairs
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1])
ax = fig.add_subplot(gs[0, :2])
mm = imat.loc[imp.index, imp.index].to_numpy().copy()
np.fill_diagonal(mm, np.nan)
im = ax.imshow(mm, cmap=SEQ, interpolation="nearest")
lab = [LABELS[f] for f in imp.index]
ax.set_xticks(range(len(lab)), lab, rotation=55, ha="right", fontsize=7.5)
ax.set_yticks(range(len(lab)), lab, fontsize=7.5)
for i in range(len(lab)):
    for k in range(len(lab)):
        if i != k and mm[i, k] >= np.nanquantile(mm, 0.9):
            ax.text(k, i, f"{mm[i, k]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if mm[i, k] > np.nanquantile(mm, 0.97) else INK)
ax.set_title("Mean |SHAP interaction| per pair (°C), diagonal omitted", fontsize=10, color=INK, loc="left")
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.ax.tick_params(labelsize=7)
cb.outline.set_linewidth(0.5)
axt = fig.add_subplot(gs[0, 2])
axt.axis("off")
txt = "Top pairs (mean |φij+φji|, °C)\n\n" + "\n".join(
    f"{r['rank']}. {LABELS[r.feature_a]} × {LABELS[r.feature_b]}: {r.mean_abs_interaction:.3f}"
    for _, r in pairs.head(6).iterrows())
axt.text(0, 0.95, txt, va="top", fontsize=8, color=INK, wrap=True)
xi = X.float32_exact(df.loc[inter_rows, FEATS]).to_numpy()
for k, (_, r) in enumerate(TOP_PAIRS.iterrows()):
    a, b = FEATS.index(r.feature_a), FEATS.index(r.feature_b)
    ax = fig.add_subplot(gs[1, k])
    lo_, hi_ = np.quantile(xi[:, b], [0.02, 0.98])
    sc = ax.scatter(xi[:, a], inter[:, a, b] + inter[:, b, a], c=np.clip(xi[:, b], lo_, hi_),
                    cmap=SEQ_ORD, s=5, alpha=0.6, linewidths=0, rasterized=True)
    ax.axhline(0, color="#c3c2b7", linewidth=0.6)
    ax.set_xlabel(LABELS[r.feature_a], fontsize=8.5, color=INK2)
    ax.set_ylabel("Interaction SHAP (°C)", fontsize=8.5, color=INK2)
    ax.set_title(f"#{r['rank']} {LABELS[r.feature_a]} × {LABELS[r.feature_b]}", fontsize=9,
                 color=INK, loc="left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cb.set_label(LABELS[r.feature_b], fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.5)
    style_axes(ax, grid_axis="both")
fig.text(0.01, -0.01, f"Path-dependent SHAP interaction values, LightGBM, {len(inter_rows):,}-row "
         "stratified subsample (shap has no interventional interaction algorithm; D-036).",
         fontsize=7.5, color=MUTED)
fig.tight_layout()
P.save_figure(fig, c.figure(6, "shap_interactions"))
plt.close(fig)

# T6.5 dominant driver maps: the top 3 categories named (the palette's all-pairs
# limit for a map), the rest folded into "Other" with their shares listed.
def dominant_map(column, table, labels, title, note, path) -> None:
    top3 = list(table.index[:3])
    codes = per_cell[column].map({f: i for i, f in enumerate(top3)}).fillna(3).to_numpy()
    grid, extent = P.cells_to_grid(per_cell.x_utm.to_numpy(), per_cell.y_utm.to_numpy(), codes)
    fig, ax = plt.subplots(figsize=(9, 7.4))
    ax.imshow(grid, extent=extent, origin="upper", interpolation="nearest", vmin=-0.5, vmax=3.5,
              cmap=ListedColormap([BLUE, ORANGE, AQUA, "#c3c2b7"]))
    ax.set_aspect("equal")
    ax.set_xlabel("UTM 43N easting (km)", fontsize=9)
    ax.set_ylabel("UTM 43N northing (km)", fontsize=9)
    ax.tick_params(labelsize=8, length=3, width=0.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("0.6")
    rest = table.iloc[3:]
    other = f"Other ({rest.share.sum():.1%})\n" + "\n".join(
        f"   {labels[f]} {r.share:.1%}" for f, r in rest.iterrows())
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in (BLUE, ORANGE, AQUA, "#c3c2b7")]
    names = [f"{labels[f]} ({table.loc[f, 'share']:.1%})" for f in top3] + [other]
    ax.legend(handles, names, frameon=True, framealpha=0.9, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1), title="Largest |SHAP| in the cell", title_fontsize=8)
    ax.set_title(title, fontsize=11, pad=8)
    fig.text(0.01, 0.02, f"Interventional SHAP on all {len(per_cell):,} land cells × 3 years, averaged "
             f"per cell. {note} White: water-majority cells, excluded (D-034).", fontsize=7.5,
             color=MUTED, wrap=True)
    P.save_figure(fig, path)
    plt.close(fig)


st = stability.set_index("level")
dominant_map("dominant_group", dom_g, GROUP_LABELS,
             "Dominant driver group per 100 m cell (LightGBM, mean SHAP over 2023-25)",
             f"Same dominant group under XGBoost on {st.loc['group', 'lgbm_vs_xgb']:.0%} of sampled "
             f"rows (per feature: {st.loc['feature', 'lgbm_vs_xgb']:.0%}).",
             c.figure(6, "dominant_driver_map"))
dominant_map("dominant", dom, LABELS,
             "Supplementary: dominant single feature per cell (LightGBM)",
             f"Unstable within the spectral block: same feature under XGBoost on only "
             f"{st.loc['feature', 'lgbm_vs_xgb']:.0%} of sampled rows; use the grouped map.",
             c.figure(6, "dominant_driver_map_features"))

print(f"\nDONE in {(time.perf_counter() - t_start) / 60:.1f} min")
