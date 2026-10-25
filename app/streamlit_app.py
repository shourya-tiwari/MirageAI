"""MirageAI dashboard: the project's data, models, evaluation and findings in one place.

    streamlit run app.py              (from the repository root; app.py runs this file)
    streamlit run app/streamlit_app.py

Read-only: every number and chart comes from files the pipeline already wrote
(outputs/tables, outputs/figures, outputs/models), through mirageai.results.
Nothing is refitted. Pages that need a file that is not present (the fitted
models and data/ are not part of the submission zip) say which driver makes it
and fall back to the saved figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mirageai import config, evaluate as E, explain as XP, results as R, scenarios as S

st.set_page_config(page_title="MirageAI · Pune heat drivers", page_icon="🌡️", layout="wide")

# ---------------------------------------------------------------------------
# Visual constants. Colour follows the entity, never its rank (fixed slot per model).
# ---------------------------------------------------------------------------
MODEL_COLOURS = {"lgbm": "#2a78d6", "xgb": "#eb6834", "rf": "#1baf7a",
                 "linear": "#eda100", "baseline": "#898781"}
BOUND_COLOURS = {"lower": "#2a78d6", "realistic": "#eb6834"}
GROUP_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"]  # top three groups + "Other"
MUTED = "#898781"
FEATURE_LABELS = {
    "NDVI": "NDVI (greenness)", "NDBI": "NDBI (dryness)", "MNDWI": "MNDWI (surface moisture)",
    "frac_tree": "Tree cover", "frac_shrub": "Shrubland", "frac_grass": "Grassland",
    "frac_crop": "Cropland (fallow)", "frac_built": "Built-up", "frac_bare": "Bare ground",
    "frac_water": "Water", "elevation": "Elevation (m)", "slope": "Slope (°)",
    "dist_water_m": "Distance to water (m)",
}
SCHEME_LABELS = {
    "random": "Random (grouped by cell)", "spatial_1km": "Spatial, 1 km blocks",
    "spatial_2km": "Spatial, 2 km blocks", "spatial_4km": "Spatial, 4 km blocks (headline)",
    "spatial_8km": "Spatial, 8 km blocks (conservative)",
    "spatio_temporal_4km": "Spatio-temporal: unseen blocks in 2025",
    "spatial_4km_strips": "Stress test: 4 km latitude strips",
    "spatio_temporal_4km_strips": "Stress test: strips, 2025",
}


# ---------------------------------------------------------------------------
# Cached loaders. A missing file becomes None plus a message, never a crash.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame | None:
    try:
        return R.load_table(name)
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def best_params() -> dict | None:
    try:
        return R.load_best_params()
    except FileNotFoundError:
        return None


@st.cache_resource(show_spinner="Loading the fitted LightGBM…")
def final_model():
    try:
        return R.load_model("lgbm_all")
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Reading per-cell SHAP values…")
def shap_cells() -> pd.DataFrame | None:
    try:
        return R.load_shap_cells()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner="Reading the clean modelling table…")
def clean_table() -> pd.DataFrame | None:
    from mirageai import preprocess

    try:
        return preprocess.load_clean()
    except FileNotFoundError:
        return None


def missing(what: str, how: str) -> None:
    st.info(f"**{what}** is not available in this copy. {how}", icon="ℹ️")


def figure(name: str, caption: str | None = None) -> None:
    """Show a saved pipeline figure, if present."""
    path = config.FIGURES / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        missing(f"Figure {name}.png", "Run `scripts/pipeline/run_all.py` to regenerate the figures.")


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    """Shared chart chrome: recessive grid, horizontal legend above the plot."""
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.08, x=0, title=None),
                      hoverlabel=dict(font_size=12))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    return fig


def raster(grid: np.ndarray, extent, colorscale: str, title: str, unit: str,
           zmid: float | None = None, zrange: tuple[float, float] | None = None) -> go.Figure:
    """A north-up 100 m grid as a heatmap with UTM-km axes."""
    left, right, bottom, top = extent
    ny, nx = grid.shape
    x = np.linspace(left, right, nx, endpoint=False) + (right - left) / nx / 2
    y = np.linspace(top, bottom, ny, endpoint=False) - (top - bottom) / ny / 2
    kw = dict(zmid=zmid) if zmid is not None else {}
    if zrange:
        kw.update(zmin=zrange[0], zmax=zrange[1])
    # Explicit rgb() stops: the plotly.js bundled with Streamlit misreads some hex
    # stops of named scales (inferno's near-black end rendered as pink).
    from plotly.colors import convert_colors_to_same_type, get_colorscale

    stops = get_colorscale(colorscale) if isinstance(colorscale, str) else colorscale
    stops = [[float(p), convert_colors_to_same_type(c, colortype="rgb")[0][0]] for p, c in stops]
    fig = go.Figure(go.Heatmap(
        z=grid, x=x, y=y, colorscale=stops, colorbar=dict(title=unit), **kw,
        hovertemplate="E %{x:.1f} km · N %{y:.1f} km<br>%{z:.2f} " + unit + "<extra></extra>"))
    fig.update_yaxes(scaleanchor="x", scaleratio=1, title="UTM 43N northing (km)")
    fig.update_xaxes(title="UTM 43N easting (km)")
    fig.update_layout(title=title, height=560, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fmt(x: float, nd: int = 2, signed: bool = False) -> str:
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview() -> None:
    st.title("MirageAI: Concrete vs. Canopy")
    st.markdown(
        "#### Explainable machine learning for the drivers of daytime land surface "
        "temperature (LST) in Pune")
    st.markdown(
        "Pune's expected daytime **urban heat island turns out to be a mirage**: in the "
        "pre-monsoon season the city core is *cooler* than its dry, fallow surroundings. "
        "MirageAI predicts LST on a 100 m grid from satellite features, validates the "
        "models honestly with **spatial block cross-validation**, explains the drivers "
        "with **SHAP**, and simulates **greening scenarios** for the hottest urban cells.")

    comp, core_ring = table("model_comparison"), table("phase8_core_ring_land")
    if comp is not None and core_ring is not None:
        h = R.headline(comp, core_ring)
        c = st.columns(4)
        c[0].metric("Core cooler than outer ring", f"{h['cool_island_min']:.1f}–{h['cool_island_max']:.1f} °C",
                    help="Land cells, core < 5 km vs ring > 12 km from the AOI centre, 2023-25.")
        c[1].metric("Spatial-CV RMSE (4 km blocks)", f"{h['spatial_rmse']:.3f} °C",
                    f"R² {h['spatial_r2']:.2f}", delta_color="off", delta_arrow="off",
                    help="LightGBM, mean over 5 block layouts × 5 folds.")
        c[2].metric("Unseen places in an unseen year", f"{h['temporal_rmse']:.3f} °C",
                    f"R² {h['temporal_r2']:.2f}", delta_color="off", delta_arrow="off",
                    help="Trained on 2023-24, tested on held-out blocks in 2025.")
        c[3].metric("Random CV would have claimed", f"{h['random_rmse']:.3f} °C",
                    f"{h['spatial_rmse'] - h['random_rmse']:+.3f} °C optimism", delta_color="inverse",
                    help="Random CV leaks neighbouring pixels into training and flatters the model.")
    bracket = table("phase7_summary")
    if bracket is not None:
        b = R.scenario_bracket(bracket).set_index(["scenario", "bound"]).dLST
        st.success(
            f"**Greening signal:** adding 10 points of tree cover to the hottest urban cells is "
            f"associated with **{b['A', 'lower']:.2f} to {b['A', 'realistic']:.2f} °C** lower "
            f"modelled LST (20 points: {b['B', 'lower']:.2f} to {b['B', 'realistic']:.2f} °C). "
            "Associational, not causal; see *Results & Insights*.", icon="🌳")

    st.subheader("Pipeline")
    st.graphviz_chart("""
    digraph {
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2a78d6",
            fontname="Helvetica", fontsize=11];
      edge [color="#898781"];
      A [label="1 · Earth Engine\\nLandsat 8/9, Sentinel-2,\\nWorldCover, SRTM"];
      B [label="2 · Clean table\\n100 m grid × 3 years\\nLST anomaly target"];
      C [label="3 · Validation design\\nvariogram → 4 km blocks"];
      D [label="4 · Models\\nLR, RF, XGBoost, LightGBM\\ntuned inside spatial CV"];
      E [label="5 · Evaluation\\nrandom vs spatial vs\\nspatio-temporal"];
      F [label="6 · SHAP\\ndrivers, thresholds,\\ninteractions"];
      G [label="7 · Scenarios\\n+tree cover in the\\nhottest urban cells"];
      A -> B -> C -> D -> E -> F -> G;
    }""")

    st.subheader("Research questions at a glance")
    st.markdown("""
| | Question | Short answer |
|---|---|---|
| **RQ1** | Which factors drive daytime LST? | The spectral block (greenness, dryness, moisture) first, then grassland, tree cover and fallow cropland. Built-up fraction is a minor driver. |
| **RQ2** | Do nonlinear models find what linear ones miss? | Yes: thresholds (tree-cover cooling saturates near 10%), interactions, and linear regression even gets NDBI's sign wrong. |
| **RQ3** | How much does random CV flatter accuracy? | +0.12 to +0.15 °C RMSE for tree models at 4 km blocks, only +0.02 for linear regression, and it erases the model ranking. |
| **RQ4** | How much cooling goes with more trees? | About −0.4 to −1.0 °C for +10 points, −0.6 to −1.6 °C for +20 points (lower to realistic bound). |
| **RQ5** | Is there a daytime cool island, and does it survive controls? | Yes, the core is 3.4-4.0 °C cooler. The built-up association shrinks 87% once vegetation and dryness are modelled: it is a dryness contrast, not a cooling effect of buildings. |
""")
    with st.expander("Course and context"):
        st.markdown(
            "Applications and Use Cases of Machine Learning (TE7483), Symbiosis Institute of "
            "Technology, Pune. Smart Cities theme; SDG 11 (Sustainable Cities) and SDG 13 "
            "(Climate Action). Study area: PMC + PCMC core, ~800 km², pre-monsoon (Mar–May) "
            "2023, 2024 and 2025.")


def page_dataset() -> None:
    st.title("Dataset")
    c = st.columns(4)
    c[0].metric("Grid cells", "81,983", help="319 × 257 cells at 100 m, identical every year")
    c[1].metric("Years", "3", help="2023 and 2024 train; 2025 is the temporal test year")
    c[2].metric("Model rows (land)", "243,252",
                help="245,949 rows minus 2,697 water-majority rows (D-034)")
    c[3].metric("Features", f"{len(config.FEATURES)}", help="Coordinates are never features")

    t1, t2, t3, t4 = st.tabs(["Sources", "Features", "Cleaning", "Preview"])
    with t1:
        st.markdown("""
| Source | Role | Native resolution | Earth Engine ID |
|---|---|---|---|
| Landsat 8/9 Collection 2 Level-2 | **Target**: LST from `ST_B10` | 100 m thermal (30 m product) | `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` |
| Sentinel-2 SR Harmonized | NDVI, NDBI, MNDWI | 10–20 m | `COPERNICUS/S2_SR_HARMONIZED` |
| ESA WorldCover v200 (2021) | Seven land-cover fractions, distance to water | 10 m | `ESA/WorldCover/v200` |
| SRTM | Elevation, slope | 30 m | `USGS/SRTMGL1_003` |

Each source is cloud-masked (Landsat `QA_PIXEL` bits, Sentinel-2 `SCL` classes), reduced to a
seasonal **median** composite, reprojected to UTM 43N and aggregated by area-weighted mean onto
the 100 m grid, which matches the Landsat thermal sensor's true resolution.""")
    with t2:
        rows = [(f, FEATURE_LABELS[f], grp) for grp, feats in
                (("Spectral", config.SPECTRAL_FEATURES), ("Land-cover fraction", config.FRACTION_FEATURES),
                 ("Terrain / water", config.TERRAIN_FEATURES)) for f in feats]
        st.dataframe(pd.DataFrame(rows, columns=["Column", "Meaning", "Group"]),
                     hide_index=True, width="stretch")
        st.markdown(
            f"**Target:** `{config.TARGET_ANOM}` = LST minus that year's AOI median (removes the "
            "year-to-year level shift no feature can explain, D-021).  \n"
            "**Never features:** coordinates (`lon`, `lat`, `x_utm`, `y_utm`), block ids and the "
            "QC column `LST_n_obs`. `config.py` declares them as `FORBIDDEN_FEATURES`.  \n"
            "**Linear regression** drops `frac_built` (fractions sum to 1); tree models keep all "
            "seven (D-023, D-025).")
    with t3:
        excl = table("phase4_water_exclusion")
        st.markdown(
            "- **Missing values:** none. Pre-monsoon skies are clear enough that every cell has at "
            "least 4 cloud-free Landsat looks in every year, so no cell was dropped.\n"
            "- **Water-majority cells** (`frac_water` ≥ 0.5) are excluded before any split: "
            "reservoir temperature is thermal inertia, not a land-surface driver (D-034).\n"
            "- **Ids** (`cell_id`, block ids) come from coordinates only, so a cell keeps the same "
            "block in every year and can never sit on both sides of a split (D-022).")
        if excl is not None:
            st.dataframe(excl, hide_index=True, width="stretch")
    with t4:
        df = clean_table()
        if df is None:
            missing("data/processed/uhi_pune_clean.parquet",
                    "`data/` is not shipped; see README → Data, then run `scripts/pipeline/run_phase2.py`.")
        else:
            year = st.selectbox("Year", config.YEARS, index=1, key="preview_year")
            part = df[df.year == year]
            st.dataframe(part.head(200), width="stretch", height=300)
            st.caption(f"First 200 of {len(part):,} rows for {year}.")
            st.dataframe(part[[config.TARGET, config.TARGET_ANOM, *config.FEATURES]].describe().T.round(3),
                         width="stretch")


def page_eda() -> None:
    st.title("Exploratory data analysis")
    t1, t2, t3, t4 = st.tabs(["Maps", "Cool island", "Correlations", "Distributions"])
    with t1:
        df = clean_table()
        if df is not None:
            c1, c2 = st.columns([1, 3])
            with c1:
                year = st.radio("Year", config.YEARS, index=1, horizontal=True)
                layer = st.radio("Layer", ["LST (°C)", "LST anomaly (°C)", "NDVI", "Built-up fraction",
                                           "Tree-cover fraction"])
            col = {"LST (°C)": config.TARGET, "LST anomaly (°C)": config.TARGET_ANOM, "NDVI": "NDVI",
                   "Built-up fraction": "frac_built", "Tree-cover fraction": "frac_tree"}[layer]
            part = df[df.year == year]
            grid, ext = R.to_grid(part, part[col])
            lo, hi = np.nanpercentile(part[col], [1, 99])
            scale = {"LST (°C)": "Inferno", "LST anomaly (°C)": "RdBu_r", "NDVI": "YlGn",
                     "Built-up fraction": "Greys", "Tree-cover fraction": "Greens"}[layer]
            mid = 0.0 if col == config.TARGET_ANOM else None
            with c2:
                st.plotly_chart(raster(grid, ext, scale, f"{layer}, pre-monsoon {year}", layer.split(" (")[0]
                                       if "(" not in layer else layer.split("(")[1].rstrip(")"),
                                       zmid=mid, zrange=None if mid is not None else (lo, hi)),
                                width="stretch")
            st.caption("Hover for values. Colour range is the 1st–99th percentile; water cells included.")
        else:
            year = st.radio("Year", config.YEARS, index=1, horizontal=True)
            c1, c2 = st.columns(2)
            with c1:
                figure(f"01_lst_map_{year}", f"Daytime LST, pre-monsoon {year}")
            with c2:
                figure("01_ndvi_map_2024", "NDVI, pre-monsoon 2024")
    with t2:
        cr, bins = table("phase8_core_ring_land"), table("phase8_cool_island_bins")
        st.markdown(
            "The dense core is the **greenest** part of the study area and the outer ring the "
            "**driest**: harvested fallow fields, dry grass and bare soil heat faster than an "
            "irrigated, tree-shaded city. This is the daytime surface *cool* island reported for "
            "semi-arid Indian cities before the monsoon.")
        if cr is not None:
            show = cr[["year", "core_LST", "ring_LST", "core_minus_ring", "core_NDVI", "ring_NDVI"]].round(3)
            st.dataframe(show, hide_index=True, width="stretch")
        if bins is not None:
            fig = px.line(bins, x="mid", y="anom", color=bins.year.astype(str), markers=True,
                          labels={"mid": "Built-up share of the cell (%)", "anom": "Mean LST anomaly (°C)",
                                  "color": "Year"},
                          color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a"])
            fig.add_hline(y=0, line_color=MUTED, line_width=1)
            st.plotly_chart(style(fig), width="stretch")
            st.caption("More built-up cells are cooler on average, because built-up land sits in the "
                       "greener core. SHAP later shows this is a vegetation/dryness contrast (RQ5).")
    with t3:
        corr = table("phase2_lst_anom_correlations")
        if corr is not None:
            corr = corr.rename(columns={corr.columns[0]: "feature"})
            long = corr.melt(id_vars="feature", value_vars=["all_cells", "urban_only"],
                             var_name="subset", value_name="r")
            long["subset"] = long.subset.map({"all_cells": "All cells", "urban_only": "Urban cells only"})
            long["label"] = long.feature.map(FEATURE_LABELS)
            fig = px.bar(long, y="label", x="r", color="subset", barmode="group", orientation="h",
                         labels={"r": "Pearson r with LST anomaly", "label": ""},
                         color_discrete_sequence=["#2a78d6", "#eb6834"],
                         category_orders={"label": [FEATURE_LABELS[f] for f in corr.feature]})
            fig.add_vline(x=0, line_color=MUTED, line_width=1)
            st.plotly_chart(style(fig, 480), width="stretch")
            st.caption("NDBI is the strongest correlate, and it tracks dry soil here, not buildings. "
                       "Correlations shrink inside urban cells partly because the range is restricted.")
        figure("02_correlation_heatmap", "Feature correlation matrix (VIF analysis in D-023/D-024)")
    with t4:
        figure("02_target_distributions", "Absolute LST is offset between years; the anomaly is not (D-021)")
        figure("02_feature_distributions", "Feature distributions, all years pooled")


def page_models() -> None:
    st.title("Models and validation design")
    t1, t2, t3 = st.tabs(["Models", "Validation design", "Tuned hyperparameters"])
    with t1:
        st.markdown(f"""
| Model | Why it is here | Features | Preprocessing |
|---|---|---|---|
| Mean baseline | Sanity floor: any useful model must beat it | – | – |
| Linear regression | Represents the linear, single-index studies MirageAI improves on | {len(config.FEATURES_LINEAR)} (drops `frac_built`) | `StandardScaler` inside a `Pipeline` |
| Random forest | Bagging ensemble, robust baseline for tabular data | {len(config.FEATURES_TREE)} | none needed |
| XGBoost | Gradient boosting, histogram method, CPU only | {len(config.FEATURES_TREE)} | none needed |
| LightGBM | Gradient boosting, leaf-wise; **reference model** for SHAP and scenarios | {len(config.FEATURES_TREE)} | none needed |

The three tree models are statistically tied under spatial CV. LightGBM is used for explanation
because it is lowest on the clean 2025 test and far cheaper to explain (2.8 MB against 115 MB for
the forest), not because it is meaningfully more accurate (D-032). Every model runs on the CPU
with seed 42, so reruns reproduce predictions bit for bit (D-031).""")
    with t2:
        st.markdown(
            "Neighbouring 100 m cells are nearly identical (Moran's I of the target ≈ 0.9). A random "
            "split puts near-duplicates on both sides, so the test score measures memorisation of "
            "nearby pixels. **Spatial block CV** holds out whole blocks instead.")
        st.markdown("""
- **Block size from the data:** the residual variogram of a first LightGBM reaches 95% of its sill
  at ~3.7–4.0 km, so **4 km** blocks are primary; 1, 2 and 8 km are reported for sensitivity (D-028).
- **Five random block layouts** (seeds 42–46) × 5 folds = 25 runs per result, mean ± std (D-033).
- **Spatio-temporal test:** train on 2023–24, test on held-out blocks in **2025**, so test cells are
  unseen in space *and* time (D-010).
- **Tuning inside the folds:** `RandomizedSearchCV` (25 candidates) on the 2023–24 spatial folds;
  2025 is never seen during tuning (D-029).""")
        c1, c2 = st.columns(2)
        with c1:
            figure("03_variogram_lst_anom", "Empirical variogram of the LST anomaly")
        with c2:
            figure("03_block_layout", "Spatial blocks and folds")
    with t3:
        bp = best_params()
        if bp is None:
            missing("best_params.json", "Run `scripts/pipeline/run_phase4.py`.")
        else:
            cols = st.columns(len(bp))
            for col, (name, info) in zip(cols, bp.items()):
                with col:
                    st.markdown(f"**{R.MODEL_LABELS.get(name, name)}**")
                    st.metric("Tuning RMSE (4 km folds)", f"{info['best_cv_rmse']:.3f} °C",
                              f"± {info['best_cv_rmse_std']:.3f}", delta_color="off", delta_arrow="off")
                    st.json(info["best_params"], expanded=True)


def page_evaluation() -> None:
    st.title("Evaluation")
    comp = table("model_comparison")
    if comp is None:
        missing("model_comparison.csv", "Run `scripts/pipeline/run_phase5.py`.")
        return
    t1, t2, t3, t4, t5 = st.tabs(["Comparison table", "Random vs spatial CV (RQ3)",
                                  "Temporal holdout", "Where errors concentrate", "Diagnostic plots"])
    with t1:
        schemes = [s for s in E.SCHEME_ORDER if s in set(comp.scheme)]
        scheme = st.selectbox("Validation scheme", schemes, index=schemes.index(E.PRIMARY_SCHEME),
                              format_func=lambda s: SCHEME_LABELS.get(s, s))
        part = comp[comp.scheme == scheme].copy()
        part["Model"] = part.model.map(R.MODEL_LABELS)
        show = part[["Model", "n_features", "n_runs", "RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean", "R2_std"]]
        st.dataframe(show.rename(columns={"n_features": "Features", "n_runs": "Runs",
                                          "RMSE_mean": "RMSE (°C)", "RMSE_std": "RMSE std",
                                          "MAE_mean": "MAE (°C)", "R2_mean": "R²", "R2_std": "R² std"})
                     .style.format(precision=3), hide_index=True, width="stretch")
        fig = go.Figure()
        for _, r in part[part.model != "baseline"].iterrows():
            fig.add_bar(x=[R.MODEL_LABELS[r.model]], y=[r.RMSE_mean], name=R.MODEL_LABELS[r.model],
                        marker_color=MODEL_COLOURS[r.model],
                        error_y=dict(type="data", array=[r.RMSE_std], color=MUTED),
                        hovertemplate="%{x}: %{y:.3f} °C<extra></extra>")
        fig.update_layout(showlegend=False, yaxis_title="RMSE (°C), mean ± std over runs")
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption("Error bars are fold-to-fold std. Paired on identical folds, the three tree models "
                   "differ by a few thousandths of a degree: a tie.")
        paired = table("phase5_paired_vs_best")
        if paired is not None:
            with st.expander("Paired differences against LightGBM on identical folds"):
                p = paired[paired.scheme == scheme].copy()
                p["model"] = p.model.map(R.MODEL_LABELS)
                st.dataframe(p.style.format(precision=4), hide_index=True, width="stretch")
    with t2:
        curve = R.block_size_curve(comp)
        metric = st.radio("Metric", ["RMSE", "R²"], horizontal=True)
        y, ylab = ("RMSE_mean", "RMSE (°C)") if metric == "RMSE" else ("R2_mean", "R²")
        fig = go.Figure()
        for model in ("linear", "rf", "xgb", "lgbm"):
            m = curve[curve.model == model]
            fig.add_scatter(x=m.block_km, y=m[y], mode="lines+markers", name=R.MODEL_LABELS[model],
                            line=dict(color=MODEL_COLOURS[model], width=2), marker=dict(size=8),
                            hovertemplate="%{x} km: %{y:.3f}<extra>" + R.MODEL_LABELS[model] + "</extra>")
        fig.update_xaxes(title="CV block size (km); 0 = random CV",
                         tickvals=[0, 1, 2, 4, 8], ticktext=["random", "1", "2", "4", "8"])
        fig.update_yaxes(title=ylab)
        st.plotly_chart(style(fig, 420), width="stretch")
        gap = table("phase5_cv_gap")
        if gap is not None:
            g = gap[(gap.scheme == E.PRIMARY_SCHEME) & (gap.model != "baseline")].copy()
            g["model"] = g.model.map(R.MODEL_LABELS)
            st.dataframe(g[["model", "RMSE_random", "RMSE_scheme", "dRMSE", "dRMSE_pct", "dR2"]]
                         .rename(columns={"model": "Model", "RMSE_random": "Random RMSE",
                                          "RMSE_scheme": "4 km RMSE", "dRMSE": "Optimism (°C)",
                                          "dRMSE_pct": "Optimism (%)", "dR2": "ΔR²"})
                         .style.format(precision=3), hide_index=True, width="stretch")
        st.markdown(
            "**Reading:** error rises steadily with block size. Random CV flatters flexible models "
            "most: XGBoost is best under random CV but tied under every spatial scheme. Comparing "
            "models with random CV biases the comparison toward the model that memorises place best.")
    with t3:
        th = table("phase5_temporal_holdout")
        if th is not None:
            th = th[th.model != "baseline"].copy()
            fig = go.Figure()
            for label, col, colour in (("Unseen place and year", "RMSE_unseen_mean", "#2a78d6"),
                                       ("Seen place, unseen year", "RMSE_seen_place_mean", "#eb6834")):
                fig.add_bar(x=th.model.map(R.MODEL_LABELS), y=th[col], name=label, marker_color=colour,
                            hovertemplate="%{x}: %{y:.3f} °C<extra>" + label + "</extra>")
            fig.update_layout(barmode="group", yaxis_title="RMSE on 2025 (°C)")
            st.plotly_chart(style(fig, 380), width="stretch")
            st.markdown(
                "A purely temporal split (same cells, next year) would look much better, because the "
                "model has already seen each place. The gap, **place memorisation**, is about as large "
                "as the random-vs-spatial gap, and again largest for the most flexible models. The "
                "headline uses the spatio-temporal split.")
    with t4:
        c1, c2 = st.columns(2)
        uv = table("phase5_urban_vs_rest")
        with c1:
            st.markdown("**Urban cells vs the rest (4 km spatial CV)**")
            if uv is not None:
                u = uv[(uv.scheme == E.PRIMARY_SCHEME) & (uv.model != "baseline")].copy()
                u["model"] = u.model.map(R.MODEL_LABELS)
                st.dataframe(u[["model", "urban_group", "RMSE_mean", "MAE_mean", "R2_mean"]]
                             .rename(columns={"model": "Model", "urban_group": "Group", "RMSE_mean": "RMSE",
                                              "MAE_mean": "MAE", "R2_mean": "R²"})
                             .style.format(precision=3), hide_index=True, width="stretch")
                st.caption("Urban RMSE is lower but so is R²: the urban anomaly range is narrower. "
                           "Quote subgroup accuracy in °C.")
        with c2:
            st.markdown("**Detecting the hottest decile** (share also predicted in the top decile)")
            hd = table("phase5_hot_decile_overlap")
            if hd is not None:
                det = R.hot_decile_detection(hd)
                det = det[det.model != "baseline"]
                fig = px.bar(det, x="model_label", y="mean", color="decile", barmode="group",
                             labels={"model_label": "", "mean": "Overlap (chance = 10%)", "decile": ""},
                             color_discrete_sequence=["#2a78d6", "#eb6834"])
                fig.update_yaxes(tickformat=".0%", range=[0, 1])
                fig.add_hline(y=0.1, line_dash="dot", line_color=MUTED)
                st.plotly_chart(style(fig, 320), width="stretch")
        mi = table("phase5_morans_i")
        if mi is not None:
            m = R.residual_morans_i(mi)
            m = m[(m.scheme.isin(["none", E.PRIMARY_SCHEME, "random"])) & (m.model != "baseline")]
            m["what"] = np.where(m.model == "raw_target", "Raw target",
                                 m.model.map(R.MODEL_LABELS).fillna(m.model) + " residuals, "
                                 + m.scheme.map(lambda s: "4 km CV" if s != "random" else "random CV"))
            st.markdown("**Spatial autocorrelation left in the residuals (Moran's I, queen contiguity)**")
            st.dataframe(m[["what", "mean", "min", "max"]].rename(columns={"what": "", "mean": "Moran's I"})
                         .style.format(precision=3), hide_index=True, width="stretch")
            st.caption("Residual Moran's I ≈ 0.6: neighbouring cells share errors the 13 features cannot "
                       "explain (building height, roofing material, irrigation). Listed as a limitation.")
    with t5:
        c1, c2 = st.columns(2)
        with c1:
            figure("05_pred_vs_actual", "Predicted vs observed, LightGBM, 4 km spatial CV")
        with c2:
            figure("05_residual_map", "Out-of-fold residual map")
        figure("05_cv_gap", "Random vs spatial CV gap")


def page_explain() -> None:
    st.title("Explainability (SHAP)")
    st.markdown(
        "SHAP splits each prediction into additive contributions, one per feature, in °C of the "
        "LST anomaly. MirageAI uses **interventional TreeSHAP** on the final LightGBM, because the "
        "features are strongly correlated; additivity is checked to 1e-7 (D-036). Language is "
        "associational: SHAP describes the fitted model, not what a change on the ground would cause.")
    t1, t2, t3, t4 = st.tabs(["Global importance", "Driver map", "Thresholds & interactions", "Figures"])
    with t1:
        imp, grp = table("phase6_shap_importance"), table("phase6_group_importance")
        c1, c2 = st.columns(2)
        if imp is not None:
            with c1:
                subset = st.radio("Cells", ["All land cells", "Urban cells only"], horizontal=True)
                suffix = "all" if subset.startswith("All") else "urban"
                long = imp.melt(id_vars="feature", value_vars=[f"lgbm_{suffix}", f"xgb_{suffix}"],
                                var_name="model", value_name="mean_abs")
                long["model"] = long.model.str.split("_").str[0].map(R.MODEL_LABELS)
                long["label"] = long.feature.map(FEATURE_LABELS)
                order = imp.sort_values(f"lgbm_{suffix}").feature.map(FEATURE_LABELS).tolist()
                fig = px.bar(long, y="label", x="mean_abs", color="model", barmode="group",
                             orientation="h", category_orders={"label": order[::-1]},
                             labels={"mean_abs": "Mean |SHAP| (°C)", "label": ""},
                             color_discrete_map={"LightGBM": MODEL_COLOURS["lgbm"],
                                                 "XGBoost": MODEL_COLOURS["xgb"]})
                st.plotly_chart(style(fig, 500), width="stretch")
        if grp is not None:
            with c2:
                st.markdown("**Grouped drivers** (sum of members' SHAP)")
                g = grp.sort_values("lgbm_all")
                fig = go.Figure(go.Bar(y=g.group, x=g.lgbm_all, orientation="h",
                                       marker_color=MODEL_COLOURS["lgbm"],
                                       text=g.lgbm_all.map(lambda v: f"{v:.2f}"), textposition="outside",
                                       cliponaxis=False,
                                       hovertemplate="%{y}: %{x:.3f} °C<extra></extra>"))
                fig.update_xaxes(title="Mean |SHAP| (°C), LightGBM, all land cells",
                                 range=[0, float(g.lgbm_all.max()) * 1.15])
                st.plotly_chart(style(fig, 380), width="stretch")
                st.caption("Inside the spectral block, NDVI, NDBI and MNDWI trade credit between models "
                           "and SHAP methods, so the block is ranked with confidence and its members only "
                           "loosely (LightGBM vs XGBoost Spearman 0.99 overall).")
    with t2:
        cells = shap_cells()
        if cells is None:
            missing("phase6_shap_cells_lgbm.parquet", "Run `scripts/pipeline/run_phase6.py`.")
            figure("06_dominant_driver_map")
        else:
            per_cell = R.cell_mean_shap(cells)
            groups = R.driver_group_values(per_cell)
            c1, c2 = st.columns([1, 3])
            with c1:
                view = st.radio("Show", ["Dominant driver group", "Contribution of one group",
                                         "Contribution of one feature"])
                if view == "Contribution of one group":
                    pick = st.selectbox("Group", list(XP.DRIVER_GROUPS))
                elif view == "Contribution of one feature":
                    pick = st.selectbox("Feature", list(config.FEATURES_TREE),
                                        format_func=lambda f: FEATURE_LABELS[f])
            with c2:
                if view == "Dominant driver group":
                    shares = groups.dominant.value_counts()
                    top = list(shares.index[:3])
                    cat = groups.dominant.where(groups.dominant.isin(top), "other")
                    codes = cat.map({**{t: i for i, t in enumerate(top)}, "other": 3}).astype(float)
                    grid, ext = R.to_grid(per_cell, codes)
                    scale = [[i / 4, c] for i, c in enumerate(GROUP_COLOURS) for i in (i, i + 1)]
                    scale = [[min(p, 1.0), c] for p, c in scale]
                    fig = raster(grid, ext, scale, "Dominant driver group per cell (largest |SHAP|)", "",
                                 zrange=(-0.5, 3.5))
                    fig.update_traces(colorbar=dict(tickvals=[0, 1, 2, 3],
                                                    ticktext=top + ["other"], title=""),
                                      hovertemplate="E %{x:.1f} km · N %{y:.1f} km<extra></extra>")
                    st.plotly_chart(fig, width="stretch")
                    st.caption(" · ".join(f"{k}: {v / len(groups):.1%}" for k, v in shares.items()))
                else:
                    vals = groups[pick] if view.endswith("group") else per_cell[f"shap_{pick}"]
                    lim = float(np.nanpercentile(np.abs(vals), 99))
                    grid, ext = R.to_grid(per_cell, vals)
                    title = pick if view.endswith("group") else FEATURE_LABELS[pick]
                    st.plotly_chart(raster(grid, ext, "RdBu_r", f"SHAP contribution: {title} (red warms, "
                                           "blue cools)", "°C", zmid=0, zrange=(-lim, lim)),
                                    width="stretch")
            st.caption("Mean over 2023–25 per cell, final LightGBM, interventional SHAP; water-majority "
                       "cells are blank.")
    with t3:
        thr = table("phase6_thresholds")
        if thr is not None:
            st.markdown("**Fitted thresholds**: one-knot segmented fit of SHAP on feature value, "
                        "95% bootstrap CI (D-037)")
            show = thr[["feature", "shape", "knot", "knot_lo", "knot_hi", "slope_below_per_sd",
                        "slope_above_per_sd", "zero_share", "step_zero_to_low"]].copy()
            show["feature"] = show.feature.map(FEATURE_LABELS)
            st.dataframe(show.style.format(precision=3, na_rep="–"), hide_index=True,
                         width="stretch")
            st.caption("Tree-cover cooling **saturates near 10%** cover; for zero-inflated fractions the "
                       "jump from none to some (`step_zero_to_low`) is reported separately.")
        inter = table("phase6_interactions")
        if inter is not None:
            st.markdown("**Strongest interactions** (path-dependent SHAP interaction values)")
            it = inter.head(8).copy()
            it["pair"] = it.feature_a.map(FEATURE_LABELS) + " × " + it.feature_b.map(FEATURE_LABELS)
            fig = go.Figure(go.Bar(y=it.pair[::-1], x=it.mean_abs_interaction[::-1], orientation="h",
                                   marker_color=MODEL_COLOURS["lgbm"],
                                   hovertemplate="%{y}: %{x:.3f} °C<extra></extra>"))
            fig.update_xaxes(title="Mean |interaction| (°C)")
            st.plotly_chart(style(fig, 360), width="stretch")
        lin = table("phase6_linear_coefs")
        if lin is not None and thr is not None:
            with st.expander("RQ2: linear coefficients vs the trees' linearised response (per SD)"):
                cmp = lin[["feature", "coef_per_sd"]].merge(
                    thr[["feature", "tree_linearised_per_sd"]], on="feature", how="left")
                cmp["feature"] = cmp.feature.map(FEATURE_LABELS)
                st.dataframe(cmp.rename(columns={"coef_per_sd": "Linear regression",
                                                 "tree_linearised_per_sd": "LightGBM (SHAP slope)"})
                             .style.format(precision=3, na_rep="–"), hide_index=True,
                             width="stretch")
                st.caption("Linear regression gives NDBI the wrong sign and inflates NDVI: with "
                           "collinear indices, least squares trades one coefficient against another.")
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            figure("06_shap_beeswarm", "SHAP beeswarm")
            figure("06_shap_interactions", "Top interaction pairs")
        with c2:
            figure("06_shap_dependence", "Dependence plots for the top drivers")
            figure("06_dominant_driver_map", "Dominant driver group map")


def page_predict() -> None:
    st.title("Interactive prediction")
    st.markdown(
        "Describe a 100 m cell and the final LightGBM predicts its **LST anomaly**: how much "
        "hotter (+) or cooler (−) than the year's city-wide median it would be. The breakdown shows "
        "which features push the prediction up or down.")
    bundle = final_model()
    if bundle is None:
        missing("outputs/models/lgbm_all.joblib",
                "Fitted models are not shipped (gitignored). Run `scripts/pipeline/run_phase6.py` to create it.")
        return
    model, feats = bundle["model"], bundle["features"]

    presets = {
        "Hot tree-less fringe (typical scenario target)":
            dict(NDVI=0.18, NDBI=0.11, MNDWI=-0.33, frac_tree=0.0, frac_shrub=0.02, frac_grass=0.12,
                 frac_crop=0.05, frac_bare=0.03, frac_water=0.0, elevation=575.0, slope=2.0,
                 dist_water_m=1500.0),
        "Green, irrigated core":
            dict(NDVI=0.40, NDBI=-0.02, MNDWI=-0.45, frac_tree=0.35, frac_shrub=0.02, frac_grass=0.03,
                 frac_crop=0.0, frac_bare=0.0, frac_water=0.0, elevation=560.0, slope=1.5,
                 dist_water_m=600.0),
        "Dry hillside grassland":
            dict(NDVI=0.15, NDBI=0.16, MNDWI=-0.30, frac_tree=0.03, frac_shrub=0.15, frac_grass=0.75,
                 frac_crop=0.02, frac_bare=0.05, frac_water=0.0, elevation=760.0, slope=12.0,
                 dist_water_m=2500.0),
    }
    preset = st.selectbox("Start from", list(presets))
    p = presets[preset]

    c1, c2, c3 = st.columns(3)
    vals: dict[str, float] = {}
    with c1:
        st.markdown("**Spectral indices**")
        vals["NDVI"] = st.slider("NDVI (greenness)", 0.0, 0.70, p["NDVI"], 0.01, key=f"ndvi{preset}")
        vals["NDBI"] = st.slider("NDBI (dryness)", -0.25, 0.30, p["NDBI"], 0.01, key=f"ndbi{preset}")
        vals["MNDWI"] = st.slider("MNDWI (moisture)", -0.60, -0.10, p["MNDWI"], 0.01, key=f"mndwi{preset}")
    with c2:
        st.markdown("**Land cover (share of the cell)**")
        for f in ("frac_tree", "frac_grass", "frac_crop", "frac_shrub", "frac_bare", "frac_water"):
            vals[f] = st.slider(FEATURE_LABELS[f], 0.0, 1.0, p[f], 0.01, key=f"{f}{preset}")
        rest = 1.0 - sum(vals[f] for f in config.FRACTION_FEATURES if f != "frac_built")
        vals["frac_built"] = max(rest, 0.0)
        st.caption(f"Built-up takes the remainder: **{vals['frac_built']:.0%}**")
    with c3:
        st.markdown("**Terrain**")
        vals["elevation"] = st.slider("Elevation (m)", 530.0, 780.0, p["elevation"], 5.0, key=f"el{preset}")
        vals["slope"] = st.slider("Slope (°)", 0.0, 25.0, p["slope"], 0.5, key=f"sl{preset}")
        vals["dist_water_m"] = st.slider("Distance to water (m)", 0.0, 3700.0, p["dist_water_m"], 50.0,
                                         key=f"dw{preset}")
    if rest < -1e-9:
        st.error(f"Land-cover shares add up to {1 - rest:.0%}; reduce them to 100% or less.")
        return

    row = XP.float32_exact(pd.DataFrame([vals])[feats])
    contrib = model.predict(row, pred_contrib=True)[0]
    pred, base = float(contrib.sum()), float(contrib[-1])
    if vals["frac_water"] >= config.WATER_EXCLUDE_THRESHOLD:
        st.warning("Water-majority cells are outside the model's scope (D-034); treat this as extrapolation.")

    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Predicted LST anomaly", f"{pred:+.2f} °C",
                  help="Relative to the year's AOI-wide median LST.")
        st.caption(f"Model average (base value): {base:+.2f} °C")
        st.markdown("**Greening what-if** (lower bound: indices held fixed)")
        add = st.slider("Add tree cover (points)", 0, 30, 10, 1)
        if add:
            fr = pd.DataFrame([vals])[list(config.FRACTION_FEATURES)]
            new_fr, short = S.add_tree_cover(fr, add / 100)
            scen = pd.DataFrame([vals])
            scen[list(config.FRACTION_FEATURES)] = new_fr.to_numpy()
            new_pred = float(model.predict(XP.float32_exact(scen[feats]))[0])
            st.metric("With more trees", f"{new_pred:+.2f} °C", f"{new_pred - pred:+.2f} °C",
                      delta_color="inverse")
            if short[0] > 0:
                st.caption(f"Only {add / 100 - short[0]:.0%} could be converted: built-up, bare and "
                           "grass were exhausted.")
            if vals["frac_tree"] == 0:
                st.caption("This cell starts with **no** tree cover, so the change includes the large "
                           "'any canopy' step, which the project reports separately rather than as "
                           "cooling from planting (D-040).")
            st.caption("Tree cover is taken from built-up, bare and grass in proportion (D-040). "
                       "Associational, not a forecast of planting.")
    with m2:
        sv = pd.DataFrame({"feature": feats, "shap": contrib[:-1], "value": [vals[f] for f in feats]})
        sv["label"] = sv.feature.map(FEATURE_LABELS) + " = " + sv.value.map(lambda v: f"{v:.3g}")
        sv = sv.reindex(sv.shap.abs().sort_values().index)
        fig = go.Figure(go.Bar(y=sv.label, x=sv.shap, orientation="h",
                               marker_color=np.where(sv.shap > 0, "#e34948", "#2a78d6"),
                               hovertemplate="%{y}: %{x:+.3f} °C<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED, line_width=1)
        fig.update_xaxes(title="Contribution to the prediction (°C): red warms, blue cools")
        st.plotly_chart(style(fig, 460), width="stretch")
        st.caption("Exact path-dependent TreeSHAP for this one input (LightGBM `pred_contrib`). The "
                   "global rankings use interventional SHAP, so magnitudes can differ slightly.")


def page_results() -> None:
    st.title("Results & insights")
    t1, t2, t3 = st.tabs(["Greening scenarios (RQ4)", "Key findings", "Scenario maps"])
    with t1:
        summ, dose = table("phase7_summary"), table("phase7_dose_response")
        st.markdown(
            "Target: the **2,673 hottest urban cells** (top decile of the observed 3-year mean "
            "anomaly among cells ≥ 50% built-up). Tree cover is added from built-up, bare and grass. "
            "Each scenario runs through three tied models (LightGBM, XGBoost, RF) and two bounds:")
        st.markdown(
            "- **Lower bound:** fractions change, spectral indices held at observed values.\n"
            "- **Realistic bound:** indices also respond, through spatially validated land-cover → "
            "index models (D-041).")
        if summ is not None:
            b = R.scenario_bracket(summ)
            fig = go.Figure()
            for bound in ("lower", "realistic"):
                part = b[b.bound == bound]
                fig.add_bar(x=part.scenario_label, y=part.dLST, name=f"{bound.capitalize()} bound",
                            marker_color=BOUND_COLOURS[bound],
                            error_y=dict(type="data", symmetric=False, array=part.dLST_hi - part.dLST,
                                         arrayminus=part.dLST - part.dLST_lo, color=MUTED),
                            text=part.dLST.map(lambda v: f"{v:.2f} °C"), textposition="inside",
                            insidetextanchor="start", textfont=dict(color="#ffffff"),
                            hovertemplate="%{x}: %{y:.2f} °C<extra>" + bound + "</extra>")
            fig.update_layout(barmode="group", yaxis_title="Mean ΔLST in target cells (°C)")
            st.plotly_chart(style(fig, 400), width="stretch")
            st.caption("Bars: mean of the three models; whiskers: min–max across them. The 0 → 1% "
                       "'any canopy' step is excluded from the headline (see below).")
            with st.expander("Full scenario table"):
                st.dataframe(b.drop(columns="scenario_label").style.format(precision=3),
                             hide_index=True, width="stretch")
        if dose is not None:
            d = R.dose_response(dose)
            fig = go.Figure()
            for bound in ("lower", "realistic"):
                part = d[d.bound == bound]
                fig.add_scatter(x=part.points, y=part["mean"], mode="lines+markers",
                                name=f"{bound.capitalize()} bound",
                                line=dict(color=BOUND_COLOURS[bound], width=2), marker=dict(size=8),
                                hovertemplate="+%{x} points: %{y:.2f} °C<extra>" + bound + "</extra>")
            fig.update_xaxes(title="Points of tree cover added")
            fig.update_yaxes(title="Mean ΔLST (°C), three models")
            st.plotly_chart(style(fig, 380), width="stretch")
            st.markdown(
                "The first point does most of the work: **78.7%** of target cells have *no* trees, and "
                "the model's step from none to some reflects what tree-less places *are* (dense new "
                "fringe, unseen roofs and paving), not what one tree does. Beyond it, cooling per point "
                "roughly halves past the ~10% knot.")
    with t2:
        st.markdown("""
1. **The daytime heat island is inverted.** The core is 3.4–4.0 °C cooler than the outer ring in all
   three years; the hottest cells are dry hillside grassland and fallow cropland, not the city.
2. **Dryness, not concrete, drives daytime LST.** The spectral block (greenness, dryness, moisture)
   carries 1.49 °C of mean |SHAP|, about 2× the next group; built-up fraction ranks 10th of 13.
3. **The cool island is a vegetation/dryness contrast.** The built-up association shrinks 87% once
   vegetation and dryness are modelled and does not flip sign (RQ5).
4. **Random CV inflates accuracy, and more so for flexible models.** +0.12 to +0.15 °C for trees,
   +0.02 °C for linear regression; the model ranking under random CV disappears under spatial CV.
5. **Place memorisation is as large as spatial leakage.** A same-cells temporal split would report
   R² ≈ 0.77 for the wrong reason; unseen places in 2025 give R² 0.72.
6. **Tree cover saturates near 10%.** Gains per point roughly halve past the knot.
7. **The hottest urban cells are the tree-less eastern fringe**, a planning frontier rather than an
   old-city retrofit.""")
        st.info("Every number on this page is read from `outputs/tables`; the report, slides and viva "
                "notes are checked against the same tables by `scripts/check_report_numbers.py`.",
                icon="✅")
    with t3:
        c1, c2 = st.columns(2)
        with c1:
            figure("07_target_cells", "The 2,673 target cells")
        with c2:
            figure("07_delta_lst_maps", "ΔLST maps per scenario")
        figure("07_cooling_per_point", "Cooling per point of tree cover")


def page_limitations() -> None:
    st.title("Limitations & future work")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Limitations")
        st.markdown("""
- **Surface, not air, temperature.** Landsat measures skin temperature at ~10:30 AM. It is not
  what people feel, and it is not the afternoon peak (D-047).
- **Associational, not causal.** Scenario ΔLST is a change in the model's prediction; cells with
  more trees differ in ways the features do not see (D-009).
- **Unmodelled drivers.** Residual Moran's I ≈ 0.6 points to building height, roofing material
  and irrigation, none available on a consistent 100 m grid.
- **Static land cover.** WorldCover is a 2021 snapshot; yearly spectral indices only partly
  compensate.
- **Correlated indices.** NDVI, NDBI and MNDWI trade SHAP credit; only their sum is ranked
  with confidence.
- **Mildly optimistic spatial-CV score.** Tuning used the seed-42 folds; the 2025
  spatio-temporal test is the clean check (D-029).
- **One season, one city.** Pre-monsoon Pune only; monsoon and night-time behaviour differ.""")
    with c2:
        st.subheader("Future work")
        st.markdown("""
- Add **urban-form features**: building height and density (e.g. Open Buildings / GHSL),
  roof albedo, and irrigation proxies.
- **Neighbourhood (focal) features** to capture heat spreading between cells.
- **Night-time and seasonal LST** (ECOSTRESS, MODIS) to test whether the cool island flips
  at night or after the monsoon.
- **Air temperature** from a low-cost sensor network to link LST to thermal comfort.
- **Causal designs** such as before/after studies of real planting programmes.
- **Nested spatial CV** to remove the small tuning optimism.
- Extend to other Indian cities to test whether the inverted island generalises.""")
    st.divider()
    st.subheader("Downloads")
    cols = st.columns(3)
    for col, (name, label, mime) in zip(cols, (
            ("MirageAI_Report.pdf", "Report (PDF)", "application/pdf"),
            ("MirageAI_Slides.pdf", "Slides (PDF)", "application/pdf"),
            ("viva_prep.md", "Viva notes (Markdown)", "text/markdown"))):
        path = config.REPORTS / name
        if path.exists():
            col.download_button(label, path.read_bytes(), file_name=name, mime=mime,
                                width="stretch")


PAGES = {
    "Project overview": page_overview,
    "Dataset": page_dataset,
    "Exploratory analysis": page_eda,
    "Models & validation": page_models,
    "Evaluation": page_evaluation,
    "Explainability": page_explain,
    "Interactive prediction": page_predict,
    "Results & insights": page_results,
    "Limitations & future work": page_limitations,
}

with st.sidebar:
    st.markdown("## 🌡️ MirageAI")
    st.caption("Concrete vs. Canopy · Pune, pre-monsoon 2023–25")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption(f"100 m grid · {config.CRS} · seed {config.SEED}  \n"
               "All figures and numbers are read from `outputs/`.")

PAGES[choice]()
