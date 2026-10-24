"""T8.4 / T9.11: build reports/MirageAI_Slides.pptx (python-pptx) and its PDF export.

    python scripts/build_slides.py

A 19-slide deck for a 10-15 minute viva, in the order Title, Motivation,
Problem statement, Objectives, Dataset, Methodology, ML pipeline, EDA, Models,
Evaluation, Results (accuracy, drivers, greening), Dashboard demo, Key
insights, Limitations, Future scope, Conclusion, Thank you. Each slide's
speaker notes open with the presenter's name.

Every number on a slide or in the speaker notes is formatted here from the
output tables (never typed), and scripts/check_report_numbers.py re-checks the
built deck against the same tables before the PDF is exported. Body text is
word-counted per slide (<= 45 words; the title, objectives, conclusion and
closing slides are declared exceptions). Figures come from outputs/figures and
docs/images. PowerPoint (via COM) writes the PDF.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

from mirageai import config as c
from mirageai import models as mo

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "MirageAI_Slides.pptx"
PDF = ROOT / "reports" / "MirageAI_Slides.pdf"
FIG = ROOT / "outputs" / "figures"
IMG = ROOT / "docs" / "images"

ACCENT = RGBColor(0x2A, 0x78, 0xD6)  # the one accent colour
NAVY = RGBColor(0x14, 0x32, 0x5C)
INK = RGBColor(0x0B, 0x0B, 0x0B)
INK2 = RGBColor(0x52, 0x51, 0x4E)
MUTED = RGBColor(0x89, 0x87, 0x81)
RULE = RGBColor(0xE1, 0xE0, 0xD9)
TINT = RGBColor(0xEE, 0xF4, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Calibri"
W, H = 13.333, 7.5
TOTAL = 19
MAX_BODY_WORDS = 45
WORD_LIMIT_EXCEPTIONS = {"MirageAI (title)", "Objectives", "Conclusion", "Thank you"}
FOOTER = "MirageAI · Concrete vs. Canopy"

# Presenters, in speaking order; each notes page starts with "[First name]".
TEAM = ["Shourya Tiwari", "Shashwat Kumar", "Rishik Sinha", "Shubhankar Bhan"]
P1, P2, P3, P4 = (f"[{n.split()[0]}]" for n in TEAM)

# ---------------------------------------------------------------------------
# Numbers, all from the tables
# ---------------------------------------------------------------------------
T = lambda n: pd.read_csv(c.table(n))  # noqa: E731
p1 = T("phase8_core_ring_land")  # land cells only (D-034)
water = T("phase4_water_exclusion").set_index("year")
rr = T("phase4_residual_range")
mc = T("model_comparison").set_index(["scheme", "model"])
gap = T("phase5_cv_gap").set_index(["model", "scheme"])
th = T("phase5_temporal_holdout").set_index(["scheme", "model"])
gi = T("phase6_group_importance").set_index("group")
ra = T("phase6_rank_agreement").set_index("comparison")
dg = T("phase6_dominant_driver_groups").set_index("dominant")
ds = T("phase6_dominant_stability").set_index("level")
thr = T("phase6_thresholds").set_index("feature")
rq5 = T("phase6_rq5_frac_built").set_index("subset")
mi = T("phase5_morans_i")
cib = T("phase8_cool_island_bins")
sm = T("phase7_summary")
ens = sm[sm.model == "ensemble"].set_index(["scenario", "bound"])
ms = T("phase7_marginal_summary").set_index("bound")
tc = T("phase7_target_cells")

N = {}
N["ring_lo"] = f"{-p1.core_minus_ring.max():.1f}"
N["ring_hi"] = f"{-p1.core_minus_ring.min():.1f}"
N["cell_years"] = f"{int(water.loc['all', 'rows_after']):,}"
N["cells"] = f"{int(water.loc['2023', 'rows_after']):,}"
N["water_rows"] = f"{int(water.loc['all', 'rows_removed']):,}"
N["res_range"] = f"{rr.range_95_m.median():,.0f}"
lo_bin = cib[cib.mid < 10].groupby("year").anom.mean()
hi_bin = cib[(cib.mid > 80) & (cib.mid < 90)].groupby("year").anom.mean()
N["built_gradient"] = f"{(lo_bin - hi_bin).mean():.1f}"
N["gap_linear"] = f"{gap.loc[('linear', 'spatial_4km'), 'dRMSE']:.2f}"
tree_gaps = [gap.loc[(m, "spatial_4km"), "dRMSE"] for m in ("rf", "xgb", "lgbm")]
N["gap_tree_lo"], N["gap_tree_hi"] = f"{min(tree_gaps):.2f}", f"{max(tree_gaps):.2f}"
N["gap_xgb"] = f"{gap.loc[('xgb', 'spatial_4km'), 'dRMSE']:.2f}"
rm = lambda s, m: f"{mc.loc[(s, m), 'RMSE_mean']:.2f}"  # noqa: E731
tree_4km = [mc.loc[("spatial_4km", m), "RMSE_mean"] for m in ("rf", "xgb", "lgbm")]
N["tied_lo"], N["tied_hi"] = f"{min(tree_4km):.2f}", f"{max(tree_4km):.2f}"
K = "spatio_temporal_4km"
N["r2_2025"] = f"{th.loc[(K, 'lgbm'), 'R2_unseen_mean']:.2f}"
N["rmse_2025"] = f"{th.loc[(K, 'lgbm'), 'RMSE_unseen_mean']:.2f}"
mem = [th.loc[(K, m), "memorisation_dRMSE_mean"] for m in ("rf", "xgb", "lgbm")]
N["mem_lo"], N["mem_hi"] = f"{min(mem):.2f}", f"{max(mem):.2f}"
N["mem_linear"] = f"{th.loc[(K, 'linear'), 'memorisation_dRMSE_mean']:.2f}"
N["spectral"] = f"{gi.loc['spectral', 'lgbm_all']:.2f}"
N["dry"] = f"{gi.loc['dry cover', 'lgbm_all']:.2f}"
N["dry_urban"] = f"{gi.loc['dry cover', 'lgbm_urban']:.2f}"
N["spec_ratio"] = f"{gi.loc['spectral', 'lgbm_all'] / gi.loc['dry cover', 'lgbm_all']:.1f}"
N["spearman"] = f"{ra.loc['lgbm vs xgb, all cells', 'spearman']:.2f}"
N["dom_spec"] = f"{100 * dg.loc['spectral', 'share']:.1f}"
N["dom_dry"] = f"{100 * dg.loc['dry cover', 'share']:.1f}"
N["dom_dry_urban"] = f"{100 * dg.loc['dry cover', 'share_of_urban']:.1f}"
N["group_stab"] = f"{100 * ds.loc['group', 'lgbm_vs_xgb']:.0f}"
t = thr.loc["frac_tree"]
N["knot"] = f"{100 * t.pos_knot:.1f}"
N["slope_below"] = f"{t.pos_slope_below / 10:.2f}"
N["slope_above"] = f"{t.pos_slope_above / 10:.2f}"
N["rq5_raw"] = f"{rq5.loc['all', 'raw_slope_per_0.1']:.2f}"
N["rq5_shap"] = f"{rq5.loc['all', 'shap_slope_per_0.1']:.3f}"
N["rq5_shrink"] = f"{100 * (1 - rq5.loc['all', 'shap_slope_per_0.1'] / rq5.loc['all', 'raw_slope_per_0.1']):.0f}"
N["targets"] = f"{len(tc):,}"
lg = mi[(mi.scheme == "spatial_4km") & (mi.model == "lgbm")].morans_I
N["moran"] = f"{lg.mean():.1f}"
for s in ("A", "B", "C"):
    N[f"{s}_lo"] = f"{-ens.loc[(s, 'lower'), 'dLST_excl_first_point']:.2f}"
    N[f"{s}_hi"] = f"{-ens.loc[(s, 'realistic'), 'dLST_excl_first_point']:.2f}"
N["A_lo1"] = f"{-ens.loc[('A', 'lower'), 'dLST_excl_first_point']:.1f}"
N["A_hi1"] = f"{-ens.loc[('A', 'realistic'), 'dLST_excl_first_point']:.1f}"
N["per_pt_below"] = f"{-ms.loc['realistic', 'per_point_2pct_to_knot']:.2f}"
N["per_pt_above"] = f"{-ms.loc['realistic', 'per_point_12_to_30pct']:.2f}"
N["first_pt"] = f"{-ms.loc['realistic', 'first_point_0_to_1pct']:.1f}"
N["n_seeds"] = str(len(c.CV_SEEDS))
N["n_folds"] = str(c.N_SPLITS)
N["n_runs"] = str(len(c.CV_SEEDS) * c.N_SPLITS)
N["n_features"] = str(len(c.FEATURES_TREE))
N["n_iter"] = str(mo.TUNE_N_ITER)
N["subsample"] = f"{mo.TUNE_SUBSAMPLE:,}"

PLANNER = ("Pune's hottest urban surfaces are tree-sparse fringe developments, mostly on the "
           "eastern edge. In these data, raising them to ~10% tree cover is associated with "
           "~0.4–1.0 °C cooler daytime surfaces, with gains per point roughly halving beyond "
           "~10–12% cover. These are cross-sectional associations, not measured effects of planting.")
_findings = (ROOT / "reports" / "phase7_findings.md").read_text(encoding="utf-8")
assert re.sub(r"\s+", " ", PLANNER.replace("–", "-")) in re.sub(r"\s+", " ", _findings), \
    "planner sentence differs from phase7_findings.md"

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
# The default template carries its own author metadata; replace it.
cp = prs.core_properties
cp.title = "Concrete vs. Canopy: Explainable Machine Learning for Urban Heat Island Drivers in Pune"
cp.subject = "MirageAI · TE7483 case study, Symbiosis Institute of Technology, Pune"
cp.author = cp.last_modified_by = "; ".join(TEAM)
cp.keywords = "urban heat island; land surface temperature; spatial cross-validation; SHAP"
cp.comments = ""
BLANK = prs.slide_layouts[6]
body_words: dict[int, int] = {}
manifest = []


def _words(text: str) -> int:
    return len([w for w in re.split(r"\s+", text) if re.search(r"[A-Za-z0-9]", w)])


def slide_no(slide) -> int:
    return list(prs.slides).index(slide) + 1


def textbox(slide, x, y, w, h, runs, size=18, color=INK2, bold=False, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, count=True, bullet=False, spacing=6, font=FONT):
    """runs: str or list of paragraphs; a paragraph is str or list of (text, overrides) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        parts = [(para, {})] if isinstance(para, str) else para
        if bullet:
            parts = [("▪  ", {"color": ACCENT})] + list(parts)
        for text, ov in parts:
            r = p.add_run()
            r.text = text
            f = r.font
            f.name = ov.get("font", font)
            f.size = Pt(ov.get("size", size))
            f.bold = ov.get("bold", bold)
            f.color.rgb = ov.get("color", color)
            if count and text.strip() != "▪":
                n = slide_no(slide)
                body_words[n] = body_words.get(n, 0) + _words(text)
    return tb


def rect(slide, x, y, w, h, fill, shape=MSO_SHAPE.ROUNDED_RECTANGLE):
    box = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.fill.background()
    box.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        box.adjustments[0] = 0.08
    return box


def picture(slide, path, x, y, w, h):
    """Place an image, scaled to fit the box and centred in it; returns its file name."""
    path = Path(path) if Path(path).is_absolute() else FIG / path
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    slide.shapes.add_picture(str(path), Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2),
                             Inches(pw), Inches(ph))
    return path.name


def new_slide(kicker, title, notes, dark=False):
    s = prs.slides.add_slide(BLANK)
    n = slide_no(s)
    if dark:
        s.background.fill.solid()
        s.background.fill.fore_color.rgb = NAVY
    if title and dark:
        textbox(s, 0.6, 1.2, 12.1, 1.3, title, size=60, color=WHITE, bold=True, count=False)
    elif title:
        textbox(s, 0.6, 0.62, 12.1, 0.75, title, size=30, color=INK, bold=True,
                count=False)
        textbox(s, 0.62, 0.3, 8, 0.35, kicker.upper(), size=12, color=ACCENT, bold=True, count=False)
    rule = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.6), Inches(6.98), Inches(W - 0.6),
                                  Inches(6.98))
    rule.line.color.rgb = RULE
    rule.line.width = Pt(0.75)
    textbox(s, 0.55, 7.03, 6, 0.3, FOOTER, size=10, color=MUTED, count=False)
    textbox(s, 11.6, 7.03, 1.2, 0.3, f"{n} / {TOTAL}", size=10, color=MUTED, align=PP_ALIGN.RIGHT,
            count=False)
    s.notes_slide.notes_text_frame.text = notes
    return s


def callout(slide, x, y, w, big, label, big_size=34, color=ACCENT, label_color=INK2):
    big_h = big_size / 72 * 1.25  # line height of the big number, in inches
    textbox(slide, x, y, w, big_h, big, size=big_size, color=color, bold=True)
    textbox(slide, x, y + big_h, w, 0.9, label, size=15, color=label_color)


def card(slide, x, y, w, h, head, body, fill=TINT, head_color=INK, body_color=INK2, size=16):
    rect(slide, x, y, w, h, fill)
    textbox(slide, x + 0.2, y + 0.15, w - 0.4, 0.45, head, size=size + 2, color=head_color, bold=True)
    textbox(slide, x + 0.2, y + 0.62, w - 0.4, h - 0.7, body, size=size, color=body_color)


def add(slide, title, figures):
    manifest.append((slide_no(slide), title, figures))


# ---------------------------------------------------------------------------
# 1 Title  (Shourya)
# ---------------------------------------------------------------------------
s = new_slide("", None, (
    f"{P1} Good morning. I am Shourya, and with Shashwat, Rishik and Shubhankar I will present "
    f"MirageAI: Concrete vs. Canopy, explainable machine learning for urban heat island drivers "
    f"in Pune. In one sentence: we expected Pune to show a daytime urban heat island, and in the "
    f"satellite data it turns out to be a mirage. The city core is cooler than its dry "
    f"surroundings. Over the next twelve minutes we show how we measured that honestly, what the "
    f"models associate with surface heat, and what that means for greening the hottest parts of "
    f"the city. Every claim is an association from comparing places, not a causal effect."))
textbox(s, 0.6, 0.9, 7.0, 1.1, "MirageAI", size=54, color=ACCENT, bold=True, count=False)
textbox(s, 0.6, 2.05, 6.6, 1.7, "Concrete vs. Canopy: Explainable Machine Learning for Urban Heat "
        "Island Drivers in Pune", size=28, color=INK, bold=True, count=False)
rule = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(0.62), Inches(4.0), Inches(2.0), Inches(4.0))
rule.line.color.rgb = ACCENT
rule.line.width = Pt(3)
textbox(s, 0.6, 4.2, 6.6, 1.5, [" · ".join(TEAM[:2]), " · ".join(TEAM[2:])], size=17, color=INK,
        count=False, spacing=4)
textbox(s, 0.6, 5.55, 6.8, 0.9, ["Applications and Use Cases of Machine Learning (TE7483)",
                                 "Symbiosis Institute of Technology, Pune"],
        size=14, color=MUTED, count=False, spacing=2)
add(s, "MirageAI (title)", picture(s, "01_lst_map_2024.png", 7.45, 0.55, 5.5, 6.25))

# ---------------------------------------------------------------------------
# 2 Motivation  (Shourya)
# ---------------------------------------------------------------------------
s = new_slide("Motivation", "Where would greening cool Pune most?", (
    f"{P1} Indian cities are writing heat action plans, and the practical question is where "
    f"greening would cool most. Satellite land surface temperature is the usual evidence: free, "
    f"city-wide and repeatable. Two things make it easy to misuse. First, textbook cities are "
    f"warmer than their surroundings, but daytime Pune is the opposite: land cells within five "
    f"kilometres of the centre are {N['ring_lo']} to {N['ring_hi']} °C cooler than the outer ring "
    f"beyond twelve, in every one of the three years. Assume a hot core and you green the wrong "
    f"places. Second, most published temperature models are scored on randomly held-out pixels, "
    f"which flatters them."))
callout(s, 0.8, 1.9, 5.6, f"{N['ring_lo']}–{N['ring_hi']} °C",
        "cooler: city core vs outer ring, land cells, daytime, 2023–25", big_size=60)
textbox(s, 7.0, 1.95, 5.7, 3.8, ["Heat plans ask where greening cools most.",
                                 "Satellite LST: free, city-wide, repeatable.",
                                 "Assume a hot core, target the wrong places.",
                                 "Random-pixel validation flatters models."],
        size=20, bullet=True, spacing=16)
add(s, "Motivation", "— (headline number)")

# ---------------------------------------------------------------------------
# 3 Problem statement  (Shourya)
# ---------------------------------------------------------------------------
s = new_slide("Problem statement", "Predict and explain surface heat where the model has never looked", (
    f"{P1} Formally, this is supervised regression on spatially autocorrelated data. The inputs "
    f"are satellite land cover, spectral indices and terrain for every hundred-metre cell; the "
    f"output is that cell's daytime temperature anomaly. Three requirements make it hard. Accuracy "
    f"must be honest, so whole areas and a whole year are held out. Explanations must survive a "
    f"change of model or SHAP method. And greening scenarios must be ranges with stated "
    f"assumptions. One hard rule runs through everything: coordinates are never inputs, otherwise "
    f"the model memorises where a cell is instead of learning why it is hot."))
textbox(s, 0.6, 1.65, 12.1, 0.9, "Predict each 100 m cell's daytime LST anomaly in unseen places "
        "and years, and explain it.", size=20, color=INK)
for i, (a, b) in enumerate([("Honest accuracy", "Unseen areas and an unseen year."),
                            ("Trustworthy explanation", "Stable across models and methods."),
                            ("Bounded scenarios", "Ranges, not forecasts.")]):
    card(s, 0.6 + i * 4.1, 2.75, 3.85, 1.6, a, b, size=18)
textbox(s, 0.6, 4.85, 12.1, 0.6, [[("Hard rule:  ", {"color": ACCENT, "bold": True}),
                                  ("coordinates are never model inputs.", {"color": INK})]], size=19)
add(s, "Problem statement", "— (native cards)")

# ---------------------------------------------------------------------------
# 4 Objectives  (Shourya)
# ---------------------------------------------------------------------------
s = new_slide("Objectives", "Objectives: five research questions", (
    f"{P1} Five research questions structure the work. The first asks which land-cover and terrain "
    f"factors are associated with daytime surface temperature. The second asks whether nonlinear "
    f"models find thresholds and interactions that a linear model misses. The third measures how "
    f"much random cross-validation overstates accuracy. The fourth asks how much cooling is "
    f"associated with more tree cover in the hottest urban cells. The fifth asks whether Pune has "
    f"a daytime cool island, and whether the negative built-up association survives once "
    f"vegetation and dryness are modelled. Each gets a number by the end. Shashwat will now take "
    f"you through the data and method."))
rqs = [("Drivers", "Which land-cover and terrain factors go with daytime LST?"),
       ("Nonlinearity", "Do tree models find thresholds and interactions that linear models miss?"),
       ("Validation", "How much does random CV overstate accuracy?"),
       ("Greening", "How much cooling goes with more tree cover in the hottest cells?"),
       ("Cool island", "Is the built core cooler by day, and is that the buildings?")]
for i, (a, b) in enumerate(rqs):
    y0 = 1.65 + i * 1.02
    rect(s, 0.6, y0, 1.15, 0.82, ACCENT)
    textbox(s, 0.6, y0 + 0.17, 1.15, 0.5, f"RQ{i + 1}", size=20, color=WHITE, bold=True,
            align=PP_ALIGN.CENTER, count=False)
    rect(s, 1.85, y0, 10.85, 0.82, TINT)
    textbox(s, 2.05, y0 + 0.18, 10.5, 0.5, [[(f"{a}:  ", {"bold": True, "color": INK}), (b, {})]], size=18)
add(s, "Objectives", "— (native list)")

# ---------------------------------------------------------------------------
# 5 Dataset  (Shourya)
# ---------------------------------------------------------------------------
s = new_slide("Dataset", "Four open datasets on one 100 m grid", (
    f"{P1} The study area is a box over Pune and Pimpri-Chinchwad on a hundred-metre grid, the "
    f"native resolution of the Landsat thermal band. We use three pre-monsoon seasons, March to "
    f"May of 2023, 2024 and 2025, when skies are clear and surface heat peaks. All four sources come "
    f"through Google Earth Engine. Landsat 8 and 9 give the target, land surface temperature. "
    f"Sentinel-2 gives NDVI, a dryness index and a moisture index. ESA WorldCover gives seven "
    f"land-cover fractions per cell, and SRTM gives elevation and slope. We dropped "
    f"{N['water_rows']} water-majority rows, leaving {N['cells']} land cells a year, "
    f"{N['cell_years']} cell-years and {N['n_features']} features."))
add(s, "Dataset", picture(s, "01_ndvi_map_2024.png", 0.5, 1.5, 6.3, 5.3))
tiles = [("Landsat 8/9", "surface temperature (target)"), ("Sentinel-2", "NDVI, dryness, moisture"),
         ("WorldCover 2021", "7 land-cover fractions"), ("SRTM", "elevation, slope")]
for i, (a, b) in enumerate(tiles):
    y0 = 1.6 + i * 1.02
    rect(s, 7.2, y0, 5.5, 0.85, TINT)
    textbox(s, 7.4, y0 + 0.14, 5.2, 0.6, [[(a, {"bold": True, "color": INK}), (f"   {b}", {})]], size=17)
textbox(s, 7.2, 5.85, 5.6, 0.9, f"100 m grid · 3 seasons · {N['cell_years']} land cell-years",
        size=17, color=ACCENT, bold=True)

# ---------------------------------------------------------------------------
# 6 Methodology  (Shashwat)
# ---------------------------------------------------------------------------
s = new_slide("Methodology", "Validation designed before any model is tuned", (
    f"{P2} Thank you, Shourya. The target is each cell's temperature anomaly from that year's city "
    f"median, because the absolute level shifts between years with weather and scene timing. The "
    f"key design choice is validation. Neighbouring cells are nearly identical, so we hold out "
    f"whole square blocks. The block size comes from the autocorrelation the model cannot "
    f"explain: the residual variogram flattens at about {N['res_range']} metres, so we use "
    f"four-kilometre blocks. The map on the right shows one layout: blocks are dealt to "
    f"{N['n_folds']} folds at random, repeated over {N['n_seeds']} seeds, so every score is a mean "
    f"over {N['n_runs']} runs. The 2025 season is held back as an untouched test."))
steps = [("Target", "anomaly from each year's median"),
         ("Block size", f"residual range {N['res_range']} m → 4 km"),
         ("Layouts", f"{N['n_seeds']} random seeds × {N['n_folds']} folds"),
         ("Final test", "2025, unseen blocks, untouched")]
for i, (a, b) in enumerate(steps):
    y0 = 1.6 + i * 1.3
    rect(s, 0.6, y0, 4.6, 1.12, TINT)
    textbox(s, 0.8, y0 + 0.12, 4.3, 0.45, a, size=18, color=INK, bold=True)
    textbox(s, 0.8, y0 + 0.55, 4.3, 0.5, b, size=16)
add(s, "Methodology", picture(s, "03_block_layout.png", 5.45, 1.6, 7.5, 5.05))

# ---------------------------------------------------------------------------
# 7 ML pipeline  (Shashwat)
# ---------------------------------------------------------------------------
s = new_slide("ML pipeline", "Eight stages, one command", (
    f"{P2} This is the whole pipeline. We extract and composite the data in Earth Engine, then "
    f"build the hundred-metre table with the anomaly target and remove water cells. Stage three "
    f"is the validation design you just saw. We then train a baseline and four learners, tuned "
    f"inside the spatial folds, and evaluate them under random, spatial and spatio-temporal "
    f"cross-validation. Stage six explains the models with SHAP; stage seven runs the greening "
    f"scenarios; stage eight is this deck, the report and the dashboard. The blue stages are the "
    f"ones that decide whether the results can be trusted. One command rebuilds everything, and "
    f"a script re-checks every number we quote against the output tables."))
add(s, "ML pipeline", picture(s, "09_pipeline.png", 0.6, 1.5, 12.1, 4.75))
textbox(s, 0.6, 6.3, 12.1, 0.55, "One command rebuilds every table and figure; a script re-checks "
        "every quoted number.", size=17, color=ACCENT, bold=True, align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------------------
# 8 EDA  (Shashwat)
# ---------------------------------------------------------------------------
s = new_slide("Exploratory data analysis", "The mirage: the built core is cooler by day", (
    f"{P2} The first thing the data showed us. On the left is the 2024 temperature map: rivers and "
    f"reservoirs are coolest, the dry hills and fringe are hottest, and the built-up core sits in "
    f"between. On the right, cell by cell, the least-built tenth of cells runs "
    f"{N['built_gradient']} °C warmer than cells that are eighty to ninety percent built, in all "
    f"three years. The second panel explains why NDBI misleads here. It was designed as a "
    f"built-up index, but it peaks in the least-built cells, because it tracks dry grass, fallow "
    f"fields and bare soil. So we call it a dryness index throughout, and MNDWI a moisture index."))
add(s, "Exploratory data analysis", picture(s, "01_lst_map_2024.png", 0.4, 1.45, 5.3, 4.65)
    + " + " + picture(s, "08_cool_island.png", 5.8, 1.5, 7.2, 3.6))
callout(s, 6.0, 5.2, 3.3, f"{N['built_gradient']} °C", "least-built vs most-built cells", big_size=30)
textbox(s, 9.4, 5.25, 3.6, 1.5, "NDBI peaks in the least-built cells: here it measures dryness.",
        size=15)

# ---------------------------------------------------------------------------
# 9 Models  (Shashwat)
# ---------------------------------------------------------------------------
s = new_slide("Models", "A baseline and four learners on identical folds", (
    f"{P2} We compare five models on exactly the same folds. The mean baseline is the floor. "
    f"Linear regression, standardised inside each fold, stands for the linear status quo in heat "
    f"island studies. Random Forest averages many deep trees; XGBoost and LightGBM are gradient "
    f"boosting, which adds small trees one at a time to fix the previous errors. Hyperparameters "
    f"come from a randomised search of {N['n_iter']} candidates per model, scored inside the "
    f"four-kilometre spatial folds, on 2023 and 2024 only and a {N['subsample']}-row sample. So the "
    f"2025 test never influenced a single setting. All three tree models settle on large, slowly "
    f"learning ensembles."))
mods = [("Baseline", "training mean: the floor"),
        ("Linear", "standardised in-fold; the status quo"),
        ("Random Forest", "bagged deep trees"),
        ("XGBoost", "boosted trees, depth-wise"),
        ("LightGBM", "boosted trees, leaf-wise")]
for i, (a, b) in enumerate(mods):
    x0 = 0.6 + i * 2.46
    card(s, x0, 1.75, 2.3, 2.35, a, b, fill=ACCENT if i >= 2 else TINT,
         head_color=WHITE if i >= 2 else INK, body_color=WHITE if i >= 2 else INK2, size=15)
rect(s, 0.6, 4.5, 12.1, 1.75, TINT)
textbox(s, 0.85, 4.65, 11.6, 0.5, "Tuning, inside spatial CV", size=18, color=INK, bold=True)
textbox(s, 0.85, 5.2, 11.6, 0.9, f"Randomised search · {N['n_iter']} candidates per model · 4 km "
        f"spatial folds · 2023 and 2024 only · {N['subsample']}-row sample", size=17)
add(s, "Models", "— (native cards)")

# ---------------------------------------------------------------------------
# 10 Evaluation  (Shashwat)
# ---------------------------------------------------------------------------
s = new_slide("Evaluation", "Random cross-validation flatters flexible models", (
    f"{P2} We score every model by RMSE in degrees, with R squared and residual Moran's I as "
    f"diagnostics, under random, block and spatio-temporal tests. Here error rises steadily as the "
    f"held-out blocks grow from one to eight kilometres. The key finding is that the inflation "
    f"depends on the model. Going from random to four-kilometre blocks, linear regression loses "
    f"only {N['gap_linear']} °C, while the tree models lose {N['gap_tree_lo']} to "
    f"{N['gap_tree_hi']} °C. The most flexible model, XGBoost, loses the most, {N['gap_xgb']} °C. "
    f"So random validation does not just inflate scores, it changes which model looks best. "
    f"Rishik will now show the results."))
add(s, "Evaluation", picture(s, "05_cv_gap.png", 0.4, 1.4, 12.5, 4.7))
textbox(s, 0.6, 6.2, 12.2, 0.6, [[("Random → 4 km blocks:  ", {"color": INK, "bold": True}),
                                  (f"linear +{N['gap_linear']} °C,  trees +{N['gap_tree_lo']}–"
                                   f"{N['gap_tree_hi']} °C. Flexible models are flattered most.", {})]],
        size=18, align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------------------
# 11 Results: accuracy  (Rishik)
# ---------------------------------------------------------------------------
s = new_slide("Results · accuracy", "The three tree models tie once places are unseen", (
    f"{P3} Thank you, Shashwat. The table gives RMSE in degrees for each model under each test. "
    f"Under random validation XGBoost looks best. Once whole blocks are held out, the three tree "
    f"models are tied, between {N['tied_lo']} and {N['tied_hi']} °C, well inside the fold spread. "
    f"The cleanest test is 2025: places and a year the models never saw. LightGBM scores "
    f"{N['rmse_2025']} °C, R squared {N['r2_2025']}. The last column scores the same rows with a "
    f"model that had seen those places in 2023 and 2024. The difference is the memorisation gap: "
    f"{N['mem_lo']} to {N['mem_hi']} °C for the trees but only {N['mem_linear']} for linear "
    f"regression. We use LightGBM as the reference: lowest on 2025 and cheapest to explain."))
rows = [("linear", "Linear"), ("rf", "Random Forest"), ("xgb", "XGBoost"), ("lgbm", "LightGBM")]
cols = ["RMSE, °C", "Random CV", "4 km blocks", "2025 unseen", "2025 seen place"]
tbl = s.shapes.add_table(len(rows) + 1, len(cols), Inches(0.6), Inches(1.7), Inches(8.2), Inches(3.4)).table
for j, wdt in enumerate([2.0, 1.45, 1.55, 1.55, 1.65]):
    tbl.columns[j].width = Inches(wdt)
for i in range(len(rows) + 1):
    for j in range(len(cols)):
        cell = tbl.cell(i, j)
        if i == 0:
            txt = cols[j]
        elif j == 0:
            txt = rows[i - 1][1]
        else:
            m = rows[i - 1][0]
            txt = {1: rm("random", m), 2: rm("spatial_4km", m),
                   3: f"{th.loc[(K, m), 'RMSE_unseen_mean']:.2f}",
                   4: f"{th.loc[(K, m), 'RMSE_seen_place_mean']:.2f}"}[j]
        cell.text = txt
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        run = para.runs[0]
        run.font.name, run.font.size = FONT, Pt(17 if i else 15)
        run.font.bold = i == 0 or (j == 0)
        run.font.color.rgb = WHITE if i == 0 else INK
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT if i == 0 else (TINT if i % 2 else WHITE)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
add(s, "Results: accuracy", "— (native table)")
callout(s, 9.3, 1.6, 3.6, f"{N['tied_lo']}–{N['tied_hi']} °C", "trees tied, 4 km blocks", big_size=30)
callout(s, 9.3, 3.2, 3.6, f"R² {N['r2_2025']}", "2025, unseen places", big_size=30)
callout(s, 9.3, 4.8, 3.6, f"{N['mem_lo']}–{N['mem_hi']} °C", "memorisation gap, trees (linear "
        f"{N['mem_linear']})", big_size=30)

# ---------------------------------------------------------------------------
# 12 Results: drivers  (Rishik)
# ---------------------------------------------------------------------------
s = new_slide("Results · drivers", "Vegetation, dryness and moisture lead", (
    f"{P3} SHAP splits each prediction into contributions from each feature, in degrees. NDVI, "
    f"dryness and moisture are strongly correlated, so the models trade credit between them, and "
    f"we rank groups instead. The spectral block dominates: {N['spectral']} °C on average per "
    f"cell, about {N['spec_ratio']} times dry cover at {N['dry']} °C. In urban cells, dry cover "
    f"rises to {N['dry_urban']} °C. Built-up fraction, the concrete in our title, is the smallest "
    f"group. XGBoost gives the same ranking, rank correlation {N['spearman']}. The map shows the "
    f"leading group per cell: spectral in {N['dom_spec']}% of cells and dry cover in "
    f"{N['dom_dry']}%, rising to {N['dom_dry_urban']}% of urban cells. The leading group is the "
    f"same under XGBoost for {N['group_stab']}% of cells. These are associations, not causes."))
add(s, "Results: drivers", picture(s, "09_group_ranking.png", 0.4, 1.45, 6.4, 3.9)
    + " + " + picture(s, "06_dominant_driver_map.png", 6.9, 1.45, 6.1, 3.9))
callout(s, 0.8, 5.35, 3.8, f"{N['spectral']} °C", "spectral block, mean |SHAP|", big_size=30)
callout(s, 4.9, 5.35, 3.8, f"{N['spec_ratio']}×", "the dry-cover group", big_size=30)
callout(s, 9.0, 5.35, 3.8, f"{N['group_stab']}%", "same leading group, XGBoost", big_size=30)

# ---------------------------------------------------------------------------
# 13 Results: greening  (Rishik)
# ---------------------------------------------------------------------------
s = new_slide("Results · greening scenarios", "Greening the hottest urban cells", (
    f"{P3} We targeted the {N['targets']} urban cells in the hottest tenth of observed three-year "
    f"temperature: mostly tree-sparse fringe development on the eastern edge, shown on the left. "
    f"We raised their tree cover, taking the new canopy from built, bare and grass in proportion. "
    f"The lower bound holds the spectral indices fixed; the realistic bound lets them respond. "
    f"Adding ten points of tree cover is associated with {N['A_lo']} to {N['A_hi']} °C cooler "
    f"surfaces, and twenty points with {N['B_lo']} to {N['B_hi']}. Gains per point roughly halve "
    f"past the knot, from {N['per_pt_below']} to {N['per_pt_above']} °C. We exclude the first "
    f"pixel of canopy: on its own it is worth {N['first_pt']} °C, which reflects what tree-less "
    f"places are, not what one tree does."))
add(s, "Results: greening", picture(s, "07_target_cells.png", 0.25, 1.4, 6.6, 4.6)
    + " + " + picture(s, "09_scenario_bracket.png", 6.85, 1.45, 6.2, 4.15))
textbox(s, 0.6, 6.15, 12.2, 0.7, [[(f"~10% canopy: {N['A_lo1']}–{N['A_hi1']} °C cooler. ",
                                    {"color": ACCENT, "bold": True}),
                                   ("Gains per point halve past ~10–12%. First-pixel step excluded.", {})]],
        size=18, align=PP_ALIGN.CENTER)

# ---------------------------------------------------------------------------
# 14 Dashboard demo  (Rishik)
# ---------------------------------------------------------------------------
s = new_slide("Dashboard demo", "Every result, explorable in a browser", (
    f"{P3} Everything we have shown is also in an interactive Streamlit dashboard. It has nine "
    f"pages, from the dataset and exploratory maps to evaluation, explainability and the scenario "
    f"results. It is read-only: it loads the tables and figures the pipeline wrote, so it cannot "
    f"disagree with the report. The one live page is interactive prediction, shown here. We pick a "
    f"hot, tree-less fringe cell; the final LightGBM predicts its anomaly and breaks it into "
    f"per-feature contributions, red for warming and blue for cooling. Then we add tree cover and "
    f"see the lower-bound change. If the live demo is slow, this screenshot is the same page. "
    f"Shubhankar will close with what we learned."))
add(s, "Dashboard demo", picture(s, IMG / "dashboard_interactive_prediction.png", 0.5, 1.45, 5.6, 5.35))
textbox(s, 6.5, 1.6, 6.3, 0.5, "Run it", size=19, color=ACCENT, bold=True)
rect(s, 6.5, 2.1, 6.2, 0.62, TINT)
textbox(s, 6.65, 2.2, 6.0, 0.45, "streamlit run app.py", size=15, color=INK,
        font="Consolas")
textbox(s, 6.5, 3.05, 6.3, 0.5, "Demo path", size=19, color=ACCENT, bold=True)
textbox(s, 6.5, 3.55, 6.3, 2.6, ["Pick a hot, tree-less fringe cell.", "Read its per-feature breakdown.",
                                 "Add tree cover: lower-bound what-if.",
                                 "Read-only: numbers come from the pipeline."],
        size=18, bullet=True, spacing=10)

# ---------------------------------------------------------------------------
# 15 Key insights  (Shubhankar)
# ---------------------------------------------------------------------------
s = new_slide("Key insights", "Concrete vs canopy: what the evidence says", (
    f"{P4} Thank you, Rishik. Three insights. First, concrete. On raw data more built-up cover goes "
    f"with cooler surfaces, {N['rq5_raw']} °C per ten points. With vegetation and dryness in the "
    f"model, that association shrinks to {N['rq5_shap']} °C, {N['rq5_shrink']}% smaller, and it does "
    f"not flip. So the cool core is mostly a vegetation and dryness contrast, not an effect of "
    f"buildings. Second, canopy. The tree-cover response bends: about {N['slope_below']} °C per "
    f"ten points up to a knot at {N['knot']}%, then only {N['slope_above']} beyond, and both "
    f"models agree on that shape. Third, method. Random validation flatters flexible models, so "
    f"spatial validation is the honest score."))
add(s, "Key insights", picture(s, "08_tree_cover_dependence.png", 0.3, 1.45, 7.3, 5.2))
ins = [(f"−{N['rq5_shrink']}%", "built-up association once vegetation and dryness are modelled"),
       (f"{N['knot']}%", "tree-cover knot: gains per point flatten beyond"),
       ("Spatial CV", "the honest score; random CV flatters flexible models")]
for i, (big, lab) in enumerate(ins):
    y0 = 1.55 + i * 1.75
    rect(s, 7.85, y0, 0.08, 1.45, ACCENT, MSO_SHAPE.RECTANGLE)
    callout(s, 8.1, y0, 4.8, big, lab, big_size=30)

# ---------------------------------------------------------------------------
# 16 Limitations  (Shubhankar)
# ---------------------------------------------------------------------------
s = new_slide("Limitations", "What these results cannot tell us", (
    f"{P4} Four limits. One: land surface temperature is the skin temperature the satellite sees "
    f"in daytime, not air temperature or comfort. Two: everything is an association from comparing "
    f"places; we have no before-and-after planting evidence. Three: the residuals are still "
    f"clustered, with Moran's I around {N['moran']}, which points to drivers we cannot see, such "
    f"as building height, roofing material and irrigation. The map shows those clusters, red where "
    f"a cell is hotter than predicted. Four: land cover is a single 2021 map for all three years. "
    f"None of these changes the validation results; they bound how far the scenarios can be read."))
textbox(s, 0.6, 1.7, 5.7, 4.6, ["Daytime surface LST, not air temperature",
                                "Associations, not causal effects",
                                f"Residual Moran's I ≈ {N['moran']}: missing drivers",
                                "Static 2021 land cover",
                                "One city, one season"], size=19, bullet=True, spacing=14)
add(s, "Limitations", picture(s, "05_residual_map.png", 6.4, 1.45, 6.6, 5.4))

# ---------------------------------------------------------------------------
# 17 Future scope  (Shubhankar)
# ---------------------------------------------------------------------------
s = new_slide("Future scope", "Where the work goes next", (
    f"{P4} With more time we would do six things. Add the urban-form layers the residuals point "
    f"to: building height, roof material and irrigated green space. Pair surface temperature with "
    f"ground or mobile air-temperature data, to see whether the associations carry over to the "
    f"air people breathe. Use land-cover change over several years for before-and-after "
    f"comparisons. Extend to night-time and post-monsoon composites, where the heat island may "
    f"not be inverted. Run nested spatial cross-validation to remove the small tuning optimism. "
    f"And ship the index models so the dashboard can show the realistic bound live."))
fut = [("Urban form", "building height, roofs, irrigation"), ("Air temperature", "ground and mobile sensors"),
       ("Before and after", "multi-year land-cover change"), ("Night and seasons", "where the island may flip"),
       ("Nested spatial CV", "remove tuning optimism"), ("Live scenarios", "realistic bound in the dashboard")]
for i, (a, b) in enumerate(fut):
    row, col = divmod(i, 3)
    card(s, 0.6 + col * 4.1, 1.75 + row * 2.3, 3.85, 1.95, a, b, size=18)
add(s, "Future scope", "— (native cards)")

# ---------------------------------------------------------------------------
# 18 Conclusion  (Shubhankar)
# ---------------------------------------------------------------------------
s = new_slide("Conclusion", "Conclusion", (
    f"{P4} To conclude, three numbers and one sentence. Spatially validated models map the anomaly "
    f"to {N['rmse_2025']} °C RMSE, R squared {N['r2_2025']}, in places and a year they never saw. "
    f"And the sentence we would give a planner, read as written: Pune's hottest urban surfaces are "
    f"tree-sparse fringe developments, mostly on the eastern edge. In these data, raising them to "
    f"about ten percent tree cover is associated with roughly {N['A_lo1']} to {N['A_hi1']} °C "
    f"cooler daytime surfaces, with gains per point roughly halving beyond ten to twelve percent "
    f"cover. These are cross-sectional associations, not measured effects of planting. In daytime "
    f"Pune, the surface heat is in the dry fringe, not the dense core."))
for i, (big, lab) in enumerate([(f"{N['rmse_2025']} °C", "RMSE, unseen places and year"),
                                (f"R² {N['r2_2025']}", "on the 2025 test"),
                                (f"{N['A_lo1']}–{N['A_hi1']} °C", "cooler at ~10% canopy")]):
    rect(s, 0.6 + i * 4.1, 1.65, 3.85, 1.55, TINT)
    callout(s, 0.8 + i * 4.1, 1.75, 3.5, big, lab, big_size=30)
rect(s, 0.6, 3.55, 0.08, 2.9, ACCENT, MSO_SHAPE.RECTANGLE)
textbox(s, 0.95, 3.5, 11.7, 3.1, PLANNER, size=22, color=INK, spacing=0)
add(s, "Conclusion", "— (planner sentence)")

# ---------------------------------------------------------------------------
# 19 Thank you  (Shubhankar)
# ---------------------------------------------------------------------------
s = new_slide("", "Thank you", (
    f"{P4} Thank you for listening. We are happy to take questions, and every number you have seen "
    f"can be traced to an output table and a logged decision."), dark=True)
textbox(s, 0.6, 2.55, 12.1, 0.9, "Questions?", size=34, color=ACCENT, bold=True)
textbox(s, 0.6, 3.75, 12.1, 1.3, [" · ".join(TEAM[:2]), " · ".join(TEAM[2:])], size=22, color=WHITE,
        spacing=6)
textbox(s, 0.6, 5.0, 12.1, 0.9, ["Symbiosis Institute of Technology, Pune",
                                 "Applications and Use Cases of Machine Learning (TE7483)"],
        size=15, color=RGBColor(0xC9, 0xD6, 0xEA), spacing=2)
textbox(s, 0.6, 6.0, 12.1, 0.5, "Report · slides · dashboard · code: MirageAI repository", size=15,
        color=RGBColor(0xC9, 0xD6, 0xEA))
add(s, "Thank you", "—")

# ---------------------------------------------------------------------------
# Checks, save, export
# ---------------------------------------------------------------------------
titles = {n: title for n, title, _ in manifest}
over = {n: w for n, w in body_words.items()
        if w > MAX_BODY_WORDS and titles[n] not in WORD_LIMIT_EXCEPTIONS}
assert not over, f"slides over {MAX_BODY_WORDS} body words: {over}"
assert len(prs.slides) == TOTAL == len(manifest)
prs.save(OUT)
print(f"wrote {OUT.relative_to(ROOT)}")

sys.stdout.reconfigure(encoding="utf-8")  # the manifest carries dashes and "≈"
WPM = 140
rows_out = []
for i, sl in enumerate(prs.slides, 1):
    notes = sl.notes_slide.notes_text_frame.text
    words = _words(notes)
    rows_out.append({"slide": i, "title": manifest[i - 1][1], "figures": manifest[i - 1][2],
                     "body_words": body_words.get(i, 0), "note_words": words,
                     "note_seconds": round(words / WPM * 60),
                     "presenter": re.match(r"\[(\w+)\]", notes).group(1)})
summary = pd.DataFrame(rows_out)
summary.to_csv(ROOT / "reports" / "slides_manifest.csv", index=False)
print(summary.to_string(index=False))
print(f"speaker notes: {summary.note_words.sum()} words ≈ {summary.note_seconds.sum() / 60:.1f} min at {WPM} wpm")
print(summary.groupby("presenter", sort=False).note_seconds.sum().div(60).round(1).to_string())

check = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_report_numbers.py")])
if check.returncode:
    sys.exit("number check failed; fix the deck before exporting")

ps = f"""
$ErrorActionPreference = 'Stop'
$pp = New-Object -ComObject PowerPoint.Application
try {{
  $p = $pp.Presentations.Open('{OUT}', $true, $false, $false)
  $p.SaveAs('{PDF}', 32)
  $p.Close()
}} finally {{ $pp.Quit() }}
"""
out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                     capture_output=True, text=True)
if out.returncode:
    sys.exit(f"PowerPoint PDF export failed:\n{out.stderr}")
print(f"wrote {PDF.relative_to(ROOT)}")
