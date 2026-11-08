"""Defect Radar - Streamlit app for PROMISE software defect prediction.

Run with::

    streamlit run app.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluate import (compare_models, evaluate_model, get_classification_report,
                          get_confusion_matrix, get_roc_data, positive_scores, rank_models)
from src.predict import Artifacts, ensure_artifacts, predict_dataframe, prepare_inputs
from src.preprocessing import validate_schema
from src.utils import (CLASS_LABELS, FEATURE_COLUMNS, FEATURE_DESCRIPTIONS, METRIC_NAMES,
                       MODEL_FILES, TARGET_COL, TEST_DATA_PATH, setup_logging)
from src.visuals import (METRIC_COLORS, plot_class_distribution, plot_confusion_matrix,
                         plot_defect_gauge, plot_feature_importance, plot_metric_bars,
                         plot_radar, plot_roc_curves)

setup_logging()
LOGGER = logging.getLogger("app")

st.set_page_config(page_title="Defect Radar", page_icon="🐞", layout="wide")

MODEL_BLURBS: Dict[str, str] = {
    "Logistic Regression": "Linear baseline that models defect probability with a sigmoid.",
    "Decision Tree": "Shallow rule-based tree (depth 4) on complexity metrics.",
    "KNN": "Votes among the 101 most similar modules in scaled metric space.",
    "Naive Bayes": "Gaussian likelihoods with a feature-independence assumption.",
    "SVM": "RBF-kernel margin classifier with calibrated probabilities.",
}

ROC_NOTE = (
    "AUC values around 0.7 are typical for the de-duplicated PROMISE/NASA data: static code metrics only "
    "weakly separate defective from clean modules. A curve hugging the top-left corner would indicate "
    "duplicate leakage rather than a better model."
)

CSS = """
<style>
.block-container {padding-top: 4.5rem; max-width: 1250px;}
.hero {background:#111B2E; border:1px solid #1F2A44; border-left:5px solid #F97316;
       border-radius:14px; padding:26px 30px; margin-bottom:22px;}
.hero h1 {margin:0 0 6px 0; font-size:2.1rem; letter-spacing:-0.5px;}
.hero p {margin:0; color:#94A3B8; font-size:1.02rem;}
.chip {display:inline-block; padding:3px 11px; border-radius:999px; font-size:0.74rem;
       font-weight:600; letter-spacing:.4px; background:rgba(249,115,22,.15); color:#FDBA74;
       border:1px solid rgba(249,115,22,.4); margin-bottom:10px;}
.chip.ok {background:rgba(163,230,53,.12); color:#BEF264; border-color:rgba(163,230,53,.4);}
.chip.warn {background:rgba(250,204,21,.12); color:#FDE047; border-color:rgba(250,204,21,.4);}
.kpi {background:#111B2E; border:1px solid #1F2A44; border-top:3px solid var(--c,#F97316);
      border-radius:12px; padding:16px 18px; height:100%;}
.kpi .l {color:#94A3B8; font-size:.8rem; text-transform:uppercase; letter-spacing:.8px;}
.kpi .v {font-size:1.9rem; font-weight:700; margin-top:2px;}
.kpi .s {color:#94A3B8; font-size:.82rem;}
.tile {background:#111B2E; border:1px solid #1F2A44; border-radius:12px; padding:14px 16px;}
.tile .l {color:#94A3B8; font-size:.78rem; text-transform:uppercase; letter-spacing:.8px;}
.tile .v {font-size:1.6rem; font-weight:700; color:var(--c);}
.bar {height:6px; background:#1F2A44; border-radius:6px; margin-top:8px; overflow:hidden;}
.bar > div {height:100%; background:var(--c); border-radius:6px;}
.winner {background:linear-gradient(90deg,rgba(250,204,21,.14),rgba(250,204,21,0));
         border:1px solid rgba(250,204,21,.35); border-radius:12px; padding:14px 18px; margin:6px 0 16px 0;}
.step {display:flex; gap:14px; align-items:flex-start; padding:10px 0; border-bottom:1px dashed #1F2A44;}
.step:last-child {border-bottom:none;}
.step .n {min-width:30px; height:30px; border-radius:8px; background:rgba(34,211,238,.14);
          color:#67E8F9; display:flex; align-items:center; justify-content:center; font-weight:700;}
.step b {display:block;} .step span {color:#94A3B8; font-size:.9rem;}
</style>
"""


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading models (first run may retrain them)...")
def get_artifacts() -> Artifacts:
    """Load (or rebuild) the persisted artifacts once per server process."""
    return ensure_artifacts()


@st.cache_data(show_spinner=False)
def run_evaluation(df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """Score ``df`` with one model; returns metrics and raw predictions."""
    artifacts = get_artifacts()
    model = artifacts.models[model_name]
    features, y_true = prepare_inputs(df, artifacts)
    if y_true is not None:
        outcome = evaluate_model(model, features, y_true)
    else:
        outcome = {"y_pred": model.predict(features), "y_score": positive_scores(model, features)}
    outcome["y_true"] = y_true
    return outcome


@st.cache_data(show_spinner="Scoring all five models...")
def run_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate all five models on ``df`` (requires the target column)."""
    artifacts = get_artifacts()
    features, y_true = prepare_inputs(df, artifacts)
    return compare_models(artifacts.models, features, y_true)


@st.cache_data(show_spinner=False)
def run_roc_curves(df: pd.DataFrame) -> Dict[str, Any]:
    """ROC data for all models on ``df``."""
    artifacts = get_artifacts()
    features, y_true = prepare_inputs(df, artifacts)
    curves = {}
    for name in MODEL_FILES:
        result = evaluate_model(artifacts.models[name], features, y_true)
        curves[name] = get_roc_data(y_true, result["y_score"])
    return curves


# --------------------------------------------------------------------------- #
# UI helpers
# --------------------------------------------------------------------------- #
def hero(title: str, subtitle: str, chip: str = "PROMISE · NASA MDP") -> None:
    """Render the page header block."""
    st.markdown(
        f'<div class="hero"><div class="chip">{chip}</div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, sub: str = "", color: str = "#F97316", size: str = "1.9rem") -> str:
    """Return the HTML of a KPI card."""
    return (f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
            f'<div class="v" style="font-size:{size}">{value}</div><div class="s">{sub}</div></div>')


def tile(metric: str, value: float) -> str:
    """Return the HTML of a metric tile with a progress bar."""
    color = METRIC_COLORS[metric]
    shown = "n/a" if np.isnan(value) else f"{value:.4f}"
    width = 0 if np.isnan(value) else max(0.0, min(1.0, value)) * 100
    return (
        f'<div class="tile" style="--c:{color}"><div class="l">{metric}</div>'
        f'<div class="v">{shown}</div>'
        f'<div class="bar"><div style="width:{width:.1f}%"></div></div></div>'
    )


def load_dataframe(source: Any) -> Optional[pd.DataFrame]:
    """Read a CSV from a path or an uploaded file, reporting errors in the UI."""
    try:
        return pd.read_csv(source)
    except pd.errors.EmptyDataError:
        st.error("The file is empty.")
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"The file could not be parsed as CSV: {exc}")
    except OSError as exc:
        st.error(f"The file could not be read: {exc}")
    return None


def set_dataset(df: pd.DataFrame, name: str) -> None:
    """Store the active dataset in the session and refresh the page if it changed.

    The sidebar status is drawn before the page runs, so a rerun is needed for it
    to reflect the new dataset.
    """
    fingerprint = (name, df.shape, int(pd.util.hash_pandas_object(df, index=False).sum()))
    changed = st.session_state.get("data_fingerprint") != fingerprint
    st.session_state["data"] = df
    st.session_state["data_name"] = name
    st.session_state["data_fingerprint"] = fingerprint
    LOGGER.info("Dataset '%s' loaded: %d rows", name, len(df))
    if changed:
        st.rerun()


def clear_dataset() -> None:
    """Forget the active dataset (used when validation fails)."""
    had_data = st.session_state.pop("data", None) is not None
    st.session_state.pop("data_name", None)
    st.session_state.pop("data_fingerprint", None)
    if had_data:
        st.rerun()


def require_data() -> pd.DataFrame:
    """Return the active dataset or stop the page with a call to action."""
    df = st.session_state.get("data")
    if df is not None:
        return df
    st.info("No test data loaded yet. Upload a CSV on **Upload & Validate**, or use the bundled sample.")
    if TEST_DATA_PATH.exists() and st.button("Use bundled test_data.csv", type="primary"):
        frame = load_dataframe(TEST_DATA_PATH)
        if frame is not None:
            set_dataset(frame, "test_data.csv (bundled)")
            st.rerun()
    st.stop()


def sidebar_status() -> None:
    """Show brand and dataset status in the sidebar."""
    with st.sidebar:
        st.markdown("### 🐞 Defect Radar")
        st.caption("Predict defect-prone software modules from static code metrics.")
        st.divider()
        df = st.session_state.get("data")
        if df is None:
            st.markdown('<span class="chip warn">NO DATA LOADED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="chip ok">DATA READY</span>', unsafe_allow_html=True)
            st.caption(f"{st.session_state['data_name']}  \n{len(df):,} rows × {df.shape[1]} columns")
        st.divider()
        st.caption("Models: Logistic Regression · Decision Tree · KNN · Naive Bayes · SVM")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    """Landing page with dataset facts and training results."""
    artifacts = get_artifacts()
    meta = artifacts.metadata
    report = meta["preprocessing"]
    table = artifacts.training_metrics
    best = rank_models(table).iloc[0]
    dist = report["class_distribution_clean"]
    defect_rate = dist[CLASS_LABELS[1]] / sum(dist.values())

    hero("Defect Radar", "Find the risky modules before your users do - five classifiers, "
         "one PROMISE dataset, one clean workflow.")

    cols = st.columns(4)
    cols[0].markdown(kpi("Clean records", f"{report['rows_final']:,}",
                         f"from {report['rows_loaded']:,} raw rows", "#22D3EE"), unsafe_allow_html=True)
    cols[1].markdown(kpi("Metrics / features", str(meta["n_features"]), "McCabe + Halstead", "#A78BFA"),
                     unsafe_allow_html=True)
    cols[2].markdown(kpi("Defective share", f"{defect_rate:.1%}", "class imbalance handled", "#F97316"),
                     unsafe_allow_html=True)
    best_note = f"MCC {best['MCC']:.3f} · F1 {best['F1']:.3f}"
    cols[3].markdown(kpi("Top model (MCC)", str(best["ML Model"]), best_note, "#FACC15", size="1.35rem"),
                     unsafe_allow_html=True)

    st.write("")
    left, right = st.columns([5, 6], gap="large")
    with left:
        st.subheader("How it works")
        steps = [
            ("Upload", "Drop a CSV with the 21 code metrics (and optionally the <code>defects</code> "
                       "label)."),
            ("Validate", "The schema is checked; missing cells are imputed with training medians."),
            ("Evaluate", "Pick a model to see metrics, confusion matrix, ROC and per-class report."),
            ("Compare", "Rank all five models by MCC, F1 and AUC and download the results."),
        ]
        html = "".join(
            f'<div class="step"><div class="n">{i}</div><div><b>{t}</b><span>{d}</span></div></div>'
            for i, (t, d) in enumerate(steps, 1)
        )
        st.markdown(html, unsafe_allow_html=True)
    with right:
        st.subheader("Class balance (cleaned data)")
        st.plotly_chart(plot_class_distribution(dist), width="stretch")
        st.caption(f"{report['duplicates_removed']:,} duplicates and {report['invalid_removed']:,} invalid "
                   f"records removed · stratified {int((1 - meta['test_size']) * 100)}/"
                   f"{int(meta['test_size'] * 100)} split · seed {meta['random_state']}")

    st.subheader("Hold-out results from training")
    st.dataframe(
        table.style.format({m: "{:.4f}" for m in METRIC_NAMES}).highlight_max(
            subset=METRIC_NAMES, color="#7C2D12"),
        hide_index=True, width="stretch",
    )
    with st.expander("Dataset details"):
        st.write(f"**Source:** {meta['source']}")
        st.write("**Projects:** " + ", ".join(f"{k} ({v:,})" for k, v in meta["projects"].items()))
        st.write(f"**Target:** `{TARGET_COL}` → 0 = {CLASS_LABELS[0]}, 1 = {CLASS_LABELS[1]}")
        st.write(f"**Balancing:** {report['balancing']}")


def page_upload() -> None:
    """Upload, validate and preview the test CSV."""
    hero("Upload & Validate", "Bring your own module metrics - the schema is verified before any model runs.",
         "STEP 1")
    left, right = st.columns([3, 2], gap="large")
    with left:
        uploaded = st.file_uploader("Drop test_data.csv here", type=["csv"],
                                    help="Must contain the 21 metric columns; `defects` is optional.")
        if uploaded is not None and st.session_state.get("last_upload_id") != uploaded.file_id:
            st.session_state["last_upload_id"] = uploaded.file_id
            frame = load_dataframe(uploaded)
            if frame is not None:
                report = validate_schema(frame)
                st.session_state["schema_report"] = report
                if report.valid:
                    set_dataset(frame, uploaded.name)
                else:
                    clear_dataset()
    with right:
        st.markdown("**No file handy?**")
        if TEST_DATA_PATH.exists():
            if st.button("Load bundled test_data.csv", width="stretch"):
                frame = load_dataframe(TEST_DATA_PATH)
                if frame is not None:
                    st.session_state["schema_report"] = validate_schema(frame)
                    set_dataset(frame, "test_data.csv (bundled)")
            st.download_button("Download test_data.csv", TEST_DATA_PATH.read_bytes(),
                               file_name="test_data.csv", mime="text/csv", width="stretch")

    report = st.session_state.get("schema_report")
    df = st.session_state.get("data")
    if report is not None:
        if report.valid:
            st.success("Schema looks good - head over to **Evaluate Model**.")
        else:
            st.error("Schema validation failed.")
        for message in report.messages:
            (st.warning if report.valid else st.error)(message)
    if df is not None:
        cols = st.columns(3)
        cols[0].metric("Rows", f"{len(df):,}")
        cols[1].metric("Columns", df.shape[1])
        cols[2].metric("Label column", "found" if TARGET_COL in df.columns else "absent")
        with st.expander("Preview uploaded data", expanded=True):
            st.dataframe(df.head(50), width="stretch")
    with st.expander("Expected schema"):
        schema = pd.DataFrame({"Column": FEATURE_COLUMNS + [TARGET_COL],
                               "Meaning": [FEATURE_DESCRIPTIONS[c] for c in FEATURE_COLUMNS]
                               + ["Optional label: 1/true = defective, 0/false = non defective"]})
        st.dataframe(schema, hide_index=True, width="stretch")


def page_evaluate() -> None:
    """Metrics, charts and predictions for a single selected model."""
    hero("Evaluate Model", "Choose a classifier and inspect how it performs on your test data.", "STEP 2")
    df = require_data()
    artifacts = get_artifacts()

    pick, blurb = st.columns([2, 3])
    model_name = pick.selectbox("Classifier", list(MODEL_FILES), key="model_choice")
    blurb.info(MODEL_BLURBS[model_name])

    result = run_evaluation(df, model_name)
    y_true, y_pred, y_score = result["y_true"], result["y_pred"], result["y_score"]
    total = len(y_pred)
    defective = int((y_pred == 1).sum())

    st.subheader("Prediction results")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1.5])
    c1.markdown(kpi("Total records", f"{total:,}", "modules analysed", "#22D3EE"), unsafe_allow_html=True)
    c2.markdown(kpi("Defective modules", f"{defective:,}", f"{defective / total:.1%} of total", "#F97316"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Non-defective", f"{total - defective:,}", f"{(total - defective) / total:.1%} of total",
                    "#A3E635"), unsafe_allow_html=True)
    c4.plotly_chart(plot_defect_gauge(defective / total), width="stretch")

    predictions = predict_dataframe(df, model_name, artifacts)
    tabs_names = ["Confusion Matrix", "ROC Curve", "Classification Report", "Predictions"]

    if y_true is None:
        st.warning("The uploaded file has no complete `defects` column, so metrics are unavailable. "
                   "Predictions are still shown below.")
        (tab_pred,) = st.tabs(["Predictions"])
        with tab_pred:
            render_predictions(predictions, model_name)
        return

    st.subheader("Evaluation metrics")
    metrics = result["metrics"]
    for col, name in zip(st.columns(6), METRIC_NAMES):
        col.markdown(tile(name, metrics[name]), unsafe_allow_html=True)
    st.write("")

    tab_cm, tab_roc, tab_rep, tab_pred = st.tabs(tabs_names)
    with tab_cm:
        left, right = st.columns([3, 2])
        matrix = get_confusion_matrix(y_true, y_pred)
        left.pyplot(plot_confusion_matrix(matrix), width="content")
        tn, fp, fn, tp = matrix.ravel()
        right.markdown("#### Reading the matrix")
        right.markdown(f"- **True negatives:** {tn:,} clean modules correctly cleared\n"
                       f"- **False positives:** {fp:,} clean modules flagged\n"
                       f"- **False negatives:** {fn:,} defective modules missed\n"
                       f"- **True positives:** {tp:,} defects caught")
    with tab_roc:
        if len(np.unique(y_true)) < 2:
            st.warning("ROC needs both classes in the data.")
        else:
            st.plotly_chart(plot_roc_curves({model_name: get_roc_data(y_true, y_score)}),
                            width="stretch")
            st.caption(ROC_NOTE)
    with tab_rep:
        report = get_classification_report(y_true, y_pred)
        st.dataframe(report.style.format({"precision": "{:.4f}", "recall": "{:.4f}",
                                          "f1-score": "{:.4f}", "support": "{:,.0f}"}),
                     width="stretch")
        st.caption("Support = number of true records per class (accuracy row repeats the total).")
    with tab_pred:
        render_predictions(predictions, model_name)

    with st.expander("Model hyper-parameters"):
        st.json(artifacts.metadata["hyperparameters"][model_name])


def render_predictions(predictions: pd.DataFrame, model_name: str) -> None:
    """Show filtered predictions and a download button."""
    choice = st.radio("Show", ["All", CLASS_LABELS[1], CLASS_LABELS[0]], horizontal=True)
    view = predictions if choice == "All" else predictions[predictions["Predicted_Label"] == choice]
    if "Defect_Probability" in view:
        view = view.sort_values("Defect_Probability", ascending=False)
    st.caption(f"{len(view):,} rows (highest risk first)")
    st.dataframe(view.head(500), width="stretch")
    st.download_button("⬇ Download predictions (CSV)", predictions.to_csv(index=False).encode("utf-8"),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv", mime="text/csv")


def page_compare() -> None:
    """Dashboard comparing all five models."""
    hero("Compare Models", "All five classifiers scored on the same data and ranked by MCC → F1 → AUC.",
         "STEP 3")
    df = require_data()
    artifacts = get_artifacts()
    if extract_missing_target(df):
        st.warning("Comparison needs a complete `defects` column in the uploaded file.")
        st.stop()

    table = run_comparison(df)
    ranked = rank_models(table)
    top = ranked.iloc[0]
    st.markdown(
        f'<div class="winner">🏆 <b>{top["ML Model"]}</b> leads the leaderboard - '
        f'MCC {top["MCC"]:.4f} · F1 {top["F1"]:.4f} · AUC {top["AUC"]:.4f}</div>',
        unsafe_allow_html=True,
    )

    podium = st.columns(len(ranked))
    medals = ["🥇", "🥈", "🥉", "#4", "#5"]
    for col, (_, row), medal in zip(podium, ranked.iterrows(), medals):
        col.markdown(kpi(f"{medal} Rank {row['Rank']}", str(row["ML Model"]), f"MCC {row['MCC']:.3f}",
                         "#FACC15" if row["Rank"] == 1 else "#334155", size="1.2rem"),
                     unsafe_allow_html=True)
    st.write("")

    t_table, t_bar, t_radar, t_roc, t_train = st.tabs(
        ["Leaderboard", "Performance chart", "Radar", "ROC curves", "Training hold-out"])
    with t_table:
        st.dataframe(
            ranked.style.format({m: "{:.4f}" for m in METRIC_NAMES}).highlight_max(
                subset=METRIC_NAMES, color="#7C2D12"),
            hide_index=True, width="stretch",
        )
        st.caption("Highlighted cell = best value per metric.")
        st.download_button("⬇ Download comparison (CSV)", ranked.to_csv(index=False).encode("utf-8"),
                           file_name="model_comparison.csv", mime="text/csv")
    with t_bar:
        st.plotly_chart(plot_metric_bars(table), width="stretch")
    with t_radar:
        st.plotly_chart(plot_radar(table), width="stretch")
    with t_roc:
        st.plotly_chart(plot_roc_curves(run_roc_curves(df), height=480), width="stretch")
        st.caption(ROC_NOTE)
    with t_train:
        st.caption("Metrics recorded when the models were trained (stratified hold-out split).")
        st.dataframe(artifacts.training_metrics.style.format({m: "{:.4f}" for m in METRIC_NAMES}),
                     hide_index=True, width="stretch")


def extract_missing_target(df: pd.DataFrame) -> bool:
    """Return True when ``df`` lacks a usable label column."""
    return TARGET_COL not in df.columns or df[TARGET_COL].isna().any()


def page_insights() -> None:
    """Feature importance and data-cleaning transparency."""
    hero("Insights", "Which code metrics push a module towards 'defective'?", "EXPLAIN")
    artifacts = get_artifacts()

    tab_tree, tab_lr, tab_clean = st.tabs(["Decision Tree importance", "Logistic Regression weights",
                                           "Data cleaning log"])
    with tab_tree:
        top_n = st.slider("Features to show", 5, len(FEATURE_COLUMNS), 8)
        tree = artifacts.models["Decision Tree"]
        st.plotly_chart(plot_feature_importance(tree.feature_importances_, FEATURE_COLUMNS, top_n),
                        width="stretch")
        with st.expander("What do these metrics mean?"):
            st.dataframe(pd.DataFrame({"Feature": FEATURE_COLUMNS,
                                       "Description": [FEATURE_DESCRIPTIONS[c] for c in FEATURE_COLUMNS],
                                       "Importance": tree.feature_importances_.round(4)})
                         .sort_values("Importance", ascending=False),
                         hide_index=True, width="stretch")
    with tab_lr:
        coefs = artifacts.models["Logistic Regression"].coef_[0]
        st.plotly_chart(plot_feature_importance(np.abs(coefs), FEATURE_COLUMNS, 12), width="stretch")
        st.caption("Absolute standardised coefficients - larger means a stronger linear effect.")
    with tab_clean:
        report = artifacts.metadata["preprocessing"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw rows", f"{report['rows_loaded']:,}")
        c2.metric("Duplicates removed", f"{report['duplicates_removed']:,}")
        c3.metric("Invalid removed", f"{report['invalid_removed']:,}")
        st.dataframe(pd.DataFrame(report["invalid_by_rule"].items(), columns=["Invalid-record rule", "Rows"]),
                     hide_index=True, width="stretch")
        st.caption(f"{report['missing_cells']:,} missing cells were imputed with training-set medians.")


def page_about() -> None:
    """Project background."""
    hero("About", "Software defect prediction with the PROMISE repository.", "README")
    st.markdown(
        """
**Goal** - classify a software module as *defective* (1) or *non-defective* (0) from McCabe and Halstead
static code metrics.

**Data** - PROMISE / NASA MDP projects JM1, KC1, KC2 and PC1 (21 shared metrics), merged and de-duplicated.

**Pipeline** - missing values → median imputation · duplicate and invalid-record removal · stratified split ·
`log1p` + `StandardScaler` · random over-sampling of the training set · five classifiers with
cross-validated hyper-parameters and a fixed seed.

**Metrics** - Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). MCC is used as
the primary ranking criterion because it stays reliable on imbalanced data.
"""
    )


def main() -> None:
    """Register pages and run the selected one."""
    st.markdown(CSS, unsafe_allow_html=True)
    sidebar_status()
    pages = [
        st.Page(page_home, title="Home", icon=":material/home:", default=True, url_path="home"),
        st.Page(page_upload, title="Upload & Validate", icon=":material/upload_file:", url_path="upload"),
        st.Page(page_evaluate, title="Evaluate Model", icon=":material/monitoring:", url_path="evaluate"),
        st.Page(page_compare, title="Compare Models", icon=":material/leaderboard:", url_path="compare"),
        st.Page(page_insights, title="Insights", icon=":material/insights:", url_path="insights"),
        st.Page(page_about, title="About", icon=":material/info:", url_path="about"),
    ]
    st.navigation(pages).run()


main()
