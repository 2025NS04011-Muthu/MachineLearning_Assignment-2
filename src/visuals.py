"""Chart builders (Matplotlib/Seaborn and Plotly) used by the Streamlit app."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from src.utils import CLASS_LABELS, METRIC_NAMES  # noqa: E402

ACCENT = "#F97316"
CYAN = "#22D3EE"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
GRID = "rgba(148,163,184,0.18)"

MODEL_COLORS: Dict[str, str] = {
    "Logistic Regression": "#22D3EE",
    "Decision Tree": "#F97316",
    "KNN": "#A3E635",
    "Naive Bayes": "#F472B6",
    "SVM": "#A78BFA",
}
METRIC_COLORS: Dict[str, str] = {
    "Accuracy": "#22D3EE",
    "AUC": "#A78BFA",
    "Precision": "#A3E635",
    "Recall": "#F472B6",
    "F1": "#F97316",
    "MCC": "#FACC15",
}


def _style(fig: go.Figure, height: int = 420, title: str = "") -> go.Figure:
    """Apply the shared dark, transparent Plotly look."""
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title={"text": title, "x": 0.0, "font": {"size": 16, "color": TEXT}} if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": TEXT},
        margin={"l": 40, "r": 20, "t": 60 if title else 20, "b": 40},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def plot_confusion_matrix(matrix: np.ndarray, title: str = "") -> plt.Figure:
    """Seaborn heatmap of a 2x2 confusion matrix with counts and row percentages.

    Args:
        matrix: Confusion matrix (rows = actual, columns = predicted). Colours show
            the row-normalised share so both classes are equally visible.
        title: Optional title.
    """
    labels = [CLASS_LABELS[0], CLASS_LABELS[1]]
    row_totals = matrix.sum(axis=1, keepdims=True).clip(min=1)
    shares = matrix / row_totals
    annotations = np.array(
        [[f"{matrix[i, j]:,}\n{shares[i, j]:.1%}" for j in range(2)] for i in range(2)]
    )
    cmap = LinearSegmentedColormap.from_list("radar", ["#16233A", "#C2410C", ACCENT, "#FDBA74"])
    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    fig.patch.set_alpha(0.0)
    axis.set_facecolor("none")
    sns.heatmap(
        shares, annot=annotations, fmt="", cmap=cmap, vmin=0, vmax=1, cbar=False, linewidths=3,
        linecolor="#0B1220", xticklabels=labels, yticklabels=labels, ax=axis,
        annot_kws={"fontsize": 13, "fontweight": "bold", "color": "white"},
    )
    axis.set_xlabel("Predicted", color=TEXT, labelpad=10)
    axis.set_ylabel("Actual", color=TEXT, labelpad=10)
    axis.tick_params(colors=TEXT)
    if title:
        axis.set_title(title, color=TEXT, pad=12)
    fig.tight_layout()
    return fig


def plot_roc_curves(
    curves: Mapping[str, Tuple[np.ndarray, np.ndarray, float]], height: int = 440
) -> go.Figure:
    """ROC curves for one or several models.

    Args:
        curves: Mapping ``model -> (fpr, tpr, auc)``.
    """
    fig = go.Figure()
    for name, (fpr, tpr, auc) in curves.items():
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                line={"color": MODEL_COLORS.get(name, ACCENT), "width": 3},
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess",
                   line={"color": MUTED, "dash": "dash"})
    )
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return _style(fig, height)


def plot_metric_bars(table: pd.DataFrame, height: int = 440) -> go.Figure:
    """Grouped bar chart: one cluster per model, one bar per metric."""
    fig = go.Figure()
    for metric in METRIC_NAMES:
        fig.add_trace(
            go.Bar(
                x=table["ML Model"], y=table[metric], name=metric,
                marker_color=METRIC_COLORS[metric],
                text=table[metric].round(2), textposition="outside",
            )
        )
    fig.update_layout(barmode="group", bargap=0.18)
    fig.update_yaxes(range=[0, 1.1], title="Score")
    return _style(fig, height)


def plot_radar(table: pd.DataFrame, height: int = 460) -> go.Figure:
    """Radar chart comparing all models over the six metrics."""
    fig = go.Figure()
    for _, row in table.iterrows():
        values = [row[m] for m in METRIC_NAMES]
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1], theta=METRIC_NAMES + METRIC_NAMES[:1],
                name=row["ML Model"], fill="toself", opacity=0.55,
                line={"color": MODEL_COLORS.get(row["ML Model"], ACCENT)},
            )
        )
    fig.update_layout(polar={"bgcolor": "rgba(0,0,0,0)", "radialaxis": {"range": [0, 1], "gridcolor": GRID},
                             "angularaxis": {"gridcolor": GRID}})
    return _style(fig, height)


def plot_feature_importance(
    importances: Sequence[float], names: Sequence[str], top_n: int = 15, height: int = 480
) -> go.Figure:
    """Horizontal bar chart of the most important features.

    Args:
        importances: Importance value per feature.
        names: Feature names aligned with ``importances``.
        top_n: Number of features to show.
    """
    frame = (
        pd.DataFrame({"Feature": names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(top_n)
        .iloc[::-1]
    )
    fig = go.Figure(
        go.Bar(
            x=frame["Importance"], y=frame["Feature"], orientation="h",
            marker={"color": frame["Importance"], "colorscale": [[0, "#C2410C"], [1, "#FDBA74"]]},
            text=frame["Importance"].round(3), textposition="outside",
        )
    )
    fig.update_xaxes(title="Gini importance")
    return _style(fig, height)


def plot_defect_gauge(rate: float, height: int = 250) -> go.Figure:
    """Gauge showing the share of modules predicted defective.

    Args:
        rate: Fraction in ``[0, 1]``.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=rate * 100,
            number={"suffix": "%", "font": {"size": 38, "color": TEXT}, "valueformat": ".1f"},
            title={"text": "Predicted defect rate", "font": {"size": 14, "color": MUTED}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": MUTED},
                "bar": {"color": ACCENT},
                "bgcolor": "rgba(148,163,184,0.12)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 20], "color": "rgba(163,230,53,0.25)"},
                    {"range": [20, 40], "color": "rgba(250,204,21,0.25)"},
                    {"range": [40, 100], "color": "rgba(244,63,94,0.25)"},
                ],
            },
        )
    )
    fig.update_layout(height=height, margin={"l": 40, "r": 50, "t": 50, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)", font={"color": TEXT})
    return fig


def plot_class_distribution(distribution: Mapping[str, int], height: int = 220) -> go.Figure:
    """Single stacked horizontal bar showing the class balance."""
    colors = {CLASS_LABELS[0]: CYAN, CLASS_LABELS[1]: ACCENT}
    fig = go.Figure()
    for label, count in distribution.items():
        fig.add_trace(
            go.Bar(y=["Modules"], x=[count], name=label, orientation="h",
                   marker_color=colors.get(label, MUTED), text=f"{label}: {count:,}",
                   textposition="inside")
        )
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height)
