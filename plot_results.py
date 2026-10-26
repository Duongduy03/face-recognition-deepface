"""Vẽ biểu đồ từ results/*.csv do benchmark.py tạo ra (không tính lại gì cả).

    python plot_results.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.metrics import confusion_matrix

from config import RESULTS_DIR

# Palette tham chiếu (light): nhấn mạnh = xanh + xám; 3 slot phân loại đầu = xanh/cam/aqua
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACCENT = "#2a78d6"
DEEMPH = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)

METRICS = [
    ("accuracy", "Accuracy"),
    ("precision_macro", "Precision"),
    ("recall_macro", "Recall"),
    ("f1_macro", "F1"),
    ("top3_accuracy", "Top-3"),
    ("cohen_kappa", "Kappa"),
]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
})


def _out(name):
    return os.path.join(RESULTS_DIR, name)


def _style_axes(ax, grid_axis="x"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_f1_ranking(metrics, best):
    df = metrics.sort_values("f1_macro")
    colors = [ACCENT if m in best else DEEMPH for m in df["model"]]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(df["model"], df["f1_macro"], color=colors, height=0.62)
    _style_axes(ax, "x")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 (macro) trên tập test")
    ax.set_title(f"Xếp hạng {len(df)} mô hình theo F1 (2 mô hình tốt nhất được nhấn màu)", loc="left", fontsize=11, color=INK)
    for y_pos, (model, v) in enumerate(zip(df["model"], df["f1_macro"])):
        if model in best:
            ax.text(v + 0.01, y_pos, f"{v:.3f}", va="center", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(_out("f1_ranking.png"), dpi=150)
    plt.close(fig)


def plot_metrics_heatmap(metrics):
    df = metrics.sort_values("f1_macro", ascending=False)
    values = df[[c for c, _ in METRICS]].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.imshow(values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)), [label for _, label in METRICS])
    ax.set_yticks(range(len(df)), df["model"])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if v > 0.6 else INK)
    # khe 2px giữa các ô
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("6 độ đo trên tập test (thang 0–1, đậm hơn = tốt hơn)", loc="left", fontsize=11, pad=28)
    fig.tight_layout()
    fig.savefig(_out("metrics_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_speed_vs_f1(metrics, fps, best):
    df = metrics.merge(fps, on="model")
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # 4 mô hình đầu có F1 và FPS sát nhau nên nhãn được đặt lệch tay để không đè lên nhau
    offsets = {
        "Facenet512": (-6, 12, "right"),
        "Facenet": (0, 34, "center"),
        "VGG-Face": (4, 16, "left"),
        "ArcFace": (10, -3, "left"),
    }
    for r in df.itertuples():
        top = r.model in best
        ax.scatter(r.fps_embed_knn, r.f1_macro, s=70, color=ACCENT if top else DEEMPH,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        dx, dy, ha = offsets.get(r.model, (7, 4, "left"))
        ax.annotate(r.model, (r.fps_embed_knn, r.f1_macro), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, fontsize=8.5, color=INK if top else INK_2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=0, shrinkB=3)
                    if r.model == "Facenet" else None)
    ax.set_ylim(df["f1_macro"].min() - 0.05, 1.1)
    ax.set_xscale("log")
    lo, hi = df["fps_embed_knn"].min() / 1.5, df["fps_embed_knn"].max() * 1.5
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    _style_axes(ax, "both")
    ax.set_xlabel("FPS của bước embed + KNN (thang log, cao hơn = nhanh hơn)")
    ax.set_ylabel("F1 (macro)")
    ax.set_title("Độ chính xác so với tốc độ", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(_out("speed_vs_f1.png"), dpi=150)
    plt.close(fig)


def _stacked_barh(ax, labels, parts, names, colors=SERIES):
    left = np.zeros(len(labels))
    for vals, name, color in zip(parts, names, colors):
        ax.barh(labels, vals, left=left, color=color, label=name, height=0.62,
                edgecolor=SURFACE, linewidth=1.5)
        left += np.array(vals, dtype=float)


def _legend_below(fig, ax, ncol):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=9)


def plot_time_breakdown(fps):
    # Màu theo bước (không theo thứ hạng): detect = xanh, embed = cam, KNN = aqua
    df = fps.sort_values("fps_embed_knn")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=True)
    _stacked_barh(ax1, df["model"], [df["embed_ms"], df["knn_ms"]],
                  ["Embed", "KNN"], colors=[SERIES[1], SERIES[2]])
    _stacked_barh(ax2, df["model"], [df["detect_ms"], df["embed_ms"], df["knn_ms"]],
                  ["Detect (RetinaFace)", "Embed", "KNN"], colors=SERIES)
    for ax, title in ((ax1, "Chỉ embed + KNN"), (ax2, "Cả pipeline (gồm detect)")):
        _style_axes(ax, "x")
        ax.set_xlabel("ms mỗi ảnh (thấp hơn = nhanh hơn)")
        ax.set_title(title, loc="left", fontsize=11)
    ax2.tick_params(left=False)
    _legend_below(fig, ax2, ncol=3)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(_out("time_breakdown.png"), dpi=150)
    plt.close(fig)


def plot_failure_breakdown(summary, metrics):
    order = list(metrics.sort_values("f1_macro", ascending=False)["model"])
    df = summary.set_index("model").loc[order].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _stacked_barh(
        ax, df.index,
        [df["n_misidentified"], df["n_no_face"] + df["n_bad_crop"], df["n_rejected_unknown"]],
        ["Nhận nhầm người", "Lỗi detect/căn chỉnh (không thấy mặt hoặc crop hỏng)",
         "Bị loại thành Unknown (vượt ngưỡng)"],
    )
    _style_axes(ax, "x")
    ax.set_xlabel("Số ca lỗi trên tập test")
    ax.set_title("Các loại ca nhận diện lỗi theo mô hình", loc="left", fontsize=11)
    _legend_below(fig, ax, ncol=1)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(_out("failure_breakdown.png"), dpi=150)
    plt.close(fig)


def plot_confusion(preds, model):
    p = preds[preds["model"] == model]
    labels = sorted(set(p["true"]) | set(p["pred"]))
    cm = confusion_matrix(p["true"], p["pred"], labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(cm, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Ma trận nhầm lẫn — {model} ({len(labels)} người, chuẩn hóa theo hàng)", loc="left", fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(_out(f"confusion_{model}.png"), dpi=150)
    plt.close(fig)


def main():
    metrics = pd.read_csv(_out("metrics.csv"))
    best = json.load(open(_out("best_models.json"), encoding="utf-8"))
    plot_f1_ranking(metrics, best)
    plot_metrics_heatmap(metrics)
    if os.path.exists(_out("fps.csv")):
        fps = pd.read_csv(_out("fps.csv"))
        plot_speed_vs_f1(metrics, fps, best)
        plot_time_breakdown(fps)
    plot_failure_breakdown(pd.read_csv(_out("failure_summary.csv")), metrics)
    preds = pd.read_csv(_out("predictions.csv"))
    for model in best:
        plot_confusion(preds, model)
    print("Đã vẽ biểu đồ vào", RESULTS_DIR)


if __name__ == "__main__":
    main()
