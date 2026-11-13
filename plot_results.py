"""Vẽ biểu đồ từ file CSV do benchmark tạo ra (không tính lại gì cả).

    python plot_results.py                       # so sánh 10 thuật toán: đọc/ghi results/algorithms/ (benchmark.py)
    python plot_results.py --kind embeddings     # so sánh 9 embedding (phụ lục): đọc/ghi results/ (benchmark_embeddings.py)
    python plot_results.py --dir output/benchmark_quick   # vẽ kết quả của lần chạy --quick
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.metrics import confusion_matrix

from config import ALGO_RESULTS_DIR, RESULTS_DIR

OUT_DIR = RESULTS_DIR   # main() đổi sang thư mục của loại so sánh đang vẽ

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
    return os.path.join(OUT_DIR, name)


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


def plot_metrics_heatmap(metrics, label_col="model", filename="metrics_heatmap.png"):
    df = metrics.sort_values("f1_macro", ascending=False)
    values = df[[c for c, _ in METRICS]].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.imshow(values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(METRICS)), [label for _, label in METRICS])
    ax.set_yticks(range(len(df)), df[label_col])
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
    fig.savefig(_out(filename), dpi=150)
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


def plot_confusion(preds, model, label_col="model"):
    p = preds[preds[label_col] == model]
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


# =========================
# SO SÁNH 10 THUẬT TOÁN (results/algorithms/)
# =========================
# 10 thuật toán không dùng 10 màu: 2 thuật toán demo nhấn màu ACCENT, phần còn lại xám (kiểu "emphasis").
# Các hình dùng chung một thứ tự thuật toán: F1 macro của lần chia 80/20 giảm dần (cùng thứ tự với metrics.csv).
MARK = MUTED    # chấm / đường sai số của thuật toán không được nhấn (đậm hơn DEEMPH để còn nhìn rõ trên nền sáng)
FMT_G = FuncFormatter(lambda v, _: f"{v:g}")


def _heading(fig, title, subtitle=None):
    fig.text(0.012, 0.985, title, fontsize=12, color=INK, va="top", fontweight="bold")
    if subtitle:
        fig.text(0.012, 0.995 - 0.058, subtitle, fontsize=9, color=INK_2, va="top")


def _log_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FMT_G)
    ax.xaxis.set_minor_formatter(NullFormatter())


def plot_alg_f1_ranking(metrics, cv, best, subtitle):
    """Trái: F1 một lần chia trên toàn thang 0-1. Phải: phóng to 0,895-1, thêm trung bình ± độ lệch chuẩn của 5-fold CV."""
    lo = 0.895
    df = metrics.sort_values("f1_macro").reset_index(drop=True)      # F1 thấp ở đáy
    names = list(df["algorithm"])
    y = np.arange(len(names))
    c = cv.set_index("algorithm")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.6), sharey=True, gridspec_kw={"width_ratios": [1, 1.35]})

    ax1.barh(y, df["f1_macro"], color=[ACCENT if n in best else DEEMPH for n in names], height=0.6)
    for yi, n, v in zip(y, names, df["f1_macro"]):
        if n in best:
            ax1.text(v + 0.012, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax1.set_yticks(y, names)
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel("F1 macro, một lần chia 80/20")
    ax1.set_title("Toàn thang 0–1", loc="left", fontsize=10.5)
    _style_axes(ax1, "x")

    for yi, n, split in zip(y, names, df["f1_macro"]):
        col = ACCENT if n in best else MARK
        m = c.loc[n]
        if m["cv_f1_mean"] >= lo:
            ax2.errorbar(m["cv_f1_mean"], yi, xerr=m["cv_f1_std"], fmt="o", color=col, ecolor=col, elinewidth=1.4,
                         capsize=0, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        else:
            ax2.text(lo + 0.002, yi, f"ngoài khung: CV {m['cv_f1_mean']:.2f}, một lần chia {split:.2f}",
                     va="center", fontsize=8.5, color=INK_2)
        if split >= lo:
            # hình thoi rỗng (không tô nền) để không che chấm CV khi hai giá trị gần nhau
            ax2.scatter(split, yi, marker="D", s=34, facecolor="none", edgecolor=col, linewidth=1.5, zorder=2)
        if n in best and m["cv_f1_mean"] >= lo:
            ax2.text(max(m["cv_f1_mean"] + m["cv_f1_std"], split) + 0.002, yi, f"{m['cv_f1_mean']:.3f}",
                     va="center", fontsize=9, color=INK)
    ax2.set_xlim(lo, 1.0)
    ax2.set_xticks(np.arange(0.90, 1.001, 0.02))
    ax2.set_xlabel("F1 macro")
    ax2.set_title(f"Phóng to {lo:.3f}–1.000", loc="left", fontsize=10.5)
    _style_axes(ax2, "x")
    ax2.tick_params(left=False)

    handles = [
        Line2D([0], [0], marker="o", color=MARK, lw=1.4, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2,
               label="5-fold CV: trung bình ± độ lệch chuẩn (số ghi cạnh chấm)"),
        Line2D([0], [0], marker="D", color=MARK, lw=0, markersize=6, markerfacecolor=SURFACE, markeredgewidth=1.5,
               label="Một lần chia 80/20"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9)
    _heading(fig, "F1 macro của 10 thuật toán phân loại (cùng embedding)", subtitle)
    fig.tight_layout(rect=[0, 0.07, 1, 0.9])
    fig.savefig(_out("f1_ranking.png"), dpi=150)
    plt.close(fig)


def plot_alg_cost(timing, metrics, best, subtitle):
    """Thời gian huấn luyện, thời gian phân loại một vector và kích thước mô hình. Chấm trên trục log (không dùng thanh)."""
    order = list(metrics.sort_values("f1_macro", ascending=False)["algorithm"])[::-1]
    t = timing.set_index("algorithm").loc[order]
    y = np.arange(len(order))
    cols = [ACCENT if n in best else MARK for n in order]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 5.6), sharey=True)

    ax = axes[0]
    ax.scatter(t["fit_s"], y, c=cols, s=75, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_title("Huấn luyện (giây)", loc="left", fontsize=10.5)

    ax = axes[1]
    for yi, p, s in zip(y, t["predict_ms"], t["scores_ms"]):
        ax.plot([p, s], [yi, yi], color=AXIS, lw=1.2, zorder=1)
    # vòng rỗng (scores) to hơn chấm đặc (predict): hai giá trị trùng nhau thì chấm nằm gọn trong vòng
    ax.scatter(t["scores_ms"], y, s=190, facecolors="none", edgecolors=cols, linewidths=1.6, zorder=2)
    ax.scatter(t["predict_ms"], y, c=cols, s=60, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for yi, p, s in zip(y, t["predict_ms"], t["scores_ms"]):
        if s > 10 * p:     # chỉ ghi số ở chỗ chênh lệch lớn
            ax.text(s * 1.7, yi, f"{s:.0f} ms", va="center", fontsize=8.5, color=INK_2)
    ax.set_title("Phân loại một vector (ms)", loc="left", fontsize=10.5)
    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MARK, markeredgecolor=SURFACE, markeredgewidth=1.5,
               markersize=7, label="chấm đặc (giữa): predict, chỉ lấy nhãn"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=MARK, markeredgewidth=1.6,
               markersize=12, label="vòng rỗng: scores, lấy cả độ tin cậy để loại Unknown"),
    ], loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=2, frameon=False, fontsize=9)

    ax = axes[2]
    ax.scatter(t["model_mb"], y, c=cols, s=75, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_title("Kích thước mô hình lưu (MB)", loc="left", fontsize=10.5)

    for ax in axes:
        _log_axis(ax)
        _style_axes(ax, "x")
        ax.set_ylim(-0.7, len(order) - 0.3)
    axes[0].set_yticks(y, order)
    for ax in axes[1:]:
        ax.tick_params(left=False)
    # ghi số ở giá trị lớn nhất của hai hình ngoài cùng
    for ax, col in ((axes[0], "fit_s"), (axes[2], "model_mb")):
        i = int(np.argmax(t[col].to_numpy()))
        ax.text(t[col].iloc[i] * 0.8, y[i], f"{t[col].iloc[i]:.0f}", va="center", ha="right", fontsize=8.5, color=INK_2)
    fig.text(0.5, 0.015, "Thang log; chấm xanh = 2 thuật toán demo", ha="center", fontsize=9, color=INK_2)
    _heading(fig, "Chi phí của 10 thuật toán", subtitle)
    fig.tight_layout(rect=[0, 0.1, 1, 0.9])
    fig.savefig(_out("time_and_size.png"), dpi=150)
    plt.close(fig)


def _place_labels(fig, ax, xs, ys, names, colors, yerr=None):
    """Đặt nhãn từng điểm ở vị trí đầu tiên không chồng nhãn khác, không đè điểm (và thanh sai số dọc) và không tràn khung."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    placed = [matplotlib.transforms.Bbox.from_extents(px - 8, py - 8, px + 8, py + 8) for px, py in pts]
    if yerr is not None:
        for x, y_, s in zip(xs, ys, yerr):
            (px, lo_y), (_, hi_y) = ax.transData.transform([(x, y_ - s), (x, y_ + s)])
            placed.append(matplotlib.transforms.Bbox.from_extents(px - 3, lo_y, px + 3, hi_y))
    axbox = ax.get_window_extent(renderer)
    cands = [(9, 0, "left"), (-9, 0, "right"), (0, 11, "center"), (0, -13, "center"),
             (10, 12, "left"), (-10, 12, "right"), (10, -14, "left"), (-10, -14, "right"),
             (0, 24, "center"), (0, -27, "center"), (24, 0, "left"), (-24, 0, "right"),
             (20, 26, "left"), (-20, 26, "right"), (20, -28, "left"), (-20, -28, "right")]

    def try_label(i, cand):
        dx, dy, ha = cand
        far = abs(dx) >= 20 or abs(dy) >= 24
        ann = ax.annotate(names[i], (xs[i], ys[i]), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                          fontsize=8.5, color=colors[i],
                          arrowprops=dict(arrowstyle="-", color=MARK, lw=0.6, shrinkA=0, shrinkB=5) if far else None)
        ann.update_positions(renderer)    # bắt buộc: không có bước này hộp bao của Annotation bị tính sai vị trí
        bb = matplotlib.text.Text.get_window_extent(ann, renderer).expanded(1.04, 1.2)
        inside = axbox.x0 <= bb.x0 and bb.x1 <= axbox.x1 and axbox.y0 <= bb.y0 and bb.y1 <= axbox.y1
        # điểm phạt: số vật bị đè (+50 nếu tràn khung); 0 = chỗ trống hoàn toàn
        return ann, bb, (0 if inside else 50) + sum(bb.overlaps(o) for o in placed)

    for i in np.argsort(xs):
        best = None
        for cand in cands:
            ann, bb, penalty = try_label(i, cand)
            if best is None or penalty < best[2]:
                if best is not None:
                    best[0].remove()
                best = (ann, bb, penalty)
            else:
                ann.remove()
            if penalty == 0:
                break
        placed.append(best[1])     # không còn chỗ trống thì giữ vị trí ít đè nhất
        if best[2] and os.environ.get("PLOT_DEBUG"):
            print(f"  nhãn {names[i]}: không có chỗ trống hoàn toàn (điểm phạt {best[2]})")


def plot_alg_speed_vs_f1(metrics, cv, timing, best, subtitle):
    """F1 (5-fold CV, trung bình ± độ lệch chuẩn) so với thời gian predict một vector."""
    ylo, yhi = 0.895, 0.995
    d = cv.merge(timing[["algorithm", "predict_ms"]], on="algorithm")
    off = d[d["cv_f1_mean"] < ylo]
    d = d[d["cv_f1_mean"] >= ylo].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    colors_pt = [ACCENT if n in best else MARK for n in d["algorithm"]]
    for x, yv, s, col in zip(d["predict_ms"], d["cv_f1_mean"], d["cv_f1_std"], colors_pt):
        ax.errorbar(x, yv, yerr=s, fmt="o", color=col, ecolor=col, elinewidth=1.2, capsize=0, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    ax.set_xscale("log")
    ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20) if 0.05 <= t <= 20]
    ax.set_xlim(0.05, 20)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FMT_G)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(ylo, yhi)
    _style_axes(ax, "both")
    ax.set_xlabel("Thời gian predict một vector (ms, thang log; trái = nhanh hơn)")
    ax.set_ylabel("F1 macro, 5-fold CV (cao hơn = tốt hơn)")
    fig.tight_layout(rect=[0, 0.06, 1, 0.9])
    labels_col = [INK if n in best else INK_2 for n in d["algorithm"]]
    _place_labels(fig, ax, d["predict_ms"].to_numpy(), d["cv_f1_mean"].to_numpy(), list(d["algorithm"]), labels_col,
                  yerr=d["cv_f1_std"].to_numpy())
    if len(off):
        txt = "; ".join(f"{r.algorithm} (F1 {r.cv_f1_mean:.2f}, {r.predict_ms:.2f} ms)" for r in off.itertuples())
        fig.text(0.5, 0.015, f"Ngoài khung: {txt}", ha="center", fontsize=9, color=INK_2)
    _heading(fig, "Độ chính xác so với tốc độ phân loại (thanh dọc = ± độ lệch chuẩn CV)", subtitle)
    fig.savefig(_out("speed_vs_f1.png"), dpi=150)
    plt.close(fig)


def plot_alg_failures(summary, metrics, subtitle):
    order = list(metrics.sort_values("f1_macro", ascending=False)["algorithm"])
    df = summary.set_index("algorithm").loc[order].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.6, 5.9))
    _stacked_barh(
        ax, df.index,
        [df["n_misidentified"], df["n_no_face"] + df["n_bad_crop"], df["n_rejected_unknown"]],
        ["Nhận nhầm người", "Lỗi detect/căn chỉnh (không thấy mặt hoặc crop hỏng)",
         "Nhận đúng nhưng bị loại thành Unknown (dưới ngưỡng)"],
    )
    _style_axes(ax, "x")
    ax.set_xlabel(f"Số ca lỗi trên tập test ({int(df['n_test'].iloc[0])} mẫu)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.085), ncol=1, frameon=False, fontsize=9)
    fig.text(0.012, 0.012, "Decision Tree, Gaussian NB và LDA gần như không loại được mẫu nào (xác suất đầu ra bão hòa gần 1),\n"
             "nên đoạn màu thứ ba của chúng không so sánh được với các thuật toán còn lại.",
             fontsize=8.5, color=INK_2, va="bottom")
    _heading(fig, "Các loại ca nhận diện lỗi theo thuật toán", subtitle)
    fig.tight_layout(rect=[0, 0.22, 1, 0.9])
    fig.savefig(_out("failure_breakdown.png"), dpi=150)
    plt.close(fig)


def plot_alg_overlap(preds, subtitle):
    """Mỗi mẫu test bị sai bởi bao nhiêu thuật toán. Decision Tree sai hơn nửa số mẫu nên vẽ thêm bản không tính nó."""
    ok = preds.pivot(index="face_path", columns="algorithm", values="correct").astype(bool)
    n_algos = ok.shape[1]
    idx = range(n_algos + 1)
    all_k = (~ok).sum(axis=1).value_counts().reindex(idx, fill_value=0)
    series = [(f"Cả {n_algos} thuật toán", all_k, SERIES[0])]
    if "DecisionTree" in ok.columns:
        no_dt = (~ok.drop(columns="DecisionTree")).sum(axis=1).value_counts().reindex(idx, fill_value=0)
        series.append((f"Bỏ Decision Tree ({n_algos - 1} thuật toán)", no_dt, SERIES[1]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1, 1.25]})
    w = 0.38
    k = np.arange(n_algos + 1)
    for j, (name, counts, col) in enumerate(series):
        offs = (j - (len(series) - 1) / 2) * w
        for ax in (ax1, ax2):
            ax.bar(k + offs, counts.to_numpy(), width=w - 0.04, color=col, label=name, edgecolor=SURFACE, linewidth=1.5)
        for kk in (0, 1):    # trái: chỉ ghi số ở hai cột lớn nhất
            v = int(counts.iloc[kk])
            # cột trái thấp hơn cột phải liền kề: số đặt phía trên sẽ bị cột phải che, nên ghi sang bên trái
            if j == 0 and any(int(o.iloc[kk]) > v for _, o, _ in series[1:]):
                ax1.text(kk + offs - (w / 2 + 0.01), v, str(v), ha="right", va="top", fontsize=8.5, color=INK_2)
            else:
                ax1.text(kk + offs, v + max(all_k) * 0.012, str(v), ha="center", fontsize=8.5, color=INK_2)
        nz = [i for i in range(2, n_algos + 1) if counts.iloc[i] > 0]    # phải: ghi số ở đầu và cuối đuôi
        for i in sorted({nz[0], nz[-1]} if nz else []):
            ax2.text(i + offs, counts.iloc[i] + 0.8, str(int(counts.iloc[i])), ha="center", fontsize=8.5, color=INK_2)
    ax1.set_title("Toàn bộ", loc="left", fontsize=10.5)
    ax2.set_title("Phóng to đuôi: từ 2 thuật toán cùng sai trở lên", loc="left", fontsize=10.5)
    for ax, ticks in ((ax1, k), (ax2, k[2:])):
        ax.set_xticks(ticks)     # đặt tick trước, rồi mới đặt giới hạn (set_xticks có thể nới rộng trục)
        ax.set_xlabel("Số thuật toán dự đoán sai mẫu đó")
        _style_axes(ax, "y")
    ax2.set_xlim(1.45, n_algos + 0.6)
    tail_max = max(int(c.iloc[2:].max()) for _, c, _ in series)
    ax2.set_ylim(0, tail_max * 1.18)
    ax1.set_ylabel("Số mẫu test")
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False, fontsize=9)
    _heading(fig, "Một mẫu test bị sai bởi bao nhiêu thuật toán", subtitle)
    fig.tight_layout(rect=[0, 0.07, 1, 0.9])
    fig.savefig(_out("overlap_hist.png"), dpi=150)
    plt.close(fig)


def main_algorithms():
    for need in ("metrics.csv", "cv.csv", "timing.csv", "failure_summary.csv", "predictions.csv", "best_algorithms.json"):
        if not os.path.exists(_out(need)):
            raise SystemExit(f"Thiếu {_out(need)}. Hãy chạy `python benchmark.py` (không kèm --skip-cv / --skip-timing) trước.")
    metrics, cv, timing = (pd.read_csv(_out(f)) for f in ("metrics.csv", "cv.csv", "timing.csv"))
    summary, preds = pd.read_csv(_out("failure_summary.csv")), pd.read_csv(_out("predictions.csv"))
    with open(_out("best_algorithms.json"), encoding="utf-8") as f:
        best = json.load(f)["best"]
    subtitle = ""
    if os.path.exists(_out("run_info.json")):
        with open(_out("run_info.json"), encoding="utf-8") as f:
            info = json.load(f)
        subtitle = (f"Embedding {info['embedding']}, {info['n_people']} người, {info['n_images']} ảnh "
                    f"(train {info['n_train']} / test {info['n_test']}). Hai thuật toán demo được nhấn màu xanh.")

    plot_alg_f1_ranking(metrics, cv, best, subtitle)
    plot_metrics_heatmap(metrics, label_col="algorithm")
    plot_alg_cost(timing, metrics, best, subtitle)
    plot_alg_speed_vs_f1(metrics, cv, timing, best, subtitle)
    plot_alg_failures(summary, metrics, subtitle)
    plot_alg_overlap(preds, subtitle)
    for name in best:
        plot_confusion(preds, name, label_col="algorithm")
    print("Đã vẽ biểu đồ vào", OUT_DIR)


def main_embeddings():
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
    print("Đã vẽ biểu đồ vào", OUT_DIR)


def main():
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kind", choices=["algorithms", "embeddings"], default="algorithms",
                        help="algorithms: so sánh 10 thuật toán (mặc định); embeddings: so sánh 9 embedding (phụ lục)")
    parser.add_argument("--dir", default=None, help="thư mục chứa CSV và nơi ghi biểu đồ (mặc định theo --kind)")
    args = parser.parse_args()
    OUT_DIR = args.dir or (ALGO_RESULTS_DIR if args.kind == "algorithms" else RESULTS_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    main_algorithms() if args.kind == "algorithms" else main_embeddings()


if __name__ == "__main__":
    main()
