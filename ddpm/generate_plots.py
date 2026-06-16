import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

SAVEFIG_KW = {"bbox_inches": "tight"}

SEEDS = [42, 142, 242]
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

ALL_CONFIGS = [
    {"name": "T100_ch32_lr0.0001", "T": 100, "ch": 32, "lr": 1e-4},
    {"name": "T200_ch32_lr0.0001", "T": 200, "ch": 32, "lr": 1e-4},
    {"name": "T400_ch32_lr0.0001", "T": 400, "ch": 32, "lr": 1e-4},
    {"name": "T200_ch16_lr0.0001", "T": 200, "ch": 16, "lr": 1e-4},
    {"name": "T200_ch64_lr0.0001", "T": 200, "ch": 64, "lr": 1e-4},
    {"name": "T200_ch64_lr1e-05",  "T": 200, "ch": 64, "lr": 1e-5},
    {"name": "T200_ch64_lr0.001",  "T": 200, "ch": 64, "lr": 1e-3},
]

PHASE1 = [c for c in ALL_CONFIGS if c["ch"] == 32 and c["lr"] == 1e-4]
PHASE2 = [c for c in ALL_CONFIGS if c["T"] == 200 and c["lr"] == 1e-4]
PHASE3 = [c for c in ALL_CONFIGS if c["T"] == 200 and c["ch"] == 64]
BEST   = "T200_ch64_lr0.0001"


def load_history(cfg_name, seed):
    with open(f"ddpm_{cfg_name}/seed_{seed}/history.json") as f:
        return json.load(f)


def load_all_histories():
    data = {}
    for cfg in ALL_CONFIGS:
        name = cfg["name"]
        data[name] = {}
        for s in SEEDS:
            try:
                data[name][s] = load_history(name, s)
            except FileNotFoundError:
                print(f"  [WARN] Missing: ddpm_{name}/seed_{s}/history.json")
        if not data[name]:
            del data[name]
    return data


def _fids(histories, name):
    return [histories[name][s]["fid"][-1]["fid"] for s in SEEDS if s in histories[name]]


# ── Plot 1: FID bar chart ─────────────────────────────────────────────────
def plot_fid_barchart(histories):
    names, means, stds = [], [], []
    for cfg in ALL_CONFIGS:
        n = cfg["name"]
        if n not in histories:
            continue
        fids = _fids(histories, n)
        names.append(n.replace("_", "\n"))
        means.append(np.mean(fids))
        stds.append(np.std(fids))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    best_idx = int(np.argmin(means))
    colors = ["#4c72b0"] * len(names)
    colors[best_idx] = "#2ca02c"  # green for best

    ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("FID ")
    ax.set_title("FID Scores Across All Configurations")
    ax.set_ylim(bottom=0)

    # label above error bar, with enough padding so it never clips
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + max(3, ax.get_ylim()[1] * 0.01),
                f"{m:.1f}", ha="center", fontsize=8, fontweight="bold")

    fig.savefig(f"{PLOT_DIR}/fid_all_configs.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved fid_all_configs.png")


# ── Plot 2: Hyperparameter sweep ─────────────────────────────────────────
def plot_hyperparam_sweep(histories):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # shared Y range across all three panels
    all_means, all_stds = [], []
    for phase in [PHASE1, PHASE2, PHASE3]:
        for cfg in phase:
            n = cfg["name"]
            if n not in histories:
                continue
            fids = _fids(histories, n)
            all_means.append(np.mean(fids))
            all_stds.append(np.std(fids))
    y_max = max(m + s for m, s in zip(all_means, all_stds)) * 1.15

    def sweep_panel(ax, cfgs, param_key, param_label, title):
        pairs = []
        for cfg in cfgs:
            n = cfg["name"]
            if n not in histories:
                continue
            fids = _fids(histories, n)
            pairs.append((cfg[param_key], np.mean(fids), np.std(fids)))
        pairs.sort(key=lambda t: t[0])
        vals      = [p[0] for p in pairs]
        fid_means = [p[1] for p in pairs]
        fid_stds  = [p[2] for p in pairs]

        ax.errorbar(vals, fid_means, yerr=fid_stds, marker="o", capsize=5,
                    linewidth=2, markersize=8, color="#4c72b0")
        best_i = int(np.argmin(fid_means))
        ax.plot(vals[best_i], fid_means[best_i], "o", color="#2ca02c",
                markersize=12, zorder=5, label=f"Best: {vals[best_i]}")
        ax.set_xlabel(param_label)
        ax.set_ylabel("FID")
        ax.set_title(title)
        ax.set_ylim(0, y_max)
        ax.legend()
        if param_key == "lr":
            ax.set_xscale("log")

    sweep_panel(axes[0], PHASE1, "T",  "Timesteps $T$",
                "Phase 1: Timesteps\n(ch=32, lr=1e-4)")
    sweep_panel(axes[1], PHASE2, "ch", "Base Channels",
                "Phase 2: Channels\n(T=200, lr=1e-4)")
    sweep_panel(axes[2], PHASE3, "lr", "Learning Rate",
                "Phase 3: Learning Rate\n(T=200, ch=64)")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/hyperparam_sweep.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved hyperparam_sweep.png")


# ── Plot 3: Loss curves – all configs, log scale ──────────────────────────
def plot_loss_curves_all(histories):
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.cm.tab10
    for i, cfg in enumerate(ALL_CONFIGS):
        n = cfg["name"]
        if n not in histories:
            continue
        all_losses = [histories[n][s]["loss"] for s in SEEDS if s in histories[n]]
        min_len = min(len(l) for l in all_losses)
        arr    = np.array([l[:min_len] for l in all_losses])
        mean   = arr.mean(axis=0)
        std    = arr.std(axis=0)
        epochs = np.arange(1, min_len + 1)
        lw     = 2.2 if n == BEST else 1.0
        alpha  = 1.0 if n == BEST else 0.65
        ax.plot(epochs, mean, label=n, color=cmap(i), linewidth=lw, alpha=alpha)
        ax.fill_between(epochs, np.maximum(mean - std, 1e-9),
                        mean + std, color=cmap(i), alpha=0.1)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training Loss Curves (mean ± std over 3 seeds)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(f"{PLOT_DIR}/loss_curves_all.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved loss_curves_all.png")


# ── Plot 4: Best config loss per seed – log scale ─────────────────────────
def plot_loss_best_config(histories):
    if BEST not in histories:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for s in SEEDS:
        if s not in histories[BEST]:
            continue
        loss = histories[BEST][s]["loss"]
        ax.plot(range(1, len(loss) + 1), loss, label=f"Seed {s}", linewidth=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(f"Training Loss - Best Config ({BEST})")
    ax.legend()
    fig.savefig(f"{PLOT_DIR}/loss_best_config.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved loss_best_config.png")


# ── Plot 5: Sample progression split into multiple figures ────────────────
def plot_sample_progression():
    base = f"ddpm_{BEST}/seed_42"
    if not os.path.isdir(base):
        print("  [SKIP] sample_progression: directory not found")
        return
    all_epochs = sorted([
        int(f.split("_")[-1].split(".")[0])
        for f in os.listdir(base)
        if f.startswith("samples_epoch_") and f.endswith(".png")
    ])
    if not all_epochs:
        print("  [SKIP] sample_progression: no files found")
        return

    # pick up to 12 evenly-spaced checkpoints, always include last
    n_pick = min(12, len(all_epochs))
    idx    = np.round(np.linspace(0, len(all_epochs) - 1, n_pick)).astype(int)
    pick   = [all_epochs[i] for i in sorted(set(idx))]

    cols_per_fig = 4
    chunks = [pick[i:i + cols_per_fig] for i in range(0, len(pick), cols_per_fig)]

    for fig_idx, chunk in enumerate(chunks, 1):
        fig, axes = plt.subplots(1, len(chunk), figsize=(4.5 * len(chunk), 4.5))
        if len(chunk) == 1:
            axes = [axes]
        for ax, ep in zip(axes, chunk):
            img = Image.open(f"{base}/samples_epoch_{ep:04d}.png")
            ax.imshow(img)
            ax.set_title(f"Epoch {ep}", fontsize=12)
            ax.axis("off")
        fig.suptitle(
            f"Generated Samples - {BEST}, seed=42  (part {fig_idx}/{len(chunks)})",
            fontsize=13, y=1.01
        )
        fig.tight_layout()
        fname = f"{PLOT_DIR}/sample_progression_{fig_idx:02d}.png"
        fig.savefig(fname, **SAVEFIG_KW)
        plt.close(fig)
        print(f"  Saved {os.path.basename(fname)}")


# ── Plot 6: Final samples comparison (phase-based) ──────────────────────
def plot_final_samples_comparison():
    phase_defs = [
        ("Timestep Sweep (ch=32, lr=1e-4)",       PHASE1),
        ("Channel Sweep (T=200, lr=1e-4)",         PHASE2),
        ("Learning Rate Sweep (T=200, ch=64)",     PHASE3),
    ]
    total = len(phase_defs)

    def _last_sample(cfg_name):
        d = f"ddpm_{cfg_name}/seed_42"
        if not os.path.isdir(d):
            return None
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("samples_epoch_") and f.endswith(".png")])
        return f"{d}/{files[-1]}" if files else None

    for fig_idx, (title, cfgs) in enumerate(phase_defs, 1):
        entries = []
        for cfg in cfgs:
            p = _last_sample(cfg["name"])
            if p:
                entries.append((cfg["name"], p))
        if not entries:
            continue

        n_img = len(entries)
        fig, axes = plt.subplots(1, n_img, figsize=(5.5 * n_img, 5.5))
        if n_img == 1:
            axes = [axes]
        for ax, (name, fpath) in zip(axes, entries):
            ax.imshow(_crop_whitespace(Image.open(fpath)))
            ax.set_title(name, fontsize=11)
            ax.axis("off")
        fig.suptitle(
            f"Final Samples - {title}  (part {fig_idx}/{total})",
            fontsize=13, y=0.98
        )
        fig.subplots_adjust(top=0.90, wspace=0.05)
        fname = f"{PLOT_DIR}/final_samples_comparison_{fig_idx:02d}.png"
        fig.savefig(fname, **SAVEFIG_KW)
        plt.close(fig)
        print(f"  Saved {os.path.basename(fname)}")


# ── Plot 7: Cats vs Cats+Dogs ─────────────────────────────────────────────
def plot_cats_vs_catsdogs(histories):
    cats_fids = [histories[BEST][s]["fid"][-1]["fid"]
                 for s in SEEDS if BEST in histories and s in histories[BEST]]

    dogs_fids = []
    for s in SEEDS:
        path = f"ddpm_cats_dogs/seed_{s}/history.json"
        if os.path.exists(path):
            with open(path) as f:
                dogs_fids.append(json.load(f)["fid"][-1]["fid"])

    if not dogs_fids:
        print("  [SKIP] cats_vs_catsdogs: no cats_dogs data")
        return

    labels = ["Cats Only", "Cats + Dogs"]
    means  = [np.mean(cats_fids), np.mean(dogs_fids)]
    stds   = [np.std(cats_fids),  np.std(dogs_fids)]
    colors = ["#4c72b0", "#dd8452"]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=6, color=colors, edgecolor="white", width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("FID")
    ax.set_title("FID: Cats Only vs. Cats + Dogs")
    ax.set_ylim(bottom=0)

    # labels placed just above error bar cap, with dynamic headroom
    y_top = max(m + s for m, s in zip(means, stds))
    pad   = y_top * 0.03
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + pad, f"{m:.1f}", ha="center", fontsize=11, fontweight="bold")

    # extra top margin so text doesn't clip
    ax.set_ylim(top=y_top * 1.15)

    fig.savefig(f"{PLOT_DIR}/cats_vs_catsdogs.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved cats_vs_catsdogs.png")


# ── Helper: crop uniform border from an image ────────────────────────────
def _crop_whitespace(img, tol=240):
    """Trim near-white/near-black border rows and columns from a PIL image."""
    import numpy as np
    arr = np.array(img.convert("RGB"))
    # mask of pixels that are NOT near-white
    mask = (arr < tol).any(axis=2)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img.crop((cmin, rmin, cmax + 1, rmax + 1))


# ── Helper: gallery split into N-image chunks ─────────────────────────────
def _save_gallery_chunks(files_dir, file_list, title, prefix, cols=2, rows_per_fig=1):
    per_fig = cols * rows_per_fig
    chunks  = [file_list[i:i + per_fig] for i in range(0, len(file_list), per_fig)]
    for fig_idx, chunk in enumerate(chunks, 1):
        n_img = len(chunk)
        r     = math.ceil(n_img / cols)
        # measure first image to compute aspect-aware height
        sample_img = _crop_whitespace(Image.open(f"{files_dir}/{chunk[0]}"))
        cell_w     = 6.5
        cell_h     = cell_w * (sample_img.height / sample_img.width)
        fig_h      = cell_h * r + 0.55  # 0.55 inch for title
        fig, axes  = plt.subplots(r, cols, figsize=(cell_w * cols, fig_h))
        axes_flat  = np.array(axes).flatten()
        for ax, fname in zip(axes_flat, chunk):
            img = Image.open(f"{files_dir}/{fname}")
            ax.imshow(_crop_whitespace(img))
            ax.set_title(fname.replace(".png", ""), fontsize=11, pad=6)
            ax.axis("off")
        for ax in axes_flat[n_img:]:
            ax.axis("off")
        fig.suptitle(f"{title}  (part {fig_idx}/{len(chunks)})", fontsize=13, y=1.0)
        fig.subplots_adjust(top=1.0 - 0.5/fig_h, hspace=0.06, wspace=0.04)
        out = f"{PLOT_DIR}/{prefix}_{fig_idx:02d}.png"
        fig.savefig(out, **SAVEFIG_KW)
        plt.close(fig)
        print(f"  Saved {os.path.basename(out)}")


# ── Plot 8: Interpolation gallery – 3 strips per figure ─────────────────
def plot_interpolation_gallery():
    d = "ddpm_baseline/interpolation"
    if not os.path.isdir(d):
        print("  [SKIP] interpolation directory not found")
        return
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if not files:
        return
    FIG_W        = 14.0          # fixed figure width in inches
    TITLE_H      = 0.35          # inches reserved for suptitle
    STRIP_LABEL  = 0.25          # inches reserved for ax.set_title per strip
    rows_per_fig = 3
    chunks       = [files[i:i + rows_per_fig] for i in range(0, len(files), rows_per_fig)]
    total        = len(chunks)
    for fig_idx, chunk in enumerate(chunks, 1):
        # measure cropped aspect ratio of each strip to compute exact row heights
        images  = [_crop_whitespace(Image.open(f"{d}/{f}")) for f in chunk]
        row_h_in = [FIG_W * (img.height / img.width) for img in images]
        total_img_h = sum(row_h_in)
        n_strips    = len(chunk)
        fig_h       = total_img_h + TITLE_H + STRIP_LABEL * n_strips + 0.1 * (n_strips - 1)

        fig, axes = plt.subplots(n_strips, 1, figsize=(FIG_W, fig_h))
        if n_strips == 1:
            axes = [axes]

        # set exact height ratios so matplotlib allocates space proportionally
        fig.subplots_adjust(
            left=0, right=1, bottom=0,
            top=(total_img_h + STRIP_LABEL * n_strips) / fig_h,
            hspace=0.08,
        )
        for ax, img, fname in zip(axes, images, chunk):
            ax.imshow(img)
            ax.set_title(fname.replace(".png", ""), fontsize=11, pad=6)
            ax.axis("off")

        fig.suptitle(
            f"Latent Space Interpolation (SLERP)  (part {fig_idx}/{total})",
            fontsize=13,
            y=1.0 - TITLE_H / (2 * fig_h),
        )
        out = f"{PLOT_DIR}/interpolation_gallery_{fig_idx:02d}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {os.path.basename(out)}")


# ── Plot 9: Inpainting gallery ────────────────────────────────────────────
def plot_inpainting_gallery():
    d = "ddpm_baseline/inpainting"
    if not os.path.isdir(d):
        print("  [SKIP] inpainting directory not found")
        return
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))[:9]
    if not files:
        return
    _save_gallery_chunks(d, files, "RePaint Inpainting Results",
                         "inpainting_gallery", cols=3, rows_per_fig=1)


# ── Plot 10: Style transfer gallery ──────────────────────────────────────
def plot_style_transfer_gallery():
    d = "ddpm_baseline/style_transfer"
    if not os.path.isdir(d):
        print("  [SKIP] style_transfer directory not found")
        return
    files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    if not files:
        return
    _save_gallery_chunks(d, files, "Neural Style Transfer Results (VGG-19)",
                         "style_transfer_gallery", cols=3, rows_per_fig=1)



# ── Plot 11: Loss smoothed ────────────────────────────────────────────────
def plot_loss_smoothed_best(histories):
    if BEST not in histories:
        return
    all_losses = [histories[BEST][s]["loss"] for s in SEEDS if s in histories[BEST]]
    min_len    = min(len(l) for l in all_losses)
    arr        = np.array([l[:min_len] for l in all_losses])
    mean       = arr.mean(axis=0)
    epochs     = np.arange(1, min_len + 1)

    window   = 10
    # "valid" avoids boundary artifacts; centre each output point at its window midpoint
    smoothed = np.convolve(mean, np.ones(window) / window, mode="valid")
    offset   = (window - 1) / 2          # align smoothed[i] with epoch epochs[i + offset]
    ep_sm    = epochs[window - 1:] - offset   # same length as smoothed, centred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, mean, alpha=0.55, color="#4c72b0", linewidth=1.2, label="Raw")
    ax1.plot(ep_sm, smoothed, color="#c44e52", linewidth=2.2, label=f"Smoothed (w={window})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Full Training (mean over seeds)")
    ax1.legend()

    cutoff  = min_len // 3
    sm_mask = ep_sm >= epochs[cutoff]
    ax2.plot(epochs[cutoff:], mean[cutoff:], alpha=0.55, color="#4c72b0", linewidth=1.2)
    ax2.plot(ep_sm[sm_mask], smoothed[sm_mask], color="#c44e52", linewidth=2.2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE Loss")
    ax2.set_title("Convergence Zoom (last 2/3 of training)")

    fig.suptitle(f"Loss Analysis - Best Config ({BEST})", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/loss_smoothed_best.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved loss_smoothed_best.png")


# ── Plot 12: FID summary table ────────────────────────────────────────────
def plot_fid_table(histories):
    rows = []
    for cfg in ALL_CONFIGS:
        n = cfg["name"]
        if n not in histories:
            continue
        fids = _fids(histories, n)
        row  = [n, f"{cfg['T']}", f"{cfg['ch']}", f"{cfg['lr']:.0e}"]
        row += [f"{f:.1f}" for f in fids]
        row += [f"{np.mean(fids):.1f}", f"{np.std(fids):.1f}"]
        rows.append(row)

    dogs_fids = []
    for s in SEEDS:
        path = f"ddpm_cats_dogs/seed_{s}/history.json"
        if os.path.exists(path):
            with open(path) as f:
                dogs_fids.append(json.load(f)["fid"][-1]["fid"])
    if dogs_fids:
        row = ["cats_dogs", "200", "64", "1e-04"]
        row += [f"{f:.1f}" for f in dogs_fids]
        while len(row) < 4 + len(SEEDS):
            row.append("-")
        row += [f"{np.mean(dogs_fids):.1f}", f"{np.std(dogs_fids):.1f}"]
        rows.append(row)

    cols       = ["Config", "$T$", "ch", "lr"] + [f"Seed {s}" for s in SEEDS] + ["Mean", "Std"]
    n_cols     = len(cols)
    col_widths = [0.30] + [0.07] * (n_cols - 1)   # Config wide, rest narrow
    total_w    = sum(col_widths)

    fig, ax = plt.subplots(figsize=(16, 0.6 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # manual column widths
    for (r, c), cell in table.get_celld().items():
        cell.set_width(col_widths[c] / total_w * 0.9)
        if r == 0:
            cell.set_facecolor("#4c72b0")
            cell.set_text_props(color="white", fontweight="bold")
        elif rows[r - 1][0] == BEST:
            cell.set_facecolor("#e8f5e9")   # light green for best
        else:
            cell.set_facecolor("#f7f7f7" if r % 2 == 0 else "white")

    # Place title directly above the table with minimal gap
    fig.text(0.5, 0.98, "FID Results Summary", fontsize=14,
             ha="center", va="top", transform=fig.transFigure)
    fig.savefig(f"{PLOT_DIR}/fid_table.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved fid_table.png")


# ── Plot 13: Hero samples grid (best config, final epoch) ────────────────
def plot_best_samples_grid():
    base = f"ddpm_{BEST}/seed_42"
    if not os.path.isdir(base):
        print("  [SKIP] best_samples_grid: directory not found")
        return
    samples = sorted([f for f in os.listdir(base)
                      if f.startswith("samples_epoch_") and f.endswith(".png")])
    if not samples:
        print("  [SKIP] best_samples_grid: no sample files")
        return
    img = _crop_whitespace(Image.open(f"{base}/{samples[-1]}"))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Best Configuration: {BEST}  (final epoch, seed=42)",
                 fontsize=12, pad=6)
    fig.savefig(f"{PLOT_DIR}/best_samples_grid.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved best_samples_grid.png")


# ── Plot 14: Cats-only vs Cats+Dogs side-by-side samples ─────────────────
def plot_catsdogs_samples_comparison():
    cats_base = f"ddpm_{BEST}/seed_42"
    dogs_base = "ddpm_cats_dogs/seed_42"
    if not os.path.isdir(cats_base) or not os.path.isdir(dogs_base):
        print("  [SKIP] catsdogs_samples: directories not found")
        return

    def _last_sample(d):
        files = sorted([f for f in os.listdir(d)
                        if f.startswith("samples_epoch_") and f.endswith(".png")])
        return f"{d}/{files[-1]}" if files else None

    cats_path = _last_sample(cats_base)
    dogs_path = _last_sample(dogs_base)
    if not cats_path or not dogs_path:
        print("  [SKIP] catsdogs_samples: no sample files")
        return

    cats_img = _crop_whitespace(Image.open(cats_path))
    dogs_img = _crop_whitespace(Image.open(dogs_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    ax1.imshow(cats_img)
    ax1.set_title("Cats Only", fontsize=13, pad=6)
    ax1.axis("off")
    ax2.imshow(dogs_img)
    ax2.set_title("Cats + Dogs", fontsize=13, pad=6)
    ax2.axis("off")
    fig.suptitle(f"Sample Comparison - {BEST} (seed=42, final epoch)", fontsize=14, y=1.01)
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(f"{PLOT_DIR}/catsdogs_samples_comparison.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved catsdogs_samples_comparison.png")


def plot_cross_architecture_comparison():
    models = ["Tiny DDPM", "DCGAN", "β-VAE"]
 
    cats_means = [81.64, 101.03, 252.50]
    cats_stds  = [8.50,    5.25,    0] 
 
    dogs_means = [65.70,  85.03, 235.99]
    dogs_stds  = [1.90,    1.49,    0]
 
    x     = np.arange(len(models))
    width = 0.35
    colors_cats = "#4c72b0"
    colors_dogs = "#dd8452"
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    bars_cats = ax.bar(x - width / 2, cats_means, width,
                       yerr=cats_stds, capsize=5,
                       color=colors_cats, edgecolor="white", label="Cats Only")
    bars_dogs = ax.bar(x + width / 2, dogs_means, width,
                       yerr=dogs_stds, capsize=5,
                       color=colors_dogs, edgecolor="white", label="Cats + Dogs")
 
    # value labels above each bar
    for bars, means, stds in [(bars_cats, cats_means, cats_stds),
                               (bars_dogs, dogs_means, dogs_stds)]:
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + max(2, ax.get_ylim()[1] * 0.005),
                    f"{m:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
 
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("FID")
    ax.set_title("Cross-Architecture FID Comparison")
    ax.set_ylim(bottom=0)
    ax.legend()

    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/cross_architecture_comparison.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved cross_architecture_comparison.png")
def plot_best_samples_grid():
    base = f"ddpm_{BEST}/seed_42"
    if not os.path.isdir(base):
        print("  [SKIP] best_samples_grid: directory not found")
        return
    samples = sorted([f for f in os.listdir(base)
                      if f.startswith("samples_epoch_") and f.endswith(".png")])
    if not samples:
        print("  [SKIP] best_samples_grid: no sample files")
        return
    img = _crop_whitespace(Image.open(f"{base}/{samples[-1]}"))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"Best Configuration: {BEST}  (final epoch, seed=42)",
                 fontsize=12, pad=6)
    fig.savefig(f"{PLOT_DIR}/best_samples_grid.png", **SAVEFIG_KW)
    plt.close(fig)
    print("  Saved best_samples_grid.png")
 
 




# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading histories...")
    histories = load_all_histories()
    print(f"Loaded {len(histories)} configs\n")

    print("Generating plots...")
    plot_fid_barchart(histories)
    plot_hyperparam_sweep(histories)
    plot_loss_curves_all(histories)
    plot_loss_best_config(histories)
    plot_loss_smoothed_best(histories)
    plot_fid_table(histories)
    plot_sample_progression()
    plot_final_samples_comparison()
    plot_cats_vs_catsdogs(histories)
    plot_interpolation_gallery()
    plot_inpainting_gallery()
    plot_style_transfer_gallery()
    plot_best_samples_grid()
    plot_catsdogs_samples_comparison()
    plot_cross_architecture_comparison()


    print(f"\nDone - all plots saved to '{PLOT_DIR}/'")