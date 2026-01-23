from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm


# ============================================================
# Config defaults (can be overridden from run script)
# ============================================================
DEFAULTS: Dict[str, Any] = {
    "LABELS_NAME": "clusters.labels.csv",
    "OUT_DIRNAME": "qc_dashboard_png",

    # If set (Path or str), all PNGs go there
    # If None -> per session / OUT_DIRNAME
    "OUT_ROOT": None,

    # canvas
    "OUT_W_PX": 1400,
    "OUT_H_PX": 800,
    "OUT_DPI": 150,

    # density (good clusters median depths)
    "DEPTH_BINS": 80,
    "DEPTH_SMOOTH_SIGMA": 2.2,

    # heatmap
    "HEAT_TBINS": 240,
    "HEAT_DBINS": 260,
    "HEAT_MAX_POINTS": 900_000,
    "HEAT_CLIP_PERCENTILE": 99.6,
    "HEAT_TIME_UNIT": "min",  # "s" or "min"

    # fixed depth axis (requested)
    "HEAT_DEPTH_MIN": 0.0,
    "HEAT_DEPTH_MAX": 4000.0,

    # optional: fixed time axis max (None = auto from data)
    "HEAT_TIME_MAX": None,  # e.g. 60.0 if HEAT_TIME_UNIT="min"

    # IMPORTANT: avoid white holes in LogNorm
    # If True: add +1 pseudo-count to all bins so zeros don't get masked by LogNorm
    "HEAT_ADD_PSEUDOCOUNT": True,

    # style
    "BLUE": "#2A6FDB",
    "TITLE_SIZE": 11,
    "FONT_SIZE": 10,
}


# ============================================================
# Style
# ============================================================
def apply_style(cfg: Dict[str, Any]) -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#E6E6E6",
        "axes.labelcolor": "#333333",
        "text.color": "#222222",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "font.size": cfg["FONT_SIZE"],
        "axes.titleweight": "normal",
        "axes.titlesize": cfg["TITLE_SIZE"],
        "axes.labelsize": cfg["FONT_SIZE"],
        "savefig.facecolor": "white",
    })


# ============================================================
# IO
# ============================================================
def load_npy(path: Path) -> Optional[np.ndarray]:
    try:
        return np.load(path, mmap_mode="r")
    except Exception:
        return None


def read_labels(labels_csv: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(labels_csv)
    except Exception:
        return None

    if "label" not in df.columns or "cluster_id" not in df.columns:
        return None

    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce")
    df = df.dropna(subset=["cluster_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    return df


def find_probes_for_session(root: Path, session: str, labels_name: str) -> List[Path]:
    session_folder = root / session
    if not session_folder.exists():
        return []
    csvs = list(session_folder.rglob(labels_name))
    return sorted({c.parent for c in csvs})


def load_probe_data(alf_probe: Path, labels_name: str):
    labels = read_labels(alf_probe / labels_name)
    if labels is None:
        return None

    clusters = load_npy(alf_probe / "spikes.clusters.npy")
    depths   = load_npy(alf_probe / "spikes.depths.npy")
    times    = load_npy(alf_probe / "spikes.times.npy")

    if clusters is None or depths is None or times is None:
        return None

    clusters = np.asarray(clusters).astype(int, copy=False)
    depths = np.asarray(depths).astype(float, copy=False)
    times = np.asarray(times).astype(float, copy=False)

    if not (clusters.shape[0] == depths.shape[0] == times.shape[0]):
        return None

    return labels, clusters, depths, times


# ============================================================
# Helpers
# ============================================================
def gaussian_smooth(y: np.ndarray, sigma: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 5:
        return y
    sigma = float(max(sigma, 0.0))
    if sigma == 0.0:
        return y
    half = int(np.ceil(3 * sigma))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    ypad = np.pad(y, (half, half), mode="reflect")
    return np.convolve(ypad, k, mode="valid")


def style_axes(ax):
    ax.grid(alpha=0.10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def good_cluster_ids(labels: pd.DataFrame) -> np.ndarray:
    return labels.loc[labels["label"] == "good", "cluster_id"].to_numpy(dtype=int)


def compute_good_unit_depths(labels: pd.DataFrame, clusters: np.ndarray, depths: np.ndarray) -> np.ndarray:
    gids = good_cluster_ids(labels)
    out = []
    for cid in gids:
        d = depths[clusters == cid]
        d = d[np.isfinite(d)]
        if d.size:
            out.append(float(np.median(d)))
    return np.asarray(out, dtype=float)


def get_good_spikes_time_depth(times: np.ndarray, clusters: np.ndarray, depths: np.ndarray, labels: pd.DataFrame):
    gids = good_cluster_ids(labels)
    if gids.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    m = np.isin(clusters, gids)
    t = times[m]
    d = depths[m]
    ok = np.isfinite(t) & np.isfinite(d)
    return t[ok].astype(float), d[ok].astype(float)


# ============================================================
# Pro dashboard: top banner + (density | heatmap)
# ============================================================
def make_dashboard_png(
    session: str,
    alf_probe: Path,
    out_png: Path,
    cfg: Dict[str, Any],
) -> bool:
    loaded = load_probe_data(alf_probe, cfg["LABELS_NAME"])
    if loaded is None:
        return False

    labels, clusters, depths, times = loaded
    blue = cfg["BLUE"]

    # ---- summary
    n_good = int((labels["label"] == "good").sum())
    duration_s = float(np.nanmax(times)) if times.size else np.nan
    duration_min = duration_s / 60.0 if np.isfinite(duration_s) else np.nan

    # ---- plot data
    good_depths = compute_good_unit_depths(labels, clusters, depths)
    ht, hd = get_good_spikes_time_depth(times, clusters, depths, labels)

    # ---- fixed depth range (requested)
    dmin = float(cfg.get("HEAT_DEPTH_MIN", 0.0))
    dmax = float(cfg.get("HEAT_DEPTH_MAX", 4000.0))
    if not np.isfinite(dmin):
        dmin = 0.0
    if not np.isfinite(dmax) or dmax <= dmin:
        dmax = dmin + 4000.0

    # ---- figure
    fig_w_in = cfg["OUT_W_PX"] / cfg["OUT_DPI"]
    fig_h_in = cfg["OUT_H_PX"] / cfg["OUT_DPI"]
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=cfg["OUT_DPI"], constrained_layout=True)

    # 2 rows: banner + main plot
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[0.18, 1.0],
        width_ratios=[0.22, 1.0],
        hspace=0.02,
        wspace=0.06,
    )

    ax_banner = fig.add_subplot(gs[0, :])
    ax_den    = fig.add_subplot(gs[1, 0])
    ax_hm     = fig.add_subplot(gs[1, 1], sharey=ax_den)

    # -------------------------
    # Top banner (clean, not overlay)
    # -------------------------
    ax_banner.set_xlim(0, 1)
    ax_banner.set_ylim(0, 1)
    ax_banner.axis("off")

    # background band
    ax_banner.add_patch(
        mpl.patches.FancyBboxPatch(
            (0.0, 0.0), 1.0, 1.0,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#F7F7F7",
            edgecolor="#E6E6E6",
            linewidth=0.9,
            transform=ax_banner.transAxes,
            clip_on=False,
        )
    )

    dur_str = f"{duration_min:.1f} min" if np.isfinite(duration_min) else ("NA" if not np.isfinite(duration_s) else f"{duration_s:.1f} s")
    left = f"mouse_ID: {session}   •   probe: {alf_probe.name}"
    right = f"good units: {n_good}   •   duration: {dur_str}"

    ax_banner.text(0.02, 0.62, left, ha="left", va="center", fontsize=cfg["FONT_SIZE"] + 1)
    ax_banner.text(0.02, 0.28, right, ha="left", va="center", fontsize=cfg["FONT_SIZE"])
    # subtle divider line
    ax_banner.plot([0.0, 1.0], [0.02, 0.02], color="#E6E6E6", lw=1.0, transform=ax_banner.transAxes)

    # -------------------------
    # Left: good unit density (median depth per good cluster)
    # -------------------------
    ax_den.set_title("Good unit density", pad=6)

    if good_depths.size:
        gd = good_depths[np.isfinite(good_depths)]
        gd = gd[(gd >= dmin) & (gd <= dmax)]
        if gd.size:
            counts, edges = np.histogram(gd, bins=int(cfg["DEPTH_BINS"]), range=(dmin, dmax))
            centers = 0.5 * (edges[:-1] + edges[1:])
            dens = counts.astype(float)
            if dens.max() > 0:
                dens /= dens.max()
            dens = gaussian_smooth(dens, sigma=float(cfg["DEPTH_SMOOTH_SIGMA"]))
            if dens.max() > 0:
                dens /= dens.max()

            ax_den.fill_betweenx(centers, 0, dens, color=blue, alpha=0.18, linewidth=0)
            ax_den.plot(dens, centers, color=blue, lw=2.2)

    ax_den.set_ylim(dmax, dmin)   # depth increases downward
    ax_den.set_xlim(0, 1.02)
    ax_den.set_xlabel("Norm.")
    ax_den.set_ylabel("Depth (µm)")
    style_axes(ax_den)
    ax_den.spines["right"].set_visible(False)

    # hide y ticks on heatmap side
    ax_hm.tick_params(axis="y", which="both", left=False, labelleft=False)

    # -------------------------
    # Right: heatmap (good spikes time × depth) — no white holes
    # -------------------------
    ax_hm.set_title("Good spikes (time × depth)", pad=6)

    if ht.size:
        # downsample for speed
        if ht.size > int(cfg["HEAT_MAX_POINTS"]):
            rng = np.random.default_rng(0)
            idx = rng.choice(ht.size, size=int(cfg["HEAT_MAX_POINTS"]), replace=False)
            ht2 = ht[idx].astype(float)
            hd2 = hd[idx].astype(float)
        else:
            ht2 = ht.astype(float)
            hd2 = hd.astype(float)

        ok = np.isfinite(ht2) & np.isfinite(hd2)
        ht2, hd2 = ht2[ok], hd2[ok]

        # clip to fixed depth range
        mdepth = (hd2 >= dmin) & (hd2 <= dmax)
        ht2, hd2 = ht2[mdepth], hd2[mdepth]

        if ht2.size:
            # time unit conversion
            if str(cfg.get("HEAT_TIME_UNIT", "min")).lower() == "min":
                ht2 = ht2 / 60.0
                xlabel = "Time (min)"
            else:
                xlabel = "Time (s)"

            # time max
            tmax_data = float(np.max(ht2))
            tmax_cfg = cfg.get("HEAT_TIME_MAX", None)
            tmax = float(tmax_cfg) if (tmax_cfg is not None and np.isfinite(float(tmax_cfg))) else tmax_data
            tmax = max(tmax, 1e-6)

            t_edges = np.linspace(0.0, tmax, int(cfg["HEAT_TBINS"]) + 1)
            d_edges = np.linspace(dmin, dmax, int(cfg["HEAT_DBINS"]) + 1)

            H, _, _ = np.histogram2d(ht2, hd2, bins=[t_edges, d_edges])
            Z = H.T  # depth x time

            # KEY FIX: LogNorm masks zeros (=> "holes"). Add +1 to avoid any masked bins.
            if bool(cfg.get("HEAT_ADD_PSEUDOCOUNT", True)):
                Zp = Z + 1.0
                pos = Zp[np.isfinite(Zp)]
                vmax = float(np.percentile(pos, float(cfg["HEAT_CLIP_PERCENTILE"])))
                vmax = max(vmax, 2.0)
                norm = LogNorm(vmin=1.0, vmax=vmax)
                cbar_label = "Count + 1 (log)"
                Zshow = Zp
            else:
                pos = Z[Z > 0]
                if pos.size == 0:
                    ax_hm.axis("off")
                    Zshow = None
                    norm = None
                    cbar_label = ""
                else:
                    vmax = float(np.percentile(pos, float(cfg["HEAT_CLIP_PERCENTILE"])))
                    vmax = max(vmax, 1.0)
                    norm = LogNorm(vmin=1.0, vmax=vmax)
                    cbar_label = "Count (log)"
                    Zshow = Z

            if Zshow is not None:
                cmap = mpl.cm.magma.copy()
                # keep a clean background: minimum value uses the colormap minimum (no white)
                # (no set_bad needed because we no longer have masked zeros)

                im = ax_hm.imshow(
                    Zshow,
                    aspect="auto",
                    origin="lower",
                    extent=[t_edges[0], t_edges[-1], d_edges[0], d_edges[-1]],
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",  # crisp, pro (avoid smearing)
                )

                ax_hm.set_ylim(dmax, dmin)
                ax_hm.set_xlabel(xlabel)
                ax_hm.set_ylabel("")
                ax_hm.spines["top"].set_visible(False)
                ax_hm.spines["right"].set_visible(False)

                cbar = fig.colorbar(im, ax=ax_hm, fraction=0.030, pad=0.015)
                cbar.outline.set_visible(False)
                cbar.set_label(cbar_label, rotation=90)

        else:
            ax_hm.axis("off")
    else:
        ax_hm.axis("off")

    # Save
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=cfg["OUT_DPI"])
    plt.close(fig)
    return True


# ============================================================
# Run helpers
# ============================================================
def list_sessions_with_labels(root: Path, labels_name: str) -> List[str]:
    sessions = set()
    for csv in root.rglob(labels_name):
        try:
            rel = csv.relative_to(root)
            sessions.add(rel.parts[0])
        except Exception:
            pass
    return sorted(sessions)


def run_dashboards(
    data_root: Path,
    run_mode: List[str] | str,
    cfg: Dict[str, Any],
) -> None:
    apply_style(cfg)

    if isinstance(run_mode, str) and run_mode.upper() == "ALL":
        sessions = list_sessions_with_labels(data_root, cfg["LABELS_NAME"])
        print(f"[INFO] Mode ALL: found {len(sessions)} session(s) with labels.")
    else:
        sessions = list(run_mode)

    out_root = cfg.get("OUT_ROOT", None)
    out_root_path = Path(out_root) if out_root else None
    if out_root_path:
        out_root_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] OUT_ROOT = {out_root_path}")

    for session in sessions:
        probes = find_probes_for_session(data_root, session, cfg["LABELS_NAME"])
        if not probes:
            print(f"[WARN] No probes found for session: {session}")
            continue

        if out_root_path:
            out_dir = out_root_path
        else:
            out_dir = data_root / session / cfg["OUT_DIRNAME"]
            out_dir.mkdir(parents=True, exist_ok=True)

        for alf_probe in probes:
            out_png = out_dir / f"{session}_{alf_probe.name}_dashboard.png"
            ok = make_dashboard_png(session, alf_probe, out_png, cfg)
            if ok:
                print(f"[OK] {out_png}")
            else:
                print(f"[SKIP] {session} / {alf_probe.name}")


if __name__ == "__main__":
    print("Import this module and call run_dashboards(data_root, run_mode, cfg).")
