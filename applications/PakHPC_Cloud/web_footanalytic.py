#!/usr/bin/env python3
"""
streamlit_foot_analytics_advanced.py

Advanced Streamlit app for 32×32 plantar-pressure matrices (single scan + batch).

Key features:
- Robust 32×32 CSV loader (comma/tab/whitespace; strips non-numeric headers)
- Interactive 2D heatmap + optional 3D surface (Plotly)
- Foot analytics:
  * Total load (kg-equiv)
  * Left vs Right load share (connected components)
  * COP (center of pressure)
  * Contact area (in²) using cell pitch 0.5 in => area 0.25 in²
  * Heel/Mid/Meta/Toe distribution using user-editable row counts (relative vertical rows)
  * Healthy range checks:
      Heel 45–55%, Mid 10–15%, Meta 17–27%, Toe 8–13%
  * Active-row span and estimated foot length:
      Active rows = rows where row_sum >= 1% of total (configurable)
      Foot length (in) = active_rows_count * 0.5
- Batch mode: upload a ZIP of many CSVs → per-file summary + download CSV.

Run:
  streamlit run streamlit_foot_analytics_advanced.py
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objs as go


# -----------------------------
# Config
# -----------------------------
N = 32

CELL_PITCH_IN = 0.5
CELL_AREA_IN2 = CELL_PITCH_IN * CELL_PITCH_IN  # 0.25 in²

HEALTHY_RANGES = {
    "Heel": (45.0, 55.0),
    "Mid":  (10.0, 15.0),
    "Meta": (17.0, 27.0),
    "Toe":  (8.0,  13.0),
}

DEFAULT_ROW_COUNTS = (4, 3, 5, 4)  # Heel, Mid, Meta, Toe
DEFAULT_ACTIVE_RATIO = 0.01        # 1% rule


# -----------------------------
# Helpers: loading
# -----------------------------
_number_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def _sniff_delimiter(head: str) -> Optional[str]:
    """Return ',' or '\\t' if likely, else None (whitespace)."""
    c_comma = head.count(",")
    c_tab = head.count("\t")
    if max(c_comma, c_tab) < 3:
        return None
    return "\t" if c_tab > c_comma else ","


def load_matrix_32x32_from_bytes(b: bytes) -> np.ndarray:
    """
    Load a 32×32 numeric matrix from bytes. Handles:
    - comma/tab separated
    - whitespace separated
    - files with header lines (drops non-numeric rows)
    """
    head = b[:4096].decode("utf-8", errors="ignore")
    delim = _sniff_delimiter(head)

    # Try pandas first (more tolerant)
    try:
        df = pd.read_csv(io.BytesIO(b), header=None, sep=delim, engine="python")
        # keep only numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(how="all", axis=0).dropna(how="all", axis=1)
        arr = df.values
    except Exception:
        # fallback numpy
        try:
            arr = np.loadtxt(io.BytesIO(b), delimiter=delim if delim is not None else None)
        except Exception:
            arr = np.loadtxt(io.BytesIO(b))

    # If not exactly 32x32, attempt to extract first 1024 numbers and reshape
    if arr.shape != (N, N):
        toks = _number_re.findall(b.decode("utf-8", errors="ignore"))
        if len(toks) >= N * N:
            arr = np.array([float(t) for t in toks[: N * N]], dtype=float).reshape((N, N))

    if arr.shape != (N, N):
        raise ValueError(f"Expected 32×32 matrix; got {arr.shape}")
    return arr.astype(float)


# -----------------------------
# Helpers: connected components
# -----------------------------
def connected_components(mask: np.ndarray) -> List[np.ndarray]:
    H, W = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[np.ndarray] = []

    for r in range(H):
        for c in range(W):
            if mask[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                pts = []
                while stack:
                    rr, cc = stack.pop()
                    pts.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                comps.append(np.array(pts, dtype=int))

    comps.sort(key=lambda a: a.shape[0], reverse=True)
    return comps


def split_two_feet(mask: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    comps = connected_components(mask)
    if len(comps) == 0:
        return None, None
    if len(comps) == 1:
        return comps[0], None
    return comps[0], comps[1]


# -----------------------------
# Analytics helpers
# -----------------------------
def fracs_from_row_counts(heel_rows: int, mid_rows: int, meta_rows: int, toe_rows: int):
    vals = [max(0, int(heel_rows)), max(0, int(mid_rows)), max(0, int(meta_rows)), max(0, int(toe_rows))]
    s = sum(vals)
    if s <= 0:
        d = list(DEFAULT_ROW_COUNTS)
        s2 = sum(d)
        return tuple(v / s2 for v in d), d, "Invalid row counts; using defaults."
    fracs = tuple(v / s for v in vals)
    return fracs, vals, ""


def region_rows_from_bbox(minr: int, maxr: int, fracs: Tuple[float, float, float, float]) -> Dict[str, Tuple[int, int]]:
    length = maxr - minr + 1
    raw = np.array(fracs, dtype=float) * length
    sizes = np.floor(raw).astype(int)

    rem = length - int(sizes.sum())
    frac_part = raw - sizes
    if rem > 0:
        for idx in np.argsort(-frac_part)[:rem]:
            sizes[idx] += 1

    names = ["Heel", "Mid", "Meta", "Toe"]
    bounds: Dict[str, Tuple[int, int]] = {}
    start = minr
    for name, sz in zip(names, sizes):
        end = start + sz - 1
        bounds[name] = (start, end)
        start = end + 1
    bounds["Toe"] = (bounds["Toe"][0], maxr)
    return bounds


def compute_regions(mat: np.ndarray, threshold: float, fracs: Tuple[float, float, float, float]) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """Returns (region_kg, region_pct, feet_detected)."""
    w = mat.copy()
    w[w < threshold] = 0.0

    active = w > 0
    foot1, foot2 = split_two_feet(active)
    reg = {"Heel": 0.0, "Mid": 0.0, "Meta": 0.0, "Toe": 0.0}
    feet = 0

    for comp in (foot1, foot2):
        if comp is None:
            continue
        feet += 1
        rows = comp[:, 0]
        minr, maxr = int(rows.min()), int(rows.max())
        bounds = region_rows_from_bbox(minr, maxr, fracs)
        for name, (r0, r1) in bounds.items():
            sel = comp[(rows >= r0) & (rows <= r1)]
            if sel.size == 0:
                continue
            reg[name] += float(w[sel[:, 0], sel[:, 1]].sum())

    total = float(sum(reg.values()))
    pct = {k: (100.0 * reg[k] / total if total > 0 else 0.0) for k in reg}
    return reg, pct, feet


def compute_cop(mat: np.ndarray, threshold: float) -> Tuple[Optional[float], Optional[float]]:
    w = mat.copy()
    w[w < threshold] = 0.0
    s = float(w.sum())
    if s <= 0:
        return None, None
    rr, cc = np.indices(w.shape)
    return float((rr * w).sum() / s), float((cc * w).sum() / s)


def left_right_load(mat: np.ndarray, threshold: float) -> Tuple[float, float]:
    w = mat.copy()
    w[w < threshold] = 0.0
    active = w > 0
    foot1, foot2 = split_two_feet(active)

    def load_cent(comp):
        if comp is None:
            return 0.0, None
        vals = w[comp[:, 0], comp[:, 1]]
        return float(vals.sum()), float(comp[:, 1].mean())

    l1, c1 = load_cent(foot1)
    l2, c2 = load_cent(foot2)

    if foot2 is None or c1 is None:
        return l1, 0.0
    if c1 <= c2:
        return l1, l2
    return l2, l1


def contact_area_in2(mat: np.ndarray, threshold: float) -> float:
    w = mat.copy()
    w[w < threshold] = 0.0
    return int(np.count_nonzero(w > 0)) * CELL_AREA_IN2


def compute_active_row_span(mat: np.ndarray, threshold: float, ratio: float = DEFAULT_ACTIVE_RATIO) -> Tuple[Optional[int], Optional[int], int]:
    """
    Active vertical row span:
      active rows = rows where row_sum >= ratio * total
    """
    w = mat.copy()
    w[w < threshold] = 0.0
    total = float(w.sum())
    if total <= 0:
        return None, None, 0
    row_sums = w.sum(axis=1)
    row_thr = ratio * total
    active = np.where(row_sums >= row_thr)[0]
    if active.size == 0:
        return None, None, 0
    first = int(active.min())
    last = int(active.max())
    return first, last, int(last - first + 1)


def healthy_flags(pct: Dict[str, float]) -> List[str]:
    flags = []
    for name, (lo, hi) in HEALTHY_RANGES.items():
        v = pct.get(name, 0.0)
        if not (lo <= v <= hi):
            flags.append(f"{name}_OUT")
    return flags


# -----------------------------
# Plot helpers
# -----------------------------
def plot_heatmap(mat: np.ndarray, vmin: float, vmax: float, title: str) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=mat,
            colorscale="Viridis",
            zmin=vmin,
            zmax=vmax,
            colorbar=dict(title="kg / cell"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row",
        height=520,
        margin=dict(l=40, r=20, t=60, b=40),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def plot_surface(mat: np.ndarray, title: str) -> go.Figure:
    x, y = np.meshgrid(np.arange(mat.shape[1]), np.arange(mat.shape[0]))
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=mat, colorscale="Inferno")])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Column",
            yaxis_title="Row",
            zaxis_title="kg / cell",
        ),
        height=520,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


def plot_region_bars(pct: Dict[str, float]) -> go.Figure:
    names = ["Heel", "Mid", "Meta", "Toe"]
    vals = [pct.get(k, 0.0) for k in names]
    lo = [HEALTHY_RANGES[k][0] for k in names]
    hi = [HEALTHY_RANGES[k][1] for k in names]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=vals, name="Measured %"))
    fig.add_trace(go.Scatter(x=names, y=lo, mode="lines+markers", name="Lower bound"))
    fig.add_trace(go.Scatter(x=names, y=hi, mode="lines+markers", name="Upper bound"))
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=40, b=40), yaxis_title="% of total")
    return fig


# -----------------------------
# Batch processing
# -----------------------------
def summarize_matrix(mat: np.ndarray, name: str, threshold: float,
                     heel_rows: int, mid_rows: int, meta_rows: int, toe_rows: int,
                     active_ratio: float) -> Dict[str, object]:
    fracs, counts, warn = fracs_from_row_counts(heel_rows, mid_rows, meta_rows, toe_rows)
    reg_kg, reg_pct, feet = compute_regions(mat, threshold=threshold, fracs=fracs)
    flags = healthy_flags(reg_pct)

    total = float(np.sum(np.where(mat >= threshold, mat, 0.0)))
    left, right = left_right_load(mat, threshold=threshold)
    cop_r, cop_c = compute_cop(mat, threshold=threshold)
    area = contact_area_in2(mat, threshold=threshold)
    avg = (total / area) if area > 0 else 0.0

    first_r, last_r, span_n = compute_active_row_span(mat, threshold=threshold, ratio=active_ratio)
    length_in = span_n * CELL_PITCH_IN

    return {
        "file": name,
        "total_kg": total,
        "feet_detected": feet,
        "left_kg": left,
        "right_kg": right,
        "lr_ratio": (left / right) if right > 0 else np.nan,
        "cop_r": cop_r if cop_r is not None else np.nan,
        "cop_c": cop_c if cop_c is not None else np.nan,
        "contact_area_in2": area,
        "avg_kg_per_in2": avg,
        "heel_pct": reg_pct["Heel"],
        "mid_pct": reg_pct["Mid"],
        "meta_pct": reg_pct["Meta"],
        "toe_pct": reg_pct["Toe"],
        "heel_kg": reg_kg["Heel"],
        "mid_kg": reg_kg["Mid"],
        "meta_kg": reg_kg["Meta"],
        "toe_kg": reg_kg["Toe"],
        "seg_rows": f"{counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}",
        "seg_warn": warn,
        "active_first_row": first_r if first_r is not None else np.nan,
        "active_last_row": last_r if last_r is not None else np.nan,
        "active_rows": span_n,
        "foot_length_in": length_in,
        "flags": ";".join(flags),
        "status": "HEALTHY" if not flags else "OUT_OF_RANGE",
    }


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Foot Weight Analytics", layout="wide")
st.title("Foot Weight Analytics (32×32)")

with st.sidebar:
    st.header("Controls")

    threshold = st.slider("Noise threshold (kg/cell)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
    active_ratio = st.slider("Active row rule (row sum ≥ X% total)", min_value=0.1, max_value=5.0,
                             value=DEFAULT_ACTIVE_RATIO * 100.0, step=0.1) / 100.0

    st.subheader("Segmentation rows (vertical)")
    heel_rows = st.number_input("Heel rows", min_value=0, max_value=64, value=DEFAULT_ROW_COUNTS[0], step=1)
    mid_rows  = st.number_input("Mid rows",  min_value=0, max_value=64, value=DEFAULT_ROW_COUNTS[1], step=1)
    meta_rows = st.number_input("Meta rows", min_value=0, max_value=64, value=DEFAULT_ROW_COUNTS[2], step=1)
    toe_rows  = st.number_input("Toe rows",  min_value=0, max_value=64, value=DEFAULT_ROW_COUNTS[3], step=1)

    st.caption("These row counts define the split proportions along the detected foot bounding-box (heel→toe).")

    st.subheader("Color scaling")
    auto_scale = st.checkbox("Auto vmax", value=True)
    vmin = st.number_input("vmin", value=0.0, step=0.1)
    vmax = st.number_input("vmax", value=0.0, step=0.1, help="If Auto vmax is on, this is ignored.")

    show_3d = st.checkbox("Show 3D surface", value=False)

    st.markdown("---")
    st.caption("Tip: For batch analysis, upload a ZIP containing many 32×32 CSV matrices.")


tab1, tab2 = st.tabs(["Single Scan", "Batch Summary (ZIP)"])


with tab1:
    st.subheader("Upload a single 32×32 CSV matrix")
    uploaded = st.file_uploader("Upload a 32×32 CSV", type=["csv"], key="single_csv")

    if uploaded is None:
        st.info("Upload a 32×32 CSV matrix to start.")
    else:
        try:
            mat = load_matrix_32x32_from_bytes(uploaded.getvalue())
        except Exception as e:
            st.error(f"Could not load 32×32 matrix: {e}")
            st.stop()

        # scale
        if auto_scale:
            vmax_eff = float(np.percentile(mat, 99.5)) if np.max(mat) > 0 else 1.0
            vmax_eff = max(1.0, vmax_eff)
        else:
            vmax_eff = vmax if vmax > 0 else max(1.0, float(np.max(mat)))
        vmin_eff = vmin

        fracs, counts, warn = fracs_from_row_counts(heel_rows, mid_rows, meta_rows, toe_rows)
        reg_kg, reg_pct, feet = compute_regions(mat, threshold=threshold, fracs=fracs)
        flags = healthy_flags(reg_pct)

        total = float(np.sum(np.where(mat >= threshold, mat, 0.0)))
        left, right = left_right_load(mat, threshold=threshold)
        cop_r, cop_c = compute_cop(mat, threshold=threshold)
        area = contact_area_in2(mat, threshold=threshold)
        avg = (total / area) if area > 0 else 0.0

        first_r, last_r, span_n = compute_active_row_span(mat, threshold=threshold, ratio=active_ratio)
        length_in = span_n * CELL_PITCH_IN

        # Layout
        colA, colB = st.columns([1.6, 1.0], gap="large")

        with colA:
            title = f"32×32 Heatmap | Foot length: {length_in:.1f} in (rows {span_n})"
            fig2d = plot_heatmap(mat, vmin=vmin_eff, vmax=vmax_eff, title=title)
            st.plotly_chart(fig2d, use_container_width=True)

            if show_3d:
                st.plotly_chart(plot_surface(mat, title="3D Surface"), use_container_width=True)

        with colB:
            st.markdown("### KPIs")
            st.metric("Total load (kg-equiv)", f"{total:.2f}")
            st.metric("Left / Right (kg-equiv)", f"{left:.2f} / {right:.2f}")
            if cop_r is None:
                st.metric("COP (row, col)", "-")
            else:
                st.metric("COP (row, col)", f"{cop_r:.1f}, {cop_c:.1f}")
            st.metric("Contact area (in²)", f"{area:.2f}")
            st.metric("Avg load (kg/in²)", f"{avg:.4f}")
            st.metric("Feet detected", str(feet))
            st.metric("Active rows (>=X% total)", f"{span_n} (r{first_r}–r{last_r})" if first_r is not None else "0")

            st.markdown("### Segmentation")
            st.write(f"Rows H/M/Me/T: **{counts[0]}/{counts[1]}/{counts[2]}/{counts[3]}**")
            st.write(f"Fractions: **{fracs[0]:.2f}/{fracs[1]:.2f}/{fracs[2]:.2f}/{fracs[3]:.2f}**")
            if warn:
                st.warning(warn)

            st.markdown("### Regional distribution")
            st.plotly_chart(plot_region_bars(reg_pct), use_container_width=True)

            if flags:
                st.error("Status: OUT OF RANGE (" + ", ".join(flags) + ")")
            else:
                st.success("Status: HEALTHY (within ranges)")

            # Download summary
            summary = summarize_matrix(mat, name=getattr(uploaded, "name", "upload.csv"),
                                       threshold=threshold,
                                       heel_rows=heel_rows, mid_rows=mid_rows, meta_rows=meta_rows, toe_rows=toe_rows,
                                       active_ratio=active_ratio)
            st.download_button(
                "Download summary CSV (single)",
                data=pd.DataFrame([summary]).to_csv(index=False).encode("utf-8"),
                file_name="foot_scan_summary.csv",
                mime="text/csv"
            )


with tab2:
    st.subheader("Batch summary from a ZIP (many 32×32 CSVs)")
    zip_up = st.file_uploader("Upload ZIP containing CSV files", type=["zip"], key="zip_upload")

    if zip_up is None:
        st.info("Upload a ZIP file containing multiple 32×32 CSV matrices.")
    else:
        try:
            z = zipfile.ZipFile(io.BytesIO(zip_up.getvalue()))
            names = [n for n in z.namelist() if n.lower().endswith(".csv") and ("__macosx" not in n.lower())]
            if not names:
                st.error("ZIP has no CSV files.")
                st.stop()

            rows = []
            bad_files = []
            for n in sorted(names):
                try:
                    mat = load_matrix_32x32_from_bytes(z.read(n))
                    rows.append(summarize_matrix(mat, name=os.path.basename(n),
                                                 threshold=threshold,
                                                 heel_rows=heel_rows, mid_rows=mid_rows, meta_rows=meta_rows, toe_rows=toe_rows,
                                                 active_ratio=active_ratio))
                except Exception as e:
                    bad_files.append((n, str(e)))

            df = pd.DataFrame(rows)
            st.success(f"Processed {len(df)} matrices. Failed: {len(bad_files)}")

            st.dataframe(df, use_container_width=True)

            # Summary charts
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(go.Figure(data=[go.Histogram(x=df["foot_length_in"], nbinsx=20)]).update_layout(
                    title="Foot length distribution (in)", height=360, margin=dict(l=40,r=20,t=40,b=40)
                ), use_container_width=True)
            with c2:
                st.plotly_chart(go.Figure(data=[go.Histogram(x=df["total_kg"], nbinsx=20)]).update_layout(
                    title="Total load distribution (kg-equiv)", height=360, margin=dict(l=40,r=20,t=40,b=40)
                ), use_container_width=True)

            # Out-of-range table
            out_df = df[df["status"] != "HEALTHY"].copy()
            st.markdown("### Out-of-range scans")
            st.write(f"{len(out_df)} / {len(df)} scans flagged as OUT_OF_RANGE.")
            st.dataframe(out_df[["file", "heel_pct", "mid_pct", "meta_pct", "toe_pct", "flags"]], use_container_width=True)

            # Download
            st.download_button(
                "Download batch summary CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="foot_batch_summary.csv",
                mime="text/csv"
            )

            if bad_files:
                st.markdown("### Failed files")
                st.write(pd.DataFrame(bad_files, columns=["file", "error"]))

        except Exception as e:
            st.error(f"Could not read ZIP: {e}")
