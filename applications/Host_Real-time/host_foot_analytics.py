#!/usr/bin/env python3
"""
host_foot_analytics_final.py

Single 32×32 matrix Foot Analytics GUI (Tkinter + Matplotlib) with:
- One heatmap (both feet in one 32×32 frame)
- CSV load + optional serial stream (1024 numeric tokens per frame)
- Robust analytics:
  * Total load (kg-equivalent)
  * Left vs right load share (by connected components)
  * COP (center of pressure)
  * Contact area (in²) using cell pitch = 0.5 inch (cell area = 0.25 in²)
  * Heel/Mid/Meta/Toe regional distribution

Segmentation control (your request):
- Four editable boxes letting the user define the *relative vertical rows* per region:
    Heel rows = 4
    Mid  rows = 3
    Meta rows = 5
    Toe  rows = 4
- Backend computation remains the same: we split each detected foot bbox along rows,
  but the split proportions come from these user-defined row counts.

Run:
  python3 host_foot_analytics_final.py
"""

from __future__ import annotations

import os
import re
import csv
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np

try:
    import serial  # pip install pyserial
    import serial.tools.list_ports
except Exception:
    serial = None

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Suppress harmless Matplotlib 3D warning (we do not use 3D in this GUI)
warnings.filterwarnings(
    "ignore",
    message=r"Unable to import Axes3D\..*",
    category=UserWarning,
)

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------
# Constants / configuration
# -----------------------------
N = 32

HEALTHY_RANGES = {
    "Heel": (45.0, 55.0),
    "Mid":  (10.0, 15.0),
    "Meta": (17.0, 27.0),
    "Toe":  (8.0,  13.0),
}

# Default segmentation if user provides invalid row counts
DEFAULT_ROW_COUNTS = (4, 3, 5, 4)  # Heel, Mid, Meta, Toe

# Cell geometry: 2×2 cells = 1 in² => cell pitch ≈ 0.5 in => area 0.25 in²
CELL_PITCH_IN = 0.5
CELL_AREA_IN2 = CELL_PITCH_IN * CELL_PITCH_IN  # 0.25


def fracs_from_row_counts(heel_rows: int, mid_rows: int, meta_rows: int, toe_rows: int):
    """Convert user-provided row counts to fractions that sum to 1.0."""
    vals = [max(0, int(heel_rows)), max(0, int(mid_rows)), max(0, int(meta_rows)), max(0, int(toe_rows))]
    s = sum(vals)
    if s <= 0:
        # fallback to defaults
        d = list(DEFAULT_ROW_COUNTS)
        s2 = sum(d)
        return tuple(v / s2 for v in d), d, "Invalid row counts; using defaults."
    fracs = tuple(v / s for v in vals)
    return fracs, vals, ""


# -----------------------------
# Robust CSV loader
# -----------------------------
def _sniff_delimiter(text: str) -> str:
    c_comma = text.count(",")
    c_tab = text.count("\t")
    if c_tab > c_comma:
        return "\t"
    return ","


def load_matrix_32x32_csv(path: str) -> np.ndarray:
    """Loads a 32×32 matrix from comma/tab/whitespace delimited file."""
    with open(path, "rb") as f:
        head = f.read(4096).decode("utf-8", errors="ignore")
    delim = _sniff_delimiter(head)

    try:
        arr = np.loadtxt(path, delimiter=delim)
    except Exception:
        arr = np.loadtxt(path)

    if arr.shape != (N, N):
        raise ValueError(f"Expected 32×32 but got {arr.shape} from {os.path.basename(path)}")
    return arr.astype(float)


# -----------------------------
# Connected components (4-neighborhood)
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
# Region segmentation + analytics
# -----------------------------
def region_rows_from_bbox(minr: int, maxr: int, fracs: Tuple[float, float, float, float]) -> Dict[str, Tuple[int, int]]:
    """Convert region fractions into row ranges [r0,r1] that cover bbox rows inclusive."""
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

    bounds["Toe"] = (bounds["Toe"][0], maxr)  # clamp
    return bounds


def compute_regions(mat: np.ndarray, threshold: float, fracs: Tuple[float, float, float, float]) -> Dict[str, float]:
    active = mat > threshold
    foot1, foot2 = split_two_feet(active)
    reg = {"Heel": 0.0, "Mid": 0.0, "Meta": 0.0, "Toe": 0.0}

    for comp in (foot1, foot2):
        if comp is None:
            continue
        rows = comp[:, 0]
        minr, maxr = int(rows.min()), int(rows.max())
        bounds = region_rows_from_bbox(minr, maxr, fracs)

        for name, (r0, r1) in bounds.items():
            sel = comp[(rows >= r0) & (rows <= r1)]
            if sel.size == 0:
                continue
            reg[name] += float(mat[sel[:, 0], sel[:, 1]].sum())

    return reg


def compute_cop(mat: np.ndarray, threshold: float) -> Tuple[Optional[float], Optional[float]]:
    w = mat.copy()
    w[w < threshold] = 0.0
    s = float(w.sum())
    if s <= 0:
        return None, None
    rr, cc = np.indices(w.shape)
    return float((rr * w).sum() / s), float((cc * w).sum() / s)


def left_right_load(mat: np.ndarray, threshold: float) -> Tuple[float, float]:
    active = mat > threshold
    foot1, foot2 = split_two_feet(active)

    def load_cent(comp):
        if comp is None:
            return 0.0, None
        vals = mat[comp[:, 0], comp[:, 1]]
        return float(vals.sum()), float(comp[:, 1].mean())

    l1, c1 = load_cent(foot1)
    l2, c2 = load_cent(foot2)

    if foot2 is None or c1 is None:
        return l1, 0.0

    # smaller centroid col => left
    if c1 <= c2:
        return l1, l2
    return l2, l1


def contact_area_in2(mat: np.ndarray, threshold: float) -> float:
    cells = int(np.count_nonzero(mat > threshold))
    return cells * CELL_AREA_IN2

def compute_active_row_span(mat: np.ndarray, threshold: float, ratio: float = 0.01) -> Tuple[Optional[int], Optional[int], int, float]:
    """
    Active vertical row span based on per-row load relative to total.

    Rule (your requirement):
    - Compute total = sum(mat where >= threshold)
    - Compute each row sum
    - Active rows are those with row_sum >= ratio * total  (default ratio=1%)
    - Return (first_row, last_row, span_count, row_threshold)
    """
    w = mat.copy()
    w[w < threshold] = 0.0
    total = float(w.sum())
    if total <= 0:
        return None, None, 0, 0.0
    row_sums = w.sum(axis=1)
    row_thr = ratio * total
    active = np.where(row_sums >= row_thr)[0]
    if active.size == 0:
        return None, None, 0, row_thr
    first = int(active.min())
    last = int(active.max())
    return first, last, int(last - first + 1), float(row_thr)


# -----------------------------
# Serial parsing (optional)
# -----------------------------
class SerialFrameParser:
    number_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

    def __init__(self):
        self.buf: List[float] = []

    def push_line(self, line: str) -> Optional[np.ndarray]:
        toks = self.number_re.findall(line)
        if toks:
            self.buf.extend([float(t) for t in toks])

        if len(self.buf) >= N * N:
            frame = np.array(self.buf[: N * N], dtype=float).reshape((N, N))
            self.buf = self.buf[N * N :]
            return frame
        return None


@dataclass
class FrameStats:
    total_kg: float
    left_kg: float
    right_kg: float
    contact_area_in2: float
    avg_kg_per_in2: float
    heel_pct: float
    mid_pct: float
    meta_pct: float
    toe_pct: float
    cop_r: Optional[float]
    cop_c: Optional[float]
    flags: List[str]


# -----------------------------
# GUI app
# -----------------------------
class FootAnalyticsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Foot Analytics (Single 32×32 Matrix)")
        self.geometry("1200x740")

        # state vars
        self.threshold = tk.DoubleVar(value=0.0)
        self.vmin = tk.DoubleVar(value=0.0)
        self.vmax = tk.DoubleVar(value=0.0)  # 0 -> auto

        # segmentation row counts (editable)
        self.heel_rows = tk.IntVar(value=DEFAULT_ROW_COUNTS[0])
        self.mid_rows  = tk.IntVar(value=DEFAULT_ROW_COUNTS[1])
        self.meta_rows = tk.IntVar(value=DEFAULT_ROW_COUNTS[2])
        self.toe_rows  = tk.IntVar(value=DEFAULT_ROW_COUNTS[3])

        # serial vars
        self.port_var = tk.StringVar(value="/dev/ttyUSB0")
        self.baud_var = tk.IntVar(value=115200)

        self.serial_obj = None
        self.parser = SerialFrameParser()
        self.streaming = False

        self.current = np.zeros((N, N), dtype=float)
        self.latest_stats: Optional[FrameStats] = None

        self._build_ui()
        self.refresh()

    def _build_ui(self):
        # Top toolbar
        bar = ttk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        ttk.Button(bar, text="Open 32×32 CSV", command=self.on_open_csv).pack(side=tk.LEFT, padx=5)
        ttk.Button(bar, text="Save current matrix CSV", command=self.on_save_matrix).pack(side=tk.LEFT, padx=5)
        ttk.Button(bar, text="Save stats CSV", command=self.on_save_stats).pack(side=tk.LEFT, padx=5)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        ttk.Label(bar, text="Serial Port:").pack(side=tk.LEFT)
        ttk.Entry(bar, textvariable=self.port_var, width=14).pack(side=tk.LEFT, padx=4)
        ttk.Label(bar, text="Baud:").pack(side=tk.LEFT)
        ttk.Entry(bar, textvariable=self.baud_var, width=8).pack(side=tk.LEFT, padx=4)

        ttk.Button(bar, text="Ports", command=self.on_list_ports).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Connect", command=self.on_connect).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Start", command=self.on_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(bar, text="Stop", command=self.on_stop).pack(side=tk.LEFT, padx=4)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)

        ttk.Label(bar, text="Threshold:").pack(side=tk.LEFT)
        ttk.Scale(bar, from_=0.0, to=50.0, variable=self.threshold, orient=tk.HORIZONTAL,
                  command=lambda _=None: self.refresh()).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        ttk.Label(bar, text="vmin:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar, textvariable=self.vmin, width=7).pack(side=tk.LEFT)
        ttk.Label(bar, text="vmax (0=auto):").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(bar, textvariable=self.vmax, width=7).pack(side=tk.LEFT)
        ttk.Button(bar, text="Apply scale", command=self.refresh).pack(side=tk.LEFT, padx=6)

        # Main split: plot left, stats right
        body = ttk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Figure
        self.fig = Figure(figsize=(7.8, 6.3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("32×32 Pressure Heatmap")
        self.ax.set_xlabel("Column")
        self.ax.set_ylabel("Row")
        self.im = self.ax.imshow(self.current, origin="upper", interpolation="nearest")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, fraction=0.046, pad=0.04)
        self.cbar.set_label("kg-equiv / cell")

        self.canvas = FigureCanvasTkAgg(self.fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Stats panel
        panel = ttk.Frame(body, width=380)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        panel.pack_propagate(False)

        ttk.Label(panel, text="Foot Analytics", font=("TkDefaultFont", 12, "bold")).pack(anchor="w", pady=(0, 8))

        self.lbl_total = ttk.Label(panel, text="Total: -")
        self.lbl_lr = ttk.Label(panel, text="Left/Right: -")
        self.lbl_area = ttk.Label(panel, text="Area: -")
        self.lbl_cop = ttk.Label(panel, text="COP: -")
        self.lbl_rowspan = ttk.Label(panel, text="Foot length (>=1% rows): -")
        for w in (self.lbl_total, self.lbl_lr, self.lbl_area, self.lbl_cop, self.lbl_rowspan):
            w.pack(anchor="w", pady=2)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # --- Segmentation row controls (visible + editable) ---
        seg_box = ttk.Labelframe(panel, text="Segmentation rows (vertical)")
        seg_box.pack(fill=tk.X, pady=(2, 10))

        def make_num(parent, var):
            try:
                return ttk.Spinbox(parent, from_=0, to=64, textvariable=var, width=7)
            except Exception:
                return ttk.Entry(parent, textvariable=var, width=7)

        items = [("Heel", self.heel_rows), ("Mid", self.mid_rows), ("Meta", self.meta_rows), ("Toe", self.toe_rows)]
        for r, (name, var) in enumerate(items):
            ttk.Label(seg_box, text=f"{name} rows:").grid(row=r, column=0, sticky="w", padx=8, pady=3)
            w = make_num(seg_box, var)
            w.grid(row=r, column=1, sticky="w", padx=6, pady=3)

        btn_row = ttk.Frame(seg_box)
        btn_row.grid(row=4, column=0, columnspan=2, sticky="ew", padx=8, pady=(6, 4))
        ttk.Button(btn_row, text="Apply segmentation", command=self.refresh).pack(side=tk.LEFT)
        ttk.Button(btn_row, text="Reset 4/3/5/4", command=self._reset_seg).pack(side=tk.LEFT, padx=6)

        self.lbl_seginfo = ttk.Label(seg_box, text="Fractions: -", foreground="#666")
        self.lbl_seginfo.grid(row=5, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 2))
        self.lbl_segranges = ttk.Label(seg_box, text="Row ranges: -", foreground="#666")
        self.lbl_segranges.grid(row=6, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 6))

        seg_box.columnconfigure(0, weight=1)
        seg_box.columnconfigure(1, weight=0)

        ttk.Label(panel, text="Regional % (Healthy Ranges)", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.reg_lbls = {}
        for name in ["Heel", "Mid", "Meta", "Toe"]:
            row = ttk.Frame(panel)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=f"{name:>4}:", width=6).pack(side=tk.LEFT)
            lbl = ttk.Label(row, text="- %", width=12)
            lbl.pack(side=tk.LEFT)
            lo, hi = HEALTHY_RANGES[name]
            ttk.Label(row, text=f"[{lo:.0f}–{hi:.0f}%]", foreground="#666").pack(side=tk.LEFT)
            self.reg_lbls[name] = lbl

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.lbl_status = ttk.Label(panel, text="Status: -", font=("TkDefaultFont", 11, "bold"))
        self.lbl_status.pack(anchor="w")

        ttk.Label(panel, text=f"Cell pitch: {CELL_PITCH_IN} in  | Cell area: {CELL_AREA_IN2:.2f} in²",
                  foreground="#666").pack(anchor="w", pady=(16, 0))

    def _reset_seg(self):
        self.heel_rows.set(DEFAULT_ROW_COUNTS[0])
        self.mid_rows.set(DEFAULT_ROW_COUNTS[1])
        self.meta_rows.set(DEFAULT_ROW_COUNTS[2])
        self.toe_rows.set(DEFAULT_ROW_COUNTS[3])
        self.refresh()

    # ---------------- File actions ----------------
    def on_open_csv(self):
        fp = filedialog.askopenfilename(
            title="Select a 32×32 CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not fp:
            return
        try:
            self.current = load_matrix_32x32_csv(fp)
            self.refresh()
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def on_save_matrix(self):
        fp = filedialog.asksaveasfilename(
            title="Save current matrix",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not fp:
            return
        try:
            np.savetxt(fp, self.current, fmt="%.3f", delimiter=",")
            messagebox.showinfo("Saved", f"Saved matrix:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def on_save_stats(self):
        if self.latest_stats is None:
            messagebox.showwarning("No stats", "No stats computed yet.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save stats CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not fp:
            return
        s = self.latest_stats
        row = {
            "total_kg": s.total_kg,
            "left_kg": s.left_kg,
            "right_kg": s.right_kg,
            "contact_area_in2": s.contact_area_in2,
            "avg_kg_per_in2": s.avg_kg_per_in2,
            "heel_pct": s.heel_pct,
            "mid_pct": s.mid_pct,
            "meta_pct": s.meta_pct,
            "toe_pct": s.toe_pct,
            "cop_r": "" if s.cop_r is None else s.cop_r,
            "cop_c": "" if s.cop_c is None else s.cop_c,
            "flags": ";".join(s.flags),
            "seg_rows": f"{self.heel_rows.get()}/{self.mid_rows.get()}/{self.meta_rows.get()}/{self.toe_rows.get()}",
        }
        try:
            with open(fp, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                w.writeheader()
                w.writerow(row)
            messagebox.showinfo("Saved", f"Saved stats:\n{fp}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    # ---------------- Serial actions ----------------
    def on_list_ports(self):
        if serial is None:
            messagebox.showerror("Missing pyserial", "Install with: pip install pyserial")
            return
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            messagebox.showinfo("Ports", "No serial ports found.")
            return
        msg = "\n".join([f"{p.device}  —  {p.description}" for p in ports])
        messagebox.showinfo("Ports", msg)

    def on_connect(self):
        if serial is None:
            messagebox.showerror("Missing pyserial", "Install with: pip install pyserial")
            return
        try:
            if self.serial_obj and self.serial_obj.is_open:
                self.serial_obj.close()
            self.serial_obj = serial.Serial(self.port_var.get(), int(self.baud_var.get()), timeout=0.05)
            self.parser = SerialFrameParser()
            messagebox.showinfo("Serial", f"Connected: {self.port_var.get()} @ {self.baud_var.get()}")
        except Exception as e:
            messagebox.showerror("Serial error", str(e))

    def on_start(self):
        if self.serial_obj is None or not getattr(self.serial_obj, "is_open", False):
            messagebox.showwarning("Serial", "Connect first.")
            return
        self.streaming = True
        self.after(10, self._poll_serial)

    def on_stop(self):
        self.streaming = False

    def _poll_serial(self):
        if not self.streaming or self.serial_obj is None:
            return
        try:
            for _ in range(25):
                line = self.serial_obj.readline()
                if not line:
                    break
                txt = line.decode("utf-8", errors="ignore")
                frame = self.parser.push_line(txt)
                if frame is not None and frame.shape == (N, N):
                    self.current = frame
                    self.refresh()
                    break
        except Exception as e:
            self.streaming = False
            messagebox.showerror("Serial read error", str(e))
            return

        self.after(10, self._poll_serial)

    # ---------------- Analytics + redraw ----------------
    def refresh(self):
        self._update_plot()
        self._update_stats()

    def _update_plot(self):
        try:
            vmin = float(self.vmin.get())
        except Exception:
            vmin = 0.0
        try:
            vmax_entry = float(self.vmax.get())
        except Exception:
            vmax_entry = 0.0

        vmax = vmax_entry if vmax_entry > 0 else float(np.max(self.current)) if self.current.size else 1.0

        self.im.set_data(self.current)
        self.im.set_clim(vmin=vmin, vmax=vmax)
        # Update title with active-row span (computed in _update_stats)
        span_txt = getattr(self, 'active_length_text', '')
        base_title = '32×32 Pressure Heatmap'
        self.ax.set_title(f"{base_title}  |  {span_txt}" if span_txt else base_title)
        self.canvas.draw_idle()

    def _update_stats(self):
        try:
            thr = float(self.threshold.get())
        except Exception:
            thr = 0.0

        w = self.current.copy()
        w[w < thr] = 0.0

        total = float(w.sum())
        area = contact_area_in2(w, threshold=0.0)
        avg = (total / area) if area > 0 else 0.0

        left, right = left_right_load(w, threshold=0.0)
        cop_r, cop_c = compute_cop(w, threshold=0.0)        # Active vertical rows (>=1% of total) -> estimate foot length
        first_row, last_row, span_count, _row_thr = compute_active_row_span(w, threshold=0.0, ratio=0.01)
        if first_row is None:
            self.active_length_text = "Foot length: -"
            self.lbl_rowspan.config(text="Foot length (>=1% rows): -")
        else:
            length_in = span_count * CELL_PITCH_IN  # each row ≈ 0.5 inch
            # show both: length + number of vertical rows used
            self.active_length_text = f"Foot length: {length_in:.1f} in (rows {span_count})"
            self.lbl_rowspan.config(
                text=f"Foot length (>=1% rows): {length_in:.1f} in  | rows={span_count}  | r{first_row}–r{last_row}"
            )
        fracs, counts, warn = fracs_from_row_counts(
            self.heel_rows.get(), self.mid_rows.get(), self.meta_rows.get(), self.toe_rows.get()
        )
        self.lbl_seginfo.config(
            text=f"Fractions (H/M/Me/T): {fracs[0]:.2f}/{fracs[1]:.2f}/{fracs[2]:.2f}/{fracs[3]:.2f}  (rows {counts})"
            + (f"  ⚠ {warn}" if warn else "")
        )

        # show actual row ranges based on largest detected foot bbox
        try:
            active = (w > 0)
            foot1, foot2 = split_two_feet(active)
            comp = foot1 if foot1 is not None else foot2
            if comp is None:
                self.lbl_segranges.config(text="Row ranges: (no foot detected)")
            else:
                rows = comp[:, 0]
                minr, maxr = int(rows.min()), int(rows.max())
                bounds = region_rows_from_bbox(minr, maxr, fracs)
                parts = []
                for k in ["Heel", "Mid", "Meta", "Toe"]:
                    r0, r1 = bounds[k]
                    parts.append(f"{k} {r0}–{r1} ({r1-r0+1})")
                self.lbl_segranges.config(text="Row ranges: " + " | ".join(parts))
        except Exception:
            self.lbl_segranges.config(text="Row ranges: (error computing ranges)")

        reg = compute_regions(w, threshold=0.0, fracs=fracs)
        reg_total = sum(reg.values()) if reg else 0.0

        heel_pct = 100.0 * reg["Heel"] / reg_total if reg_total > 0 else 0.0
        mid_pct  = 100.0 * reg["Mid"]  / reg_total if reg_total > 0 else 0.0
        meta_pct = 100.0 * reg["Meta"] / reg_total if reg_total > 0 else 0.0
        toe_pct  = 100.0 * reg["Toe"]  / reg_total if reg_total > 0 else 0.0

        flags = []
        for name, pct in [("Heel", heel_pct), ("Mid", mid_pct), ("Meta", meta_pct), ("Toe", toe_pct)]:
            lo, hi = HEALTHY_RANGES[name]
            if not (lo <= pct <= hi):
                flags.append(f"{name}_OUT")

        self.latest_stats = FrameStats(
            total_kg=total,
            left_kg=left,
            right_kg=right,
            contact_area_in2=area,
            avg_kg_per_in2=avg,
            heel_pct=heel_pct,
            mid_pct=mid_pct,
            meta_pct=meta_pct,
            toe_pct=toe_pct,
            cop_r=cop_r,
            cop_c=cop_c,
            flags=flags,
        )

        self.lbl_total.config(text=f"Total load: {total:,.2f} kg-equiv")
        self.lbl_lr.config(text=f"Left / Right: {left:,.2f} / {right:,.2f} kg-equiv")
        self.lbl_area.config(text=f"Contact area: {area:,.2f} in²  | Avg: {avg:,.3f} kg/in²")
        if cop_r is None:
            self.lbl_cop.config(text="COP: (row -, col -)")
        else:
            self.lbl_cop.config(text=f"COP: row {cop_r:.1f}, col {cop_c:.1f}")

        # regional label colors
        def set_reg(lbl: ttk.Label, pct: float, rng: Tuple[float, float]):
            ok = rng[0] <= pct <= rng[1]
            lbl.config(text=f"{pct:.2f} %")
            style_name = f"{id(lbl)}.TLabel"
            st = ttk.Style()
            st.configure(style_name, foreground=("#0a7" if ok else "#c00"))
            lbl.configure(style=style_name)

        set_reg(self.reg_lbls["Heel"], heel_pct, HEALTHY_RANGES["Heel"])
        set_reg(self.reg_lbls["Mid"],  mid_pct,  HEALTHY_RANGES["Mid"])
        set_reg(self.reg_lbls["Meta"], meta_pct, HEALTHY_RANGES["Meta"])
        set_reg(self.reg_lbls["Toe"],  toe_pct,  HEALTHY_RANGES["Toe"])

        if not flags:
            self.lbl_status.config(text="Status: HEALTHY (within ranges)", foreground="#0a7")
        else:
            self.lbl_status.config(text=f"Status: OUT OF RANGE ({', '.join(flags)})", foreground="#c00")


def main():
    app = FootAnalyticsApp()
    app.mainloop()


if __name__ == "__main__":
    main()
