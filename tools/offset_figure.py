"""Figure for the camera-offset comparison (from audit/ablation_offsets.json).

    python tools/offset_figure.py audit/ablation_offsets.json paper_figures/fig_offsets.png
"""
import json
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
FAULTS = ("panel_hinge", "antenna_missing", "mli_discoloration")
SERIES = [("3dgs", "3DGS baseline", "#2a78d6", "o", "-"),
          ("warp2d", "Reference image, warped", "#1baf7a", "^", "-."),
          ("ref2d", "Reference image", "#eb6834", "s", "--")]

R = json.load(open(sys.argv[1]))
runs = R["runs"]
deltas = sorted({r["delta"] for r in runs})
x = list(range(len(deltas)))
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), dpi=300)
for key, label, col, mk, ls in SERIES:
    faults, recall = [], []
    for d in deltas:
        rs = [r for r in runs if r["delta"] == d]
        faults.append(100 * sum(r[key][s]["correct"] for r in rs for s in FAULTS) / (len(rs) * len(FAULTS)))
        recall.append(st.mean(r[key][s]["recall"] for r in rs for s in FAULTS))
    kw = dict(color=col, marker=mk, ls=ls, lw=2, ms=6, markeredgecolor="white", markeredgewidth=0.8, label=label, clip_on=False)
    ax[0].plot(x, faults, **kw)
    ax[1].plot(x, recall, **kw)
for a, t, yl in ((ax[0], "(a) Faults correctly reported", "% of 60 fault passes"),
                 (ax[1], "(b) Changed pixels recalled", "Mean pixel recall")):
    a.set_title(t, fontsize=10, loc="left")
    a.set_xticks(x)
    a.set_xticklabels([f"{d:g}" for d in deltas])
    a.set_xlabel("Camera offset from reference pose (degrees)")
    a.set_ylabel(yl)
    a.grid(axis="y", color="#e3e3e0", lw=0.6)
    a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
ax[0].set_ylim(0, 105)
ax[1].set_ylim(0, 1.05)
fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.09, 1, 1), w_pad=2.0)
fig.savefig(sys.argv[2], bbox_inches="tight", facecolor="white")
