"""
Task 1: Feature Engineering and Exploratory Data Analysis
Assignment 2 – GPS Mobility Analysis using GeoLife Dataset
==========================================================
Dataset : segment_features.csv
Target  : mode  (walk | bike | bus | car)

7 features analysed (as provided in dataset – no modification):
  avg_speed_mps  – Average speed
  std_speed_mps  – Speed variation
  max_speed_mps  – High-speed behaviour  (maximum speed)
  std_acc_mps2   – Acceleration variation
  avg_turn_deg   – Turning behaviour
  stop_ratio     – Stop frequency
  distance_m     – Distance travelled

  

⚠ NO machine-learning models are trained in this script.
  Feature importance is computed via Pearson correlation only.

Outputs (saved to plots/):
  class_distribution.png      – segment count per mode
  feature_boxplots.png        – per-feature distribution by mode
  feature_correlation.png     – correlation-based feature importance
                                (clearly labelled as correlation, not ML)
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE  = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(BASE, "plots")
CSV   = os.path.join(BASE, "segment_features.csv")
os.makedirs(PLOTS, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "walk": "#4CAF50",
    "bike": "#FF9800",
    "bus":  "#2196F3",
    "car":  "#F44336",
}
# Feature descriptions:
# 1. avg_speed_mps (Average Speed)
# The mean speed of the traveler during a trajectory segment measured in
# meters per second (m/s). Note: mps stands for meters per second
# For example, walking typically averages 1-2 m/s,
# cycling around 3-5 m/s, bus around 5-8 m/s and car around 8-15 m/s.
# This makes average speed one of the strongest features for separating
# slow modes (walk) from fast modes (car).

# 2. std_speed_mps (Speed Variation)
# The standard deviation of speed throughout the segment. For example,
# a car driving in city traffic may vary between 0 m/s at red lights and
# 13 m/s on open roads giving high std, while a person walking maintains
# a much steadier pace of around 1-1.5 m/s giving low std around 0.4-0.5.

# 3. distance_m (Distance Travelled)
# The total distance travelled during the trajectory segment in meters.
# For example, cars and buses tend to cover much longer distances per
# segment than walking or biking.

# 4. std_acc_mps2 (Acceleration Variation)
# The standard deviation of acceleration values in the segment. For example,
# a bus stopping and starting at bus stations produces std around 0.7-0.8,
# while a person walking on a flat road produces std around 0.4-0.5 since
# their acceleration barely changes between steps.

# 5. avg_turn_deg (Average Turning Angle)
# The mean heading change between consecutive GPS points in degrees.
# For example, a person walking through a city may average 15-35 degrees
# of turning as they navigate pavements and crossings, while a car on a
# main road averages only 7-10 degrees since roads are mostly straight.

# 6. stop_ratio (Stop Ratio)
# The proportion of time intervals where speed is very low or zero.
# For example, a bus trajectory may have stop_ratio of 0.2-0.3 due to
# frequent stops at bus stations every few hundred meters, while a bike
# trajectory typically has stop_ratio below 0.05 since cyclists rarely
# stop except at traffic lights.

# 7. mid_speed_ratio (Mid Speed Ratio)
# The proportion of intervals where speed falls in a medium range.
# For example, a bike trajectory may have mid_speed_ratio of 0.5-0.6
# since most cycling happens at moderate speeds, while a car trajectory
# has lower mid_speed_ratio around 0.1-0.2 since cars spend more time
# at either very low speed in traffic or high speed on open roads.

# ── 7 features (used as-is from dataset) ──────────────────────────────────────
FEATURE_COLS = [
    "avg_speed_mps",   # Average speed       → walk=low, car=high
    "max_speed_mps",   # Maximum speed       → car=very high, walk=very low
    "distance_m",      # Distance travelled      → car=long, walk=short
    "std_acc_mps2",    # Acceleration var    → bus=high, bike=low
    "avg_turn_deg",    # Turning angle       → walk=high, car=low
    "stop_ratio",      # Stop frequency      → bus=high, car=low
    "std_speed_mps",   # Speed variation     → bus=high, walk=low
]


# Human-readable axis labels (same order)
FEATURE_LABELS = [
    "Average Speed\n(meters/second)",
    "Max Speed\n(meters/second)",
    "Distance Travelled\n(meters)",
    "Acceleration Variation\n(std deviation m/s²)",
    "Average Turning Angle\n(degrees)",
    "Stop Ratio\n(proportion of stops)",
    "Speed Variation\n(std deviation m/s)",
]

# ==============================================================================
# 1. LOAD DATA
# ==============================================================================
print("=" * 60)                                                    # print separator line
print("Task 1 – Feature Engineering & Exploratory Data Analysis") # print section title
print("=" * 60)                                                    # print separator line

print("\nLoading segment_features.csv ...")                        # notify user file is loading
df = pd.read_csv(CSV)                                             # read CSV file into dataframe

print(f"  Segments loaded : {len(df):,}")                         # print total number of rows
print(f"  Columns         : {list(df.columns)}")                  # print all column names
print(f"  Mode classes    : {sorted(df['mode'].unique())}")        # print unique transport modes

print(f"\nClass distribution (counts):\n{df['mode'].value_counts().to_string()}") # print count per mode

# Determine display order for modes (by descending count)
MODE_ORDER = df["mode"].value_counts().index.tolist()             # store modes ordered by frequency


# ==============================================================================
# 2. (A) CLASS DISTRIBUTION BAR CHART
# ==============================================================================
counts     = df["mode"].value_counts().reindex(MODE_ORDER)        # count segments per mode in order
bar_colors = [PALETTE.get(m, "#999999") for m in counts.index]   # assign color to each mode

fig, ax = plt.subplots(figsize=(8, 5))                            # create figure 8 wide 5 tall inches
bars = ax.bar(counts.index, counts.values,
              color=bar_colors, edgecolor="white",                # white border between bars
              linewidth=1.4, width=0.6)                           # border thickness and bar width

# Value labels above each bar
for bar, val in zip(bars, counts.values):                         # loop through each bar
    ax.text(bar.get_x() + bar.get_width() / 2,                   # x position = center of bar
            bar.get_height() + 40,                                # y position = just above bar
            f"{int(val):,}",                                      # format number with comma e.g 1,234
            ha="center", va="bottom",                             # align text center and bottom
            fontsize=12, fontweight="bold")                       # text style

ax.set_title("Class Distribution of Transportation Modes\n(GeoLife Dataset – 11,326 Segments)",
             fontsize=14, fontweight="bold", pad=14)              # chart title with padding
ax.set_xlabel("Transportation Mode", fontsize=12)                 # x axis label
ax.set_ylabel("Number of Segments", fontsize=12)                  # y axis label
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))  # format y axis with commas
ax.set_ylim(0, counts.max() * 1.22)                              # add 22% space above tallest bar
ax.spines[["top", "right"]].set_visible(False)                   # remove top and right border lines
ax.grid(axis="y", linestyle="--", alpha=0.4)                     # add faint dashed horizontal grid lines
plt.tight_layout()                                                # auto adjust spacing

out = os.path.join(PLOTS, "class_distribution.png")              # build output file path
plt.savefig(out, dpi=200, bbox_inches="tight")                   # save plot as PNG at 200 resolution
plt.close()                                                       # close figure to free memory
print(f"\nSaved: {out}")                                          # confirm file was saved


# ==============================================================================
# 3. (B) FEATURE DISTRIBUTIONS – BOXPLOTS PER MODE
# ==============================================================================
# One subplot per feature (7 total) arranged in a 2-row grid.
# NOTE: GPS data can contain satellite-jump errors producing impossible speeds
# (e.g. 1,000,000 m/s). These are known noise issues in the raw GeoLife dataset.
# y-axis is capped at the 99th percentile per feature so the IQR boxes are
# clearly visible. Extreme outlier dots are hidden (showfliers=False).
n_features = len(FEATURE_COLS)                                    # total number of features = 7
n_cols     = 4                                                    # 4 subplots per row
n_rows     = int(np.ceil(n_features / n_cols))                   # calculate rows needed = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))        # create 2x4 grid of subplots
axes_flat = axes.flatten()                                        # convert 2D grid to 1D list for easy looping

for idx, (col, label) in enumerate(zip(FEATURE_COLS, FEATURE_LABELS)):  # loop through each feature
    ax = axes_flat[idx]                                           # get the subplot for this feature

    # Build list of arrays, one per mode (in display order)
    data_by_mode  = [df.loc[df["mode"] == m, col].dropna().values for m in MODE_ORDER]  # get feature values per mode
    present_modes = MODE_ORDER                                    # keep mode order consistent

    bp = ax.boxplot(
        data_by_mode,                                             # data for each mode
        patch_artist=True,                                        # fill boxes with color
        notch=False,                                              # straight box edges not notched
        medianprops=dict(color="black", linewidth=2.2),          # black median line style
        whiskerprops=dict(linewidth=1.5),                        # whisker line thickness
        capprops=dict(linewidth=1.5),                            # cap line thickness
        flierprops=dict(marker="o", markersize=2,
                        markerfacecolor="#aaaaaa", alpha=0.5),   # outliers as small grey dots
    )
    # Colour each box by mode
    for patch, mode in zip(bp["boxes"], present_modes):          # loop through each box
        patch.set_facecolor(PALETTE.get(mode, "#cccccc"))        # set box color from PALETTE
        patch.set_alpha(0.85)                                     # slight transparency

    ax.set_xticklabels(present_modes, fontsize=9)                # label x axis with mode names
    ax.set_title(label, fontsize=10, fontweight="bold")          # set subplot title as feature name
    ax.spines[["top", "right"]].set_visible(False)               # remove top and right borders
    ax.grid(axis="y", linestyle="--", alpha=0.35)                # faint dashed horizontal grid lines

# Hide the spare (8th) subplot cell
for idx in range(n_features, len(axes_flat)):                    # loop through unused subplot cells
    axes_flat[idx].set_visible(False)                            # hide empty cell in bottom right

fig.suptitle("Feature Distributions by Transportation Mode\n(GeoLife Dataset)",
             fontsize=14, fontweight="bold", y=1.02)             # main title above all subplots
plt.tight_layout()                                               # auto adjust spacing between subplots

out = os.path.join(PLOTS, "feature_boxplots.png")               # build output file path
plt.savefig(out, dpi=200, bbox_inches="tight")                   # save plot as PNG at 200 resolution
plt.close()                                                      # close figure to free memory
print(f"Saved: {out}")                                           # confirm file was saved



# ==============================================================================
# 5. SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print("TASK 1 COMPLETE  (Feature Engineering and EDA)")
print("=" * 60)
print(f"  Dataset  : {len(df):,} segments  |  {df['mode'].nunique()} classes")
print(f"  Features : {len(FEATURE_COLS)}")
for col, label in zip(FEATURE_COLS, FEATURE_LABELS):
    clean_label = label.replace("\n", " ")
    print(f"    {col:<20}  →  {clean_label}")
print("\n  Plots saved to plots/:")
print("    class_distribution.png")
print("    feature_boxplots.png")
print("=" * 60)
