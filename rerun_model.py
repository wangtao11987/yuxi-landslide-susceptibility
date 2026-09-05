#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rerun_model.py
============================================================================
Reproducible re-run of the IC, RF and IC-RF coupling models used in
"Landslide susceptibility assessment based on an Information Content-Random
Forest coupling model: a case study of Yuxi City, China"
(PLOS ONE, PONE-D-26-33050, revision).

This script reconstructs the modelling pipeline from the four CSV files that
survived the data loss, so every number needed for the revised manuscript
can be recomputed consistently.

Inputs (place in the same folder as this script):
  train_dataset_base.csv     training samples (9 conditioning factors + label)
  test_dataset_base.csv      test samples (9 conditioning factors + label)
  SHAP_Values_Base.csv       SHAP values of the test-set samples
  best_rf_params.csv         tuned RF hyperparameters (optional: if present it
                             is read and used; if missing, the hard-coded
                             BEST_PARAMS below are used instead)

Outputs (written to the same folder):
  results.txt                every number needed for the manuscript
  ic_values_table.csv        per-class information content values (all factors)
  fig2.png                   Pearson correlation heatmap of the nine
                             conditioning factors (publication quality,
                             annotated r values, 600 dpi)
  fig7.png                   two-panel manuscript figure (300 dpi):
                             (a) ROC curves of IC / RF / IC-RF,
                             (b) SHAP beeswarm plot with mean |SHAP|
                             annotations on each row
  fig_roc_curves.png         single-panel ROC curves (intermediate, 300 dpi)
  fig_shap_importance.png    single-panel mean |SHAP| bar chart (intermediate)
  fig_shap_beeswarm.png      SHAP beeswarm (summary) plot, each row annotated
                             with its mean |SHAP| value (300 dpi)
  fig_cv_roc.png             5-fold CV mean ROC curves (new Fig 8, 300 dpi)
  table2_values.csv           TOL / VIF per factor (data source for Table 2)

Requirements:  python 3.7+, pandas, numpy, scikit-learn, matplotlib

Usage:         python rerun_model.py
============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Figure font: Times New Roman (PLOS style) ----
# Falls back gracefully if the font is not installed on the machine
# (Liberation Serif is metrically identical to Times New Roman).
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif",
                              "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score,
                             confusion_matrix)
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42

FEATURES = ["Aspest", "Elevation", "Slope", "MAP",
            "Distance to Roads", "Distance to Faults", "Distance to Rivers", "NDVI", "Lithology"]

REQUIRED_FILES = ["train_dataset_base.csv", "test_dataset_base.csv",
                  "SHAP_Values_Base.csv"]

# Best tuned hyperparameters from the original GridSearchCV run.
# If best_rf_params.csv is present it is read (load_params), otherwise
# these hard-coded values are used. The two must match, and both match
# what the manuscript reports.
BEST_PARAMS = dict(n_estimators=100, max_depth=5,
                   min_samples_split=25, min_samples_leaf=8,
                   max_features="sqrt", random_state=RANDOM_STATE)

OUT = []


def log(msg):
    print(msg)
    OUT.append(str(msg))


# ----------------------------------------------------------------------
# 1. Factor classification (exactly as described in the Methods section)
# ----------------------------------------------------------------------

def classify_elevation(s):
    return pd.cut(s, bins=[-np.inf, 1000, 1500, 2000, 2500, np.inf],
                  labels=["<1000", "1000-1500", "1500-2000", "2000-2500", ">2500"])


def classify_slope(s):
    return pd.cut(s, bins=[-np.inf, 5, 15, 25, 35, 45, np.inf],
                  labels=["<5", "5-15", "15-25", "25-35", "35-45", ">45"])


def classify_aspect(s):
    def one(v):
        if v < 0:
            return "flat"
        if v < 22.5 or v >= 337.5:
            return "N"
        if v < 67.5:
            return "NE"
        if v < 112.5:
            return "E"
        if v < 157.5:
            return "SE"
        if v < 202.5:
            return "S"
        if v < 247.5:
            return "SW"
        if v < 292.5:
            return "W"
        return "NW"
    return s.apply(one)


def classify_ndvi(s):
    return pd.cut(s, bins=[-np.inf, 0.0, 0.2, 0.4, 0.6, np.inf],
                  labels=["<=0", "0-0.2", "0.2-0.4", "0.4-0.6", ">0.6"])


def classify_MAP(s):
    return pd.cut(s, bins=[-np.inf, 800, 900, 1000, 1100, np.inf],
                  labels=["<=800", "800-900", "900-1000", "1000-1100", ">1100"])


def classify_distance(s):
    """Distance factors: if the column already holds small integer class
    codes (as in the surviving CSVs) use them directly, otherwise apply the
    400/800/1200/1600 m breaks described in the paper."""
    if s.max() <= 10 and s.nunique() <= 6:
        return s.round().astype(int).astype(str)
    return pd.cut(s, bins=[-np.inf, 400, 800, 1200, 1600, np.inf],
                  labels=["0-400", "400-800", "800-1200", "1200-1600", ">1600"])


def classify_lithology(s):
    return s.round().astype(int).astype(str)


def make_classes(df):
    """Return a DataFrame of class labels (strings) for every factor."""
    out = pd.DataFrame(index=df.index)
    out["Elevation"] = classify_elevation(df["Elevation"])
    out["Slope"] = classify_slope(df["Slope"])
    out["Aspect"] = classify_aspect(df["Aspest"])
    out["NDVI"] = classify_ndvi(df["NDVI"])
    out["MAP"] = classify_MAP(df["MAP"])
    out["Distance to Roads"] = classify_distance(df["Distance to Roads"])
    out["Distance to Faults"] = classify_distance(df["Distance to Faults"])
    out["Distance to Rivers"] = classify_distance(df["Distance to Rivers"])
    out["Lithology"] = classify_lithology(df["Lithology"])
    return out.astype(str)


# ----------------------------------------------------------------------
# 2. Information Content model
# ----------------------------------------------------------------------

def compute_ic_maps(train_classes, y):
    """Per-class information content, computed from the training set only.
    I = ln((Ni/N) / (Si/S)); Laplace smoothing (+0.5) avoids log(0) when a
    class contains no landslide samples."""
    N = float(y.sum())          # landslide samples
    S = float(len(y))           # all samples
    maps = {}
    for col in train_classes.columns:
        ic = {}
        for cls in sorted(train_classes[col].dropna().unique()):
            mask = (train_classes[col] == cls).values
            Ni = float((mask & (y == 1)).sum())
            Si = float(mask.sum())
            num = (Ni + 0.5) / (N + 0.5)
            den = (Si + 0.5) / (S + 0.5)
            ic[cls] = float(np.log(num / den))
        maps[col] = ic
    return maps


def ic_score(classes_df, maps):
    """Sum of class IC values across all factors for every sample."""
    total = pd.Series(0.0, index=classes_df.index)
    for col in classes_df.columns:
        total = total + classes_df[col].map(maps[col]).fillna(0.0)
    return total.values


# ----------------------------------------------------------------------
# 3. Random Forest helpers
# ----------------------------------------------------------------------

def load_params():
    """Read tuned hyperparameters from best_rf_params.csv if present,
    otherwise fall back to the hard-coded BEST_PARAMS."""
    if os.path.exists("best_rf_params.csv"):
        row = pd.read_csv("best_rf_params.csv").iloc[0]
        params = dict(n_estimators=int(row["n_estimators"]),
                      max_depth=int(row["max_depth"]),
                      min_samples_split=int(row["min_samples_split"]),
                      min_samples_leaf=int(row["min_samples_leaf"]),
                      max_features=str(row["max_features"]).strip())
        params["random_state"] = RANDOM_STATE
        return params
    return dict(BEST_PARAMS)


def make_rf(params):
    return RandomForestClassifier(**params, n_jobs=-1)


# ----------------------------------------------------------------------
# Fig 2: Pearson correlation heatmap (publication quality)
# ----------------------------------------------------------------------

def make_fig2(train, test, save_path="fig2.png", dpi=600):
    """Publication-quality Pearson correlation heatmap of the nine
    conditioning factors (manuscript Fig 2). Correlations are computed
    over all sample points (training + test, N = 1620), matching the
    SPSS-based diagnostics described in the manuscript. The layout
    (factor order, display names) follows Table 2 and the Methods text.
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    all_data = pd.concat([train[FEATURES], test[FEATURES]],
                         ignore_index=True)
    corr = all_data.corr(method="pearson")

    # Display order follows the factor listing in Table 2 / the text.
    display = ["Elevation", "Slope", "Aspect", "Distance to roads",
               "Distance to rivers", "Distance to faults", "Lithology",
               "NDVI", "MAP"]
    col_of = {"Aspect": "Aspest",
              "Distance to rivers": "Distance to Rivers",
              "Distance to roads": "Distance to Roads",
              "Distance to faults": "Distance to Faults"}
    cols = [col_of.get(name, name) for name in display]
    C = corr.loc[cols, cols].values.astype(float)
    n = len(display)

    # Symmetric colour scale: -max|r| .. +max|r| (off-diagonal), so the
    # weak correlations are still clearly visible, not washed out.
    off = np.abs(C[~np.eye(n, dtype=bool)])
    lim = float(off.max()) if off.size else 0.3
    if not np.isfinite(lim) or lim <= 0:
        lim = 0.3
    lim = float(np.ceil(lim * 20) / 20.0)     # round up to the next 0.05

    with plt.rc_context({"font.size": 11, "axes.labelsize": 11.5,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 10.5}):
        fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=dpi)
        im = ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(display, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_yticklabels(display)

        # Thin white separators between cells (minor grid on cell edges).
        edges = np.arange(-0.5, n, 1.0)
        ax.set_xticks(edges, minor=True)
        ax.set_yticks(edges, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.0)
        ax.tick_params(which="minor", length=0)
        ax.tick_params(which="major", length=0)

        # Annotate every cell with the correlation coefficient, using
        # white text on strongly coloured cells and black elsewhere.
        for i in range(n):
            for j in range(n):
                v = C[i, j]
                txt_color = "white" if abs(v) >= 0.55 * lim else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=9.0, color=txt_color)

        # Colour bar aligned to the heatmap height.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4.5%", pad=0.15)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Pearson correlation coefficient", fontsize=10.5)
        cb.set_ticks(np.round(np.linspace(-lim, lim, 5), 2))
        cb.ax.tick_params(labelsize=9.5)

        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def multicollinearity_report(train, test):
    """Recompute the TOL / VIF diagnostics that feed Table 2 from the very
    same Pearson correlation matrix that make_fig2 plots (all sample
    points, train + test). VIF_j = (C^-1)_jj of the correlation matrix,
    TOL_j = 1 / VIF_j, which is algebraically identical to the SPSS
    multicollinearity output. Factor order matches Table 2 row order.
    Writes table2_values.csv and logs the block for results.txt.
    """
    display = ["Elevation", "Slope", "Aspect",
               "Distance to roads", "Distance to rivers",
               "Distance to faults", "Lithology", "NDVI", "MAP"]
    col_of = {"Aspect": "Aspest",
              "Distance to roads": "Distance to Roads",
              "Distance to rivers": "Distance to Rivers",
              "Distance to faults": "Distance to Faults"}
    cols = [col_of.get(name, name) for name in display]

    all_data = pd.concat([train[FEATURES], test[FEATURES]],
                         ignore_index=True)
    C = all_data[cols].corr(method="pearson").values.astype(float)
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:      # singular matrix guard
        C_inv = np.linalg.pinv(C)
    vif = np.diag(C_inv)
    tol = 1.0 / vif

    log("[1b] MULTICOLLINEARITY DIAGNOSTICS (TOL / VIF)")
    log("    Recomputed from the same Pearson correlation matrix used for")
    log(f"    Fig 2 (all {len(all_data)} sample points). "
        "VIF = diag(C^-1), TOL = 1 / VIF.")
    rows = []
    for name, t, v in zip(display, tol, vif):
        log(f"    {name:<20} TOL {t:.3f}   VIF {v:.3f}")
        rows.append({"factor": name, "TOL": round(float(t), 3),
                     "VIF": round(float(v), 3)})
    i_min = int(np.argmin(tol))
    i_max = int(np.argmax(vif))
    log(f"    Minimum TOL = {tol[i_min]:.3f} ({display[i_min]})")
    log(f"    Maximum VIF = {vif[i_max]:.3f} ({display[i_max]})")
    log("    Thresholds: TOL > 0.2 and VIF < 5 -> no multicollinearity")
    log("    among the nine factors.")
    log("")
    pd.DataFrame(rows).to_csv("table2_values.csv", index=False)


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(auc=roc_auc_score(y_true, y_prob),
                accuracy=accuracy_score(y_true, y_pred),
                precision=precision_score(y_true, y_pred, zero_division=0),
                recall=recall_score(y_true, y_pred, zero_division=0),
                f1=f1_score(y_true, y_pred, zero_division=0),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


# ----------------------------------------------------------------------
# 4. Cross-validation (leakage-free: IC maps rebuilt inside each fold)
# ----------------------------------------------------------------------

def cv_auc(X, y, params, n_splits=5, use_ic=False):
    """Stratified 5-fold CV: returns (mean AUC, std AUC, mean ROC curve).

    mean ROC curve = (fpr_grid, mean_tpr, tpr_std) interpolated on a common
    FPR grid, which is what Fig 8 plots. When use_ic=True, class IC maps are
    recomputed on each fold's training part before transforming features
    (no leakage).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    tprs = []
    fpr_grid = np.linspace(0.0, 1.0, 200)
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if use_ic:
            classes_tr = make_classes(X_tr)
            classes_va = make_classes(X_va)
            maps = compute_ic_maps(classes_tr, y_tr)
            X_tr_ = np.vstack([classes_tr[c].map(maps[c]).fillna(0.0).values
                               for c in classes_tr.columns]).T
            X_va_ = np.vstack([classes_va[c].map(maps[c]).fillna(0.0).values
                               for c in classes_va.columns]).T
        else:
            X_tr_ = X_tr[FEATURES].values
            X_va_ = X_va[FEATURES].values
        clf = RandomForestClassifier(**params, n_jobs=-1)
        clf.fit(X_tr_, y_tr)
        proba = clf.predict_proba(X_va_)[:, 1]
        aucs.append(roc_auc_score(y_va, proba))
        fpr, tpr, _ = roc_curve(y_va, proba)
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        tprs[-1][0] = 0.0
    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    return (float(np.mean(aucs)), float(np.std(aucs)),
            fpr_grid, mean_tpr, tprs.std(axis=0))


def draw_beeswarm(ax, shap_df, feat_df, title=None):
    """Draw a SHAP beeswarm (summary) plot on a given Axes.

    Every sample is one dot; the x-axis is the SHAP value and the dot
    colour shows the raw feature value (blue = low, red = high), exactly
    like the classic shap.summary_plot. Features are ordered by mean
    |SHAP| (most important at the top). Each row is annotated on the
    right with its mean |SHAP| value. Requires no 'shap' package.
    """
    DISPLAY_NAMES = {
        "Aspest": "Aspect",
        "Distance to Roads": "Distance to roads",
        "Distance to Faults": "Distance to faults",
        "Distance to Rivers": "Distance to rivers",
    }
    feats = list(shap_df.columns)
    order = (shap_df[feats].abs().mean()
             .sort_values(ascending=False).index.tolist())
    mean_abs = shap_df[order].abs().mean()
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(shap_df)
    y_pos = np.arange(len(order))[::-1]          # top = most important
    xr = float(np.nanmax(np.abs(shap_df[order].values)))  # symmetric scale
    if not np.isfinite(xr) or xr == 0.0:
        xr = 1.0

    for i, f in enumerate(order):
        x = shap_df[f].values.astype(float)
        cvals = feat_df[f].values.astype(float)
        vmin, vmax = float(np.nanmin(cvals)), float(np.nanmax(cvals))
        if vmax == vmin:                         # constant feature guard
            vmax = vmin + 1.0
        y = y_pos[i] + rng.uniform(-0.18, 0.18, size=n)   # jitter
        ax.scatter(x, y, c=cvals, cmap="RdBu_r",
                   norm=plt.Normalize(vmin, vmax),
                   s=8, alpha=0.75, linewidths=0)
        ax.text(xr * 1.08, y_pos[i], f"{mean_abs[f]:.4f}",
                va="center", ha="left", fontsize=10.5, color="black")
    ax.axvline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlim(-xr * 1.12, xr * 1.60)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY_NAMES.get(f, f) for f in order],
                       fontsize=10.5)
    ax.set_xlabel("SHAP value (impact on model output)")
    if title:
        ax.set_title(title, fontsize=12)


def beeswarm_plot(shap_df, feat_df, save_path="fig_shap_beeswarm.png"):
    """Standalone SHAP beeswarm plot saved to save_path (300 dpi)."""
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=300)
    draw_beeswarm(
        ax, shap_df, feat_df,
        title="SHAP beeswarm plot (test set, mean |SHAP| per factor)")
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.06)
    cbar.set_label("Feature value (low \u2192 high)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------

def main():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        print("Missing input file(s): " + ", ".join(missing))
        print("Place them next to this script and run again.")
        sys.exit(1)

    train = pd.read_csv("train_dataset_base.csv")
    test = pd.read_csv("test_dataset_base.csv")
    shap = pd.read_csv("SHAP_Values_Base.csv")
    params = load_params()

    y_tr = train["label"].values
    y_te = test["label"].values
    n_tr_pos = int(y_tr.sum())
    n_tr_neg = int((y_tr == 0).sum())
    n_te_pos = int(y_te.sum())
    n_te_neg = int((y_te == 0).sum())
    total = len(train) + len(test)
    ratio = len(train) / total * 100

    log("=" * 72)
    log("RERUN OF IC / RF / IC-RF MODELS (PONE-D-26-33050)")
    log("=" * 72)
    log("")
    log("[1] DATASET OVERVIEW")
    log(f"    Training samples : {len(train)}  (positive {n_tr_pos}, negative {n_tr_neg})")
    log(f"    Test samples     : {len(test)}  (positive {n_te_pos}, negative {n_te_neg})")
    log(f"    Total            : {total}  (train {ratio:.1f}% / test {100 - ratio:.1f}%)")
    log(f"    RF hyperparameters: {params}")
    log("")

    # ---- multicollinearity diagnostics (TOL / VIF, feeds Table 2)
    multicollinearity_report(train, test)

    # ---- classes
    classes_tr = make_classes(train)
    classes_te = make_classes(test)

    # ---- IC model
    ic_maps = compute_ic_maps(classes_tr, y_tr)
    ic_tr = ic_score(classes_tr, ic_maps)
    ic_te = ic_score(classes_te, ic_maps)
    ic_auc_tr = roc_auc_score(y_tr, ic_tr)
    ic_auc_te = roc_auc_score(y_te, ic_te)
    ic_all = np.concatenate([ic_tr, ic_te])
    log("[2] IC MODEL (information content summed over the 9 factors)")
    log(f"    IC value range      : {ic_all.min():.3f} to {ic_all.max():.3f}")
    log(f"    IC model AUC (train): {ic_auc_tr:.4f}")
    log(f"    IC model AUC (test) : {ic_auc_te:.4f}")
    log("")

    # ---- RF model (raw features)
    rf = make_rf(params)
    rf.fit(train[FEATURES], y_tr)
    rf_tr = rf.predict_proba(train[FEATURES])[:, 1]
    rf_te = rf.predict_proba(test[FEATURES])[:, 1]
    m_tr = metrics(y_tr, rf_tr)
    m_te = metrics(y_te, rf_te)
    log("[3] RF MODEL (raw factors)")
    log(f"    RF AUC (train)      : {m_tr['auc']:.4f}")
    log(f"    RF AUC (test)       : {m_te['auc']:.4f}")
    log(f"    Test accuracy       : {m_te['accuracy']:.4f}   "
        f"precision {m_te['precision']:.4f}   recall {m_te['recall']:.4f}   "
        f"F1 {m_te['f1']:.4f}")
    log(f"    Confusion matrix (test, TN/FP/FN/TP): "
        f"{m_te['tn']}/{m_te['fp']}/{m_te['fn']}/{m_te['tp']}")
    log("")

    # ---- IC-RF coupling model
    X_tr_ic = np.vstack([classes_tr[c].map(ic_maps[c]).fillna(0.0).values
                         for c in classes_tr.columns]).T
    X_te_ic = np.vstack([classes_te[c].map(ic_maps[c]).fillna(0.0).values
                         for c in classes_te.columns]).T
    icrf = make_rf(params)
    icrf.fit(X_tr_ic, y_tr)
    icrf_tr = icrf.predict_proba(X_tr_ic)[:, 1]
    icrf_te = icrf.predict_proba(X_te_ic)[:, 1]
    cm_tr = metrics(y_tr, icrf_tr)
    cm_te = metrics(y_te, icrf_te)
    log("[4] IC-RF COUPLING MODEL (RF trained on IC-transformed factors)")
    log(f"    IC-RF AUC (train)   : {cm_tr['auc']:.4f}")
    log(f"    IC-RF AUC (test)    : {cm_te['auc']:.4f}")
    log(f"    Test accuracy       : {cm_te['accuracy']:.4f}   "
        f"precision {cm_te['precision']:.4f}   recall {cm_te['recall']:.4f}   "
        f"F1 {cm_te['f1']:.4f}")
    log(f"    Confusion matrix (test, TN/FP/FN/TP): "
        f"{cm_te['tn']}/{cm_te['fp']}/{cm_te['fn']}/{cm_te['tp']}")
    log("")

    # ---- 5-fold stratified CV
    full = pd.concat([train, test], ignore_index=True)
    y_full = full["label"].values
    cv_rf = cv_auc(full[FEATURES], y_full, params, use_ic=False)
    cv_icrf = cv_auc(full[FEATURES], y_full, params, use_ic=True)
    log("[5] 5-FOLD STRATIFIED CROSS-VALIDATION (mean AUC +/- std)")
    log(f"    RF    : {cv_rf[0]:.4f} +/- {cv_rf[1]:.4f}")
    log(f"    IC-RF : {cv_icrf[0]:.4f} +/- {cv_icrf[1]:.4f}")
    log("    NOTE: the original manuscript reported 5-fold SPATIAL BLOCK CV")
    log("    (mean AUC 0.7717). Spatial blocks require sample coordinates,")
    log("    which are not present in the surviving CSVs, so stratified")
    log("    k-fold CV is reported instead.")
    log("")

    # ---- SHAP
    mean_shap = shap[FEATURES].abs().mean().sort_values(ascending=False)
    log("[6] SHAP FEATURE IMPORTANCE (mean |SHAP|, test set, "
        "from SHAP_Values_Base.csv)")
    for f, v in mean_shap.items():
        log(f"    {f:<12} {v:.4f}")
    log("")
    log(f"    Top-3: NDVI {mean_shap.get('NDVI', float('nan')):.4f}, "
        f"Elevation {mean_shap.get('Elevation', float('nan')):.4f}, "
        f"Slope {mean_shap.get('Slope', float('nan')):.4f}")
    log("")

    # ---- per-class IC table
    ic_rows = []
    for col in classes_tr.columns:
        for cls, ic in ic_maps[col].items():
            ic_rows.append({"factor": col, "class": cls, "IC": ic})
    pd.DataFrame(ic_rows).to_csv("ic_values_table.csv", index=False)
    log("[7] Per-class IC table written to ic_values_table.csv")

    # ---- figures
    fpr_ic, tpr_ic, _ = roc_curve(y_te, ic_te)
    fpr_rf, tpr_rf, _ = roc_curve(y_te, rf_te)
    fpr_icrf, tpr_icrf, _ = roc_curve(y_te, icrf_te)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    ax.plot(fpr_ic, tpr_ic, label=f"IC (AUC = {ic_auc_te:.4f})", lw=1.6)
    ax.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {m_te['auc']:.4f})", lw=1.6)
    ax.plot(fpr_icrf, tpr_icrf, label=f"IC-RF (AUC = {cm_te['auc']:.4f})",
            lw=2.2, color="black")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (test set)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_roc_curves.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 4.5), dpi=300)
    ax.barh(mean_shap.index[::-1], mean_shap.values[::-1], color="#2b6cb0")
    ax.set_xlabel("Mean |SHAP| value")
    ax.set_title("SHAP feature importance")
    fig.tight_layout()
    fig.savefig("fig_shap_importance.png", dpi=300)
    plt.close(fig)

    # ---- Fig 7 (manuscript): (a) ROC curves, (b) SHAP beeswarm (stacked)
    with plt.rc_context({"font.size": 12.5, "axes.labelsize": 12.5,
                         "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "legend.fontsize": 10.5, "axes.titlesize": 13}):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(6.2, 8.8), dpi=300,
            gridspec_kw={"height_ratios": [1, 1.25]})
        ax1.plot(fpr_ic, tpr_ic, label=f"IC (AUC = {ic_auc_te:.4f})", lw=1.6)
        ax1.plot(fpr_rf, tpr_rf, label=f"RF (AUC = {m_te['auc']:.4f})", lw=1.6)
        ax1.plot(fpr_icrf, tpr_icrf,
                 label=f"IC-RF (AUC = {cm_te['auc']:.4f})",
                 lw=2.2, color="black")
        ax1.plot([0, 1], [0, 1], "--", color="grey", lw=1)
        ax1.set_xlabel("False positive rate")
        ax1.set_ylabel("True positive rate")
        ax1.set_title("(a) ROC curves (test set)")
        ax1.legend(loc="lower right")

        draw_beeswarm(ax2, shap[FEATURES], test[FEATURES],
                      title="(b) SHAP beeswarm plot")
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, pad=0.03)
        cbar.set_label("Feature value (low \u2192 high)", fontsize=11)
        fig.tight_layout()
        fig.savefig("fig7.png", dpi=300)
        plt.close(fig)

    # ---- SHAP beeswarm (summary) plot
    beeswarm_plot(shap[FEATURES], test[FEATURES],
                  save_path="fig_shap_beeswarm.png")

    # ---- Fig 8: 5-fold CV mean ROC curves (RF vs IC-RF)
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=300)
    ax.plot(cv_rf[2], cv_rf[3],
            label=f"RF (CV AUC = {cv_rf[0]:.4f} $\\pm$ {cv_rf[1]:.4f})",
            lw=1.8, color="#3182bd")
    ax.fill_between(cv_rf[2], cv_rf[3] - cv_rf[4], cv_rf[3] + cv_rf[4],
                    color="#3182bd", alpha=0.15)
    ax.plot(cv_icrf[2], cv_icrf[3],
            label=f"IC-RF (CV AUC = {cv_icrf[0]:.4f} $\\pm$ {cv_icrf[1]:.4f})",
            lw=2.2, color="black")
    ax.fill_between(cv_icrf[2], cv_icrf[3] - cv_icrf[4],
                    cv_icrf[3] + cv_icrf[4],
                    color="black", alpha=0.12)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate", fontsize=12)
    ax.set_title("5-fold stratified cross-validation ROC curves",
                 fontsize=12)
    ax.legend(loc="lower right", fontsize=10.5)
    fig.tight_layout()
    fig.savefig("fig_cv_roc.png", dpi=300)
    plt.close(fig)

    # ---- Fig 2: Pearson correlation heatmap of the nine factors
    make_fig2(train, test, save_path="fig2.png")

    log("[8] Figures written: fig2.png (correlation heatmap, 600 dpi), "
        "fig7.png (two-panel manuscript figure), fig_roc_curves.png, "
        "fig_shap_importance.png, fig_shap_beeswarm.png, "
        "fig_cv_roc.png (300 dpi)")

    # ---- notes
    log("")
    log("NOTES")
    log(f"  - Sample total is {total} (the manuscript currently says 1736);")
    log("    update the paper to the real counts printed above.")
    log("  - Zone-area percentages (Fig 6d) and the spatial-block CV require")
    log("    rasters/coordinates that were lost; they cannot be recomputed")
    log("    from these CSVs alone.")

    with open("results.txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(OUT) + "\n")
    print("\nAll results written to results.txt")


if __name__ == "__main__":
    main()
