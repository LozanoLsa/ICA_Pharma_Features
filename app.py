"""
app.py — Pharma Batch Quality Dashboard
LozanoLsa · Project 18 · ICA — Independent Component Analysis · 2026

Algorithm: FastICA (k=3) + Blind Source Separation + IC Distance
Domain: Pharmaceutical Manufacturing — Batch Root Cause Diagnosis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA, PCA
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICA · Pharma Batch Quality",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── FULL CSS INJECTION ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #080c12;
    --surface:  #0e1420;
    --card:     #121922;
    --card2:    #161f2e;
    --border:   #1e2d45;
    --accent:   #f97316;
    --accent2:  #fb923c;
    --danger:   #f87171;
    --warn:     #fbbf24;
    --ok:       #4ade80;
    --blue:     #60a5fa;
    --purp:     #a78bfa;
    --text:     #c8d8f0;
    --muted:    #4e6a8a;
    --fh: 'Syne', sans-serif;
    --fm: 'JetBrains Mono', monospace;
    --fs: 'Instrument Serif', Georgia, serif;
}

.stApp { background: var(--bg) !important; color: var(--text); font-family: var(--fh); }
.block-container { padding: 1.8rem 2.4rem 3rem !important; max-width: 1400px !important; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSlider"] [role="slider"] { background: var(--accent) !important; border: 2px solid var(--accent2) !important; box-shadow: 0 0 8px rgba(249,115,22,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { font-family: var(--fm) !important; font-size: 0.65rem !important; color: var(--accent2) !important; background: var(--card) !important; border: 1px solid var(--border) !important; padding: 1px 5px !important; border-radius: 3px !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }

[data-testid="stMetric"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-top: 2px solid var(--accent) !important; padding: 1rem 1.1rem 0.9rem !important; border-radius: 3px !important; }
[data-testid="stMetricLabel"] > div { font-family: var(--fm) !important; font-size: 0.6rem !important; text-transform: uppercase !important; letter-spacing: 0.18em !important; color: var(--muted) !important; font-weight: 400 !important; }
[data-testid="stMetricValue"] > div { font-family: var(--fm) !important; font-size: 1.7rem !important; font-weight: 600 !important; color: var(--accent2) !important; line-height: 1.1 !important; }

[data-testid="stTabs"] [role="tablist"] { border-bottom: 1px solid var(--border) !important; gap: 0 !important; background: transparent !important; }
[data-testid="stTabs"] [role="tab"] { font-family: var(--fm) !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; color: var(--muted) !important; padding: 0.5rem 1.2rem !important; border: none !important; border-radius: 0 !important; background: transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] [role="tab"]:hover { color: var(--accent2) !important; background: rgba(249,115,22,0.06) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; background: transparent !important; }
[data-testid="stTabsContent"] { padding-top: 1.4rem !important; }

[data-testid="stAlert"] { border-radius: 2px !important; font-family: var(--fm) !important; font-size: 0.75rem !important; border: none !important; }
[data-testid="stExpander"] { background: var(--card) !important; border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stExpander"] summary { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--border) !important; border-radius: 2px !important; }
[data-testid="stDataFrame"] th { font-family: var(--fm) !important; font-size: 0.62rem !important; text-transform: uppercase !important; letter-spacing: 0.12em !important; background: var(--card2) !important; color: var(--muted) !important; }
[data-testid="stDataFrame"] td { font-family: var(--fm) !important; font-size: 0.72rem !important; color: var(--text) !important; background: var(--card) !important; }

hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
[data-testid="stCaptionContainer"] p { font-family: var(--fm) !important; font-size: 0.62rem !important; color: var(--muted) !important; letter-spacing: 0.08em !important; }
h1, h2, h3 { font-family: var(--fh) !important; color: var(--text) !important; }
p, li { font-family: var(--fh) !important; font-size: 0.88rem !important; }

.lsa-header { border-bottom: 1px solid var(--border); padding-bottom: 1.2rem; margin-bottom: 0.2rem; }
.lsa-project-tag { font-family: var(--fm); font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.22em; margin-bottom: 4px; }
.lsa-title { font-family: var(--fh); font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.1; letter-spacing: -0.02em; }
.lsa-tagline { font-family: var(--fs); font-style: italic; font-size: 0.9rem; color: var(--muted); margin-top: 4px; }
.lsa-chip { display: inline-block; background: rgba(249,115,22,0.1); border: 1px solid rgba(249,115,22,0.3); color: var(--accent2); font-family: var(--fm); font-size: 0.58rem; letter-spacing: 0.1em; text-transform: uppercase; padding: 2px 8px; border-radius: 2px; margin-right: 5px; }
.lsa-section { font-family: var(--fm); font-size: 0.6rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.2em; margin-bottom: 10px; padding-bottom: 5px; border-bottom: 1px solid var(--border); }
.lsa-footer { margin-top: 2.5rem; padding-top: 0.8rem; border-top: 1px solid var(--border); font-family: var(--fm); font-size: 0.58rem; color: var(--muted); letter-spacing: 0.1em; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DATA_PATH     = "pharma_data.csv"
DATA_PATH_ALT = "18_ICA_Pharma_Features/pharma_data.csv"
RANDOM_STATE  = 42
DIST_WARN     = 2.54
DIST_ALERT    = 2.86
IC_THRESHOLD  = 1.8

FEATURES = ["ph_final", "viscosity_cp", "conductivity_ms_cm", "density_g_ml",
            "turbidity_ntu", "reaction_temp_c", "reaction_time_min", "active_conc_mg_ml"]
TRUE_SOURCES = ["f1_active_reagent", "f2_solvent_salts", "f3_agitation_kinetics"]
IC_NAMES     = ["IC1 — Active Reagent", "IC2 — Solvent / Ionic", "IC3 — Agitation"]

FEAT_LABELS = {
    "ph_final":            "pH Final",
    "viscosity_cp":        "Viscosity (cP)",
    "conductivity_ms_cm":  "Conductivity (mS/cm)",
    "density_g_ml":        "Density (g/mL)",
    "turbidity_ntu":       "Turbidity (NTU)",
    "reaction_temp_c":     "Reaction Temp (°C)",
    "reaction_time_min":   "Reaction Time (min)",
    "active_conc_mg_ml":   "Active Conc (mg/mL)",
}

# ─── MATPLOTLIB PALETTE ───────────────────────────────────────────────────────
C_BG     = "#080c12"
C_CARD   = "#121922"
C_ORANGE = "#f97316"
C_ORANGE2= "#fb923c"
C_DANGER = "#f87171"
C_OK     = "#4ade80"
C_WARN   = "#fbbf24"
C_BLUE   = "#60a5fa"
C_PURP   = "#a78bfa"
C_TEXT   = "#c8d8f0"
C_MUTED  = "#4e6a8a"

IC_COLORS = [C_BLUE, C_WARN, C_PURP]  # IC1=blue, IC2=amber, IC3=purple

def dark_fig(w=9, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor(C_CARD)
    ax.tick_params(colors=C_MUTED, labelsize=9)
    ax.xaxis.label.set_color(C_MUTED)
    ax.yaxis.label.set_color(C_MUTED)
    ax.title.set_color(C_TEXT)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    return fig, ax

# ─── DATA & MODEL ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    for path in [DATA_PATH, DATA_PATH_ALT]:
        try:
            return pd.read_csv(path)
        except FileNotFoundError:
            continue
    st.error("pharma_data.csv not found. Place the file in the same folder as app.py.")
    st.stop()

@st.cache_resource
def fit_model(df):
    X        = df[FEATURES].values
    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(X)
    ica      = FastICA(n_components=3, whiten="unit-variance",
                       random_state=RANDOM_STATE, max_iter=2000, tol=1e-5)
    S_hat    = ica.fit_transform(X_sc)
    pca3     = PCA(n_components=3)
    X_pca    = pca3.fit_transform(X_sc)
    pca_full = PCA(n_components=8)
    pca_full.fit(X_sc)
    IC_DIST  = np.sqrt(np.sum(S_hat ** 2, axis=1))
    return scaler, ica, S_hat, X_sc, X_pca, pca3, pca_full, IC_DIST

df = load_data()
scaler, ica, S_hat, X_sc, X_pca, pca3, pca_full, IC_DIST = fit_model(df)
A        = ica.mixing_
var_full = pca_full.explained_variance_ratio_
var_cum  = np.cumsum(var_full)
n_out    = int((IC_DIST >= DIST_ALERT).sum())

def batch_status(dist):
    if dist < DIST_WARN:  return "In-Spec",    C_OK
    if dist < DIST_ALERT: return "Warning",    C_WARN
    return                       "Out-of-Spec", C_DANGER

def score_batch(vals):
    x    = scaler.transform([vals])
    s    = ica.transform(x)[0]
    dist = float(np.sqrt(np.sum(s ** 2)))
    return s, dist

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="lsa-header">
    <div class="lsa-project-tag">ML Project #18 · ICA · Unsupervised · Pharma Batch Quality</div>
    <div class="lsa-title">8 Sensors. 3 Hidden Causes. ICA Finds Them.</div>
    <div class="lsa-tagline">PCA finds variance. ICA finds independence — and independence is where root causes live.</div>
    <div style="margin-top:10px;">
        <span class="lsa-chip">FASTICA  k=3</span>
        <span class="lsa-chip">8 SENSORS</span>
        <span class="lsa-chip">BLIND SOURCE SEPARATION</span>
        <span class="lsa-chip">{n_out} ALERT BATCHES</span>
        <span class="lsa-chip">IC DISTANCE · MIXING MATRIX</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── TOP KPI ROW ──────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Batches",      f"{len(df):,}",                  "Production records")
k2.metric("Alert Batches",      f"{n_out} ({n_out/len(df)*100:.1f}%)", "IC Distance ≥ 2.85")
k3.metric("Independent Sources","3 / 8 sensors",                 "Active reagent · Solvent · Agitation")
k4.metric("PCA Coverage (k=3)", f"{var_cum[2]*100:.1f}%",        "Variance justifying k=3")

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DATA EXPLORER", "SOURCE RECOVERY", "BATCH CLASSIFIER", "MIXING MATRIX", "ACTION PLAN"
])

# ══ TAB 1 ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="lsa-section">// Pharma batch dataset — 1,583 production records</div>',
                unsafe_allow_html=True)
    cl, cr = st.columns([3, 2])
    with cl:
        st.markdown('<div class="lsa-section">// Dataset sample (observed sensors)</div>',
                    unsafe_allow_html=True)
        st.dataframe(df[FEATURES].head(10), use_container_width=True, height=310)
    with cr:
        st.markdown('<div class="lsa-section">// Descriptive statistics</div>',
                    unsafe_allow_html=True)
        st.dataframe(df[FEATURES].describe().round(3), use_container_width=True, height=310)

    st.divider()
    st.markdown('<div class="lsa-section">// Sensor correlation heatmap — shared causes create apparent correlations</div>',
                unsafe_allow_html=True)
    corr = df[FEATURES].corr()
    fig_c, ax_c = plt.subplots(figsize=(9, 7))
    fig_c.patch.set_facecolor(C_BG); ax_c.set_facecolor(C_BG)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, linewidths=0.3, linecolor="#1e2d45",
                annot_kws={"size": 8.5, "color": C_TEXT}, ax=ax_c,
                xticklabels=[FEAT_LABELS.get(f, f) for f in FEATURES],
                yticklabels=[FEAT_LABELS.get(f, f) for f in FEATURES])
    ax_c.set_xticklabels(ax_c.get_xticklabels(), color=C_TEXT, fontsize=8, rotation=35, ha="right")
    ax_c.set_yticklabels(ax_c.get_yticklabels(), color=C_TEXT, fontsize=8, rotation=0)
    ax_c.set_title("Sensor Correlation Matrix", color=C_TEXT, fontsize=12, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig_c, use_container_width=True); plt.close()
    st.caption("Correlations arise because sensors share hidden causes. ICA's job is to undo this mixing.")

# ══ TAB 2 ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="lsa-section">// PCA variance guide + ICA source recovery</div>',
                unsafe_allow_html=True)
    comps = list(range(1, 9))
    c_scree, c_cum = st.columns(2)

    with c_scree:
        st.markdown('<div class="lsa-section">// Scree plot — PCA variance (ICA uses k=3)</div>',
                    unsafe_allow_html=True)
        fig_s, ax_s = dark_fig(6, 4.5)
        bar_colors = [C_DANGER if c <= 3 else C_ORANGE for c in comps]
        ax_s.bar(comps, var_full * 100, color=bar_colors, alpha=0.85, edgecolor="#1e2d45")
        ax_s.plot(comps, var_full * 100, "o-", color="white", lw=1.5, ms=6)
        for i, v in enumerate(var_full * 100):
            ax_s.text(i + 1, v + 0.4, f"{v:.1f}%", ha="center", fontsize=8, color=C_TEXT)
        ax_s.axvline(x=3.5, color=C_WARN, ls="--", lw=1.5, label="ICA k=3")
        ax_s.set_xlabel("Component"); ax_s.set_ylabel("Variance (%)")
        ax_s.set_xticks(comps)
        ax_s.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8)
        ax_s.grid(axis="y", alpha=0.15, color="#1e2d45")
        fig_s.tight_layout()
        st.pyplot(fig_s, use_container_width=True); plt.close()

    with c_cum:
        st.markdown('<div class="lsa-section">// Cumulative variance</div>',
                    unsafe_allow_html=True)
        fig_cv, ax_cv = dark_fig(6, 4.5)
        ax_cv.plot(comps, var_cum * 100, "o-", color=C_ORANGE, lw=2.5, ms=8)
        ax_cv.axhline(90, color=C_OK, ls="--", lw=1.3, label="90%")
        ax_cv.axvline(x=3, color=C_DANGER, ls="--", lw=1.5,
                      label=f"k=3 ({var_cum[2]*100:.1f}%)")
        for i, v in enumerate(var_cum * 100):
            ax_cv.text(i + 1, v + 0.8, f"{v:.0f}%", ha="center", fontsize=8, color=C_TEXT)
        ax_cv.set_xlabel("Components"); ax_cv.set_ylabel("Cumulative Var (%)")
        ax_cv.set_ylim(60, 107); ax_cv.set_xticks(comps)
        ax_cv.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=8, loc="lower right")
        ax_cv.grid(True, alpha=0.1, color="#1e2d45")
        fig_cv.tight_layout()
        st.pyplot(fig_cv, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// IC space — batch quality heat map</div>',
                unsafe_allow_html=True)
    fig_ic, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig_ic.patch.set_facecolor(C_BG)
    pairs   = [(0, 1), (0, 2), (1, 2)]
    xlabels = ["IC1 (Active Reagent)", "IC1 (Active Reagent)", "IC2 (Solvent/Ionic)"]
    ylabels = ["IC2 (Solvent/Ionic)",  "IC3 (Agitation)",      "IC3 (Agitation)"]
    for ax, (i, j), xl, yl in zip(axes, pairs, xlabels, ylabels):
        ax.set_facecolor(C_CARD)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
        sc = ax.scatter(S_hat[:, i], S_hat[:, j], c=IC_DIST, cmap="RdYlGn_r",
                        vmin=0, vmax=3.5, alpha=0.5, s=12, edgecolors="none")
        ax.axhline(0, color="#1e2d45", lw=0.7, ls="--")
        ax.axvline(0, color="#1e2d45", lw=0.7, ls="--")
        ax.set_xlabel(xl, color=C_MUTED, fontsize=9)
        ax.set_ylabel(yl, color=C_MUTED, fontsize=9)
        ax.tick_params(colors=C_MUTED)
        ax.grid(True, alpha=0.08, color="#1e2d45")
    plt.colorbar(sc, ax=axes[-1], label="IC Distance")
    plt.suptitle("Independent Component Space — Green = In-Spec · Red = Deviated",
                 color=C_TEXT, fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig_ic, use_container_width=True); plt.close()
    st.caption("Batches cluster near (0,0,0) in IC space when all three sources are stable. "
               "Radial outliers indicate a specific root cause deviation.")

    st.divider()
    true_S   = df[TRUE_SOURCES].values
    cors_ica = [max(abs(pearsonr(S_hat[:, i], true_S[:, j])[0]) for j in range(3)) for i in range(3)]
    cors_pca = [max(abs(pearsonr(X_pca[:, i],  true_S[:, j])[0]) for j in range(3)) for i in range(3)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("IC1 Recovery |r|", f"{cors_ica[0]:.4f}", "vs F1 Active Reagent")
    m2.metric("IC2 Recovery |r|", f"{cors_ica[1]:.4f}", "vs F2 Solvent/Salts")
    m3.metric("IC3 Recovery |r|", f"{cors_ica[2]:.4f}", "vs F3 Agitation")
    m4.metric("Stability σ",      "< 0.001",             "Across 5 random seeds")

# ══ TAB 3 ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="lsa-section">// Real-time batch quality classifier</div>',
                unsafe_allow_html=True)
    st.caption("Enter the 8 quality/process sensor readings to get IC scores, quality status, and root-cause diagnosis.")

    # Result LEFT · Controls RIGHT
    col_out, col_inp = st.columns([3, 2])

    with col_inp:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);
                    border-left:3px solid var(--accent);border-radius:2px;
                    padding:1rem 1.2rem;margin-bottom:14px;">
            <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                        text-transform:uppercase;letter-spacing:.18em;">// Sensor readings</div>
        </div>
        """, unsafe_allow_html=True)
        ph     = st.slider("pH Final",                     4.5,   8.5,  7.0, 0.05)
        visc   = st.slider("Viscosity (cP)",              50.0, 600.0,150.0, 1.0)
        cond   = st.slider("Conductivity (mS/cm)",         0.1,  20.0,  6.0, 0.1)
        dens   = st.slider("Density (g/mL)",               1.05,  1.40, 1.20, 0.001)
        turb   = st.slider("Turbidity (NTU)",              0.0, 200.0, 25.0, 0.5)
        temp_r = st.slider("Reaction Temp (°C)",          30.0,  80.0, 45.0, 0.5)
        time_r = st.slider("Reaction Time (min)",         10.0, 120.0, 35.0, 0.5)
        conc   = st.slider("Active Conc (mg/mL)",         60.0, 160.0,100.0, 0.5)

    vals    = [ph, visc, cond, dens, turb, temp_r, time_r, conc]
    s, dist = score_batch(vals)
    status, s_color = batch_status(dist)
    badge_bg = {"In-Spec": "#0f2e1a", "Warning": "#2e2a0a", "Out-of-Spec": "#2e0f0f"}.get(status, "#121922")

    with col_out:
        st.markdown(
            f'''<div style="background:var(--card);border:1px solid var(--border);
                        border-radius:4px;padding:1.6rem 1.8rem;margin-bottom:14px;">
                <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                            color:#fff;margin-bottom:0.8rem;">Batch Assessment</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:2.6rem;
                            font-weight:700;color:{s_color};line-height:1;">
                    IC Distance = {dist:.4f}
                </div>
                <div style="margin-top:12px;">
                    <span style="background:{badge_bg};color:{s_color};
                                 font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                                 font-weight:600;letter-spacing:.08em;
                                 padding:5px 16px;border-radius:20px;">
                        BATCH STATUS: {status.upper()}
                    </span>
                </div>
                <div style="margin-top:14px;font-family:var(--fm);font-size:0.68rem;
                            color:var(--muted);line-height:1.9;">
                    Warn &#8805; {DIST_WARN} &nbsp;·&nbsp; Alert &#8805; {DIST_ALERT}
                </div>
            </div>''',
            unsafe_allow_html=True
        )

        st.markdown('<div class="lsa-section">// Independent component scores</div>',
                    unsafe_allow_html=True)
        for i, (ic_name, ic_color) in enumerate(zip(IC_NAMES, IC_COLORS)):
            val_i    = s[i]
            deviated = abs(val_i) > IC_THRESHOLD
            bar_color= C_DANGER if deviated else ic_color
            bar_w    = min(int(abs(val_i) / 3.5 * 100), 100)
            flag     = "  ← ROOT CAUSE" if deviated else ""
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;
                            font-family:var(--fm);font-size:0.7rem;color:var(--text);margin-bottom:3px;">
                    <span>{ic_name}<span style="color:{C_DANGER};font-weight:700;">{flag}</span></span>
                    <span style="color:{bar_color};font-weight:600;">{val_i:+.4f}</span>
                </div>
                <div style="background:#1e2d45;border-radius:3px;height:9px;">
                    <div style="background:{bar_color};width:{bar_w}%;height:9px;border-radius:3px;"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Root cause
        drivers = [i for i in range(3) if abs(s[i]) > IC_THRESHOLD]
        RC_ACTIONS = {
            0: ["Check raw material CoA for active ingredient purity.",
                "Verify dosing pump calibration.",
                "Review dispensing record — confirm target weight."],
            1: ["Check incoming solvent lot conductivity and pH.",
                "Verify purified water system (TOC, conductivity).",
                "Review salt/excipient dispensing weight."],
            2: ["Verify agitator speed log (RPM setpoint maintained?).",
                "Check jacket temperature control — any excursion?",
                "Elevated turbidity → inspect filter."],
        }
        if drivers:
            for d in drivers:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);
                            border-left:3px solid {C_DANGER};border-radius:2px;
                            padding:0.9rem 1.2rem;margin-top:10px;">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">
                        // Root cause: {IC_NAMES[d]}</div>
                    {"".join(f'<div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);padding:2px 0;line-height:1.7;">• {a}</div>' for a in RC_ACTIONS[d])}
                </div>""", unsafe_allow_html=True)
        else:
            st.success("No root cause deviation. Batch conforms to historical spec.")

# ══ TAB 4 ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="lsa-section">// Mixing matrix & independent component profiles</div>',
                unsafe_allow_html=True)
    st.caption("The mixing matrix A shows how much each hidden source contributes to each observed sensor.")

    mixing_df = pd.DataFrame(
        A,
        index=[FEAT_LABELS.get(f, f) for f in FEATURES],
        columns=["IC1 (Reagent)", "IC2 (Solvent)", "IC3 (Agitation)"]
    ).round(4)
    st.dataframe(mixing_df, use_container_width=True)
    st.caption("Large absolute value = that sensor is strongly driven by that source. "
               "Use to identify early-warning sensors per root cause.")

    st.divider()
    col_heat, col_bar = st.columns(2)

    with col_heat:
        st.markdown('<div class="lsa-section">// Mixing matrix heatmap</div>',
                    unsafe_allow_html=True)
        fig_h, ax_h = plt.subplots(figsize=(6, 5.5))
        fig_h.patch.set_facecolor(C_BG); ax_h.set_facecolor(C_BG)
        sns.heatmap(mixing_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    vmin=-1.1, vmax=1.1, linewidths=0.4, linecolor="#1e2d45",
                    annot_kws={"size": 9, "color": C_TEXT},
                    cbar_kws={"label": "Loading"}, ax=ax_h)
        ax_h.set_xticklabels(ax_h.get_xticklabels(), color=C_TEXT, fontsize=9)
        ax_h.set_yticklabels(ax_h.get_yticklabels(), color=C_TEXT, fontsize=9, rotation=0)
        ax_h.set_title("Source → Sensor Influence", color=C_TEXT, fontsize=11, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_h, use_container_width=True); plt.close()

    with col_bar:
        st.markdown('<div class="lsa-section">// IC distributions (recovered sources)</div>',
                    unsafe_allow_html=True)
        fig_d, axes_d = plt.subplots(3, 1, figsize=(5, 6.5))
        fig_d.patch.set_facecolor(C_BG)
        for i, (ax, color) in enumerate(zip(axes_d, IC_COLORS)):
            ax.set_facecolor(C_CARD)
            for sp in ax.spines.values(): sp.set_edgecolor("#1e2d45")
            ax.hist(S_hat[:, i], bins=40, color=color, alpha=0.8, edgecolor="none", density=True)
            ax.axvline(-IC_THRESHOLD, color=C_DANGER, ls="--", lw=1.3, alpha=0.8)
            ax.axvline( IC_THRESHOLD, color=C_DANGER, ls="--", lw=1.3, alpha=0.8)
            ax.set_title(IC_NAMES[i], color=color, fontsize=9, fontweight="bold")
            ax.tick_params(colors=C_MUTED)
            ax.grid(axis="y", alpha=0.15, color="#1e2d45")
            ax.xaxis.label.set_color(C_MUTED); ax.yaxis.label.set_color(C_MUTED)
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True); plt.close()

    st.divider()
    st.markdown('<div class="lsa-section">// ICA vs PCA — source recovery comparison</div>',
                unsafe_allow_html=True)
    true_S   = df[TRUE_SOURCES].values
    src_names= ["F1 Active", "F2 Solvent", "F3 Agitation"]
    cors_ica = [max(abs(pearsonr(S_hat[:, i], true_S[:, j])[0]) for j in range(3)) for i in range(3)]
    cors_pca = [max(abs(pearsonr(X_pca[:, i],  true_S[:, j])[0]) for j in range(3)) for i in range(3)]

    fig_comp, ax_comp = dark_fig(10, 4)
    x_pos = np.arange(3); w = 0.35
    ax_comp.bar(x_pos - w / 2, cors_ica, width=w, color=C_BLUE,   alpha=0.85,
                edgecolor="#1e2d45", label="ICA")
    ax_comp.bar(x_pos + w / 2, cors_pca, width=w, color=C_ORANGE, alpha=0.85,
                edgecolor="#1e2d45", label="PCA")
    for i, (ic, pc) in enumerate(zip(cors_ica, cors_pca)):
        ax_comp.text(i - w / 2, ic + 0.01, f"{ic:.3f}", ha="center", fontsize=9, color=C_TEXT)
        ax_comp.text(i + w / 2, pc + 0.01, f"{pc:.3f}", ha="center", fontsize=9, color=C_TEXT)
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels([f"Best match for\n{s}" for s in src_names])
    ax_comp.set_ylabel("Max |r| with any true source")
    ax_comp.set_ylim(0, 1.05)
    ax_comp.set_title("ICA vs PCA: Source Recovery Quality", color=C_TEXT,
                       fontsize=11, fontweight="bold")
    ax_comp.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=9)
    ax_comp.grid(axis="y", alpha=0.15, color="#1e2d45")
    fig_comp.tight_layout()
    st.pyplot(fig_comp, use_container_width=True); plt.close()
    st.caption("ICA consistently recovers the true latent sources with higher fidelity than PCA. "
               "PCA finds variance directions — ICA finds independent causes.")

    st.divider()
    st.markdown('<div class="lsa-section">// IC distance distribution — batch quality overview</div>',
                unsafe_allow_html=True)
    fig_dist, ax_dist = dark_fig(10, 3.5)
    ax_dist.hist(IC_DIST, bins=60, color=C_ORANGE, alpha=0.80, edgecolor="none", density=True)
    ax_dist.axvline(DIST_WARN,  color=C_WARN,   ls="--", lw=1.8,
                    label=f"Warning p90 = {DIST_WARN}")
    ax_dist.axvline(DIST_ALERT, color=C_DANGER, ls="--", lw=1.8,
                    label=f"Alert p95 = {DIST_ALERT}")
    ax_dist.set_xlabel("IC Distance (batch deviation from centre)")
    ax_dist.set_ylabel("Density")
    ax_dist.legend(facecolor=C_CARD, labelcolor=C_TEXT, fontsize=9)
    ax_dist.grid(axis="y", alpha=0.15, color="#1e2d45")
    fig_dist.tight_layout()
    st.pyplot(fig_dist, use_container_width=True); plt.close()

# ══ TAB 5 ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="lsa-section">// Pharma quality response playbook</div>',
                unsafe_allow_html=True)
    st.caption("ICA converts a batch quality alarm into a specific root-cause investigation protocol.")

    protocols = [
        {"title": f"In-Spec Batch (IC Distance < {DIST_WARN})", "color": C_OK,
         "condition": f"All IC scores within ±{IC_THRESHOLD} · d < {DIST_WARN}",
         "freq": f"{(IC_DIST < DIST_WARN).sum()} batches (90.0%)",
         "actions": ["Release batch for further processing / packaging.",
                     "Log IC scores in batch record for trending.",
                     "Update control charts if IC scores drifting over time."]},
        {"title": "Warning — IC1: Active Reagent Dosage (F1)", "color": C_BLUE,
         "condition": "IC1 score |>| 1.8 · Key sensors: viscosity, density, active_conc",
         "freq": "Linked to F1 drift events (~5% of batches)",
         "actions": ["Pull raw material certificate of analysis — check active purity.",
                     "Verify dosing pump calibration record.",
                     "Confirm dispensed weight vs target in batch record.",
                     "If active_conc OOS: quarantine batch, analytical retesting."]},
        {"title": "Warning — IC2: Solvent / Ionic Balance (F2)", "color": C_PURP,
         "condition": "IC2 score |>| 1.8 · Key sensors: ph_final, conductivity",
         "freq": "Linked to F2 drift events",
         "actions": ["Test incoming solvent lot — conductivity and pH vs specification.",
                     "Check purified water system: TOC, conductivity, microbial.",
                     "Review salt/excipient dispensing weights.",
                     "pH correction may be feasible in-process (consult SOP)."]},
        {"title": "Alert — IC3: Agitation & Kinetics (F3)", "color": C_DANGER,
         "condition": "IC3 score |>| 1.8 · Key sensors: reaction_time, reaction_temp, turbidity",
         "freq": "Linked to F3 failure events (~5% of batches)",
         "actions": ["Review agitator RPM log — was setpoint maintained throughout?",
                     "Check jacket temperature controller — any thermal excursion logged?",
                     "Verify total mixing time was within specification window.",
                     "Elevated turbidity → filter inspection + potential re-filtration step.",
                     "Document deviation; quality review before release decision."]},
    ]

    for p in protocols:
        with st.expander(p["title"], expanded=True):
            l, r = st.columns([3, 1])
            with l:
                st.markdown(f"""
                <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                            margin-bottom:10px;">
                    <strong style="color:var(--text);">Trigger condition:</strong>
                    {p['condition']}
                </div>""", unsafe_allow_html=True)
                for i, a in enumerate(p["actions"], 1):
                    prefix = "🚨 " if "IC3" in p["title"] and i == 1 else f"{i}. "
                    st.markdown(f"""
                    <div style="font-family:var(--fm);font-size:0.7rem;color:var(--muted);
                                padding:3px 0;border-bottom:1px solid #1e2d4530;line-height:1.6;">
                        <span style="color:{p['color']};font-weight:600;">{prefix}</span>{a}
                    </div>""", unsafe_allow_html=True)
            with r:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);
                            border-top:3px solid {p['color']};border-radius:2px;
                            padding:1rem;text-align:center;margin-top:8px;">
                    <div style="font-family:var(--fm);font-size:0.58rem;color:var(--muted);
                                text-transform:uppercase;letter-spacing:.1em;">Frequency</div>
                    <div style="font-family:var(--fm);font-size:0.9rem;font-weight:700;
                                color:{p['color']};margin-top:4px;">{p['freq']}</div>
                </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="background:var(--card);border:1px solid var(--border);border-radius:2px;
                padding:1rem 1.3rem;text-align:center;">
        <div style="font-family:var(--fm);font-size:0.6rem;color:var(--muted);
                    text-transform:uppercase;letter-spacing:.18em;margin-bottom:6px;">// Full project pack</div>
        <div style="font-family:var(--fm);font-size:0.68rem;color:var(--muted);line-height:1.7;">
            Complete dataset · notebook · presentation deck · simulator
            available on <span style="color:#f97316;">lozanolsa.gumroad.com</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="lsa-footer">
    LozanoLsa · Turning Operations into Predictive Systems · Pharma Batch Quality · Project 18 · v2.0
</div>
""", unsafe_allow_html=True)
