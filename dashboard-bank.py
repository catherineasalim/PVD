"""
Bank Marketing Campaign — Dashboard Presentasi Visualisasi Data
Dataset : UCI Bank Marketing (bank-full.csv), Mei 2008 – Nov 2010
Materi  : Lecture 03 (Amounts), Lecture 04 (Distributions), Lecture 05 (Proportions)

Jalankan:  streamlit run dashboard_bank.py
(bank-full.csv diletakkan satu folder dengan file ini)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

st.set_page_config(
    page_title="Bank Marketing · Dari Telepon Massal ke Telepon yang Tepat",
    page_icon="📞",
    layout="wide",
)

# =============================================================================
# 1. DESIGN TOKENS
# =============================================================================
FONT = "Plus Jakarta Sans, system-ui, -apple-system, Segoe UI, sans-serif"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e9e8e2", "#c3c2b7", "#ffffff"

YES, YES_DARK, YES_LIGHT = "#2a78d6", "#1c5cab", "#cde2fb"  # subscribe
NO, NO_DARK = "#c3c2b7", "#898781"  # tidak subscribe
ACCENT = "#eb6834"  # pembanding (punya pinjaman)
RED = "#e34948"

# Diverging: merah (di bawah rata-rata) — abu (rata-rata) — biru (di atas rata-rata)
DIVERGING = [
    [0.0, "#c43c3b"],
    [0.25, "#ee9d97"],
    [0.5, "#f0efec"],
    [0.75, "#86b6ef"],
    [1.0, "#1c5cab"],
]
TIER_COLOR = {"Prioritas": "#104281", "Potensial": "#5598e7", "Umum": "#c3c2b7"}

PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False}

MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]

JOB_ID = {
    "admin.": "Admin",
    "blue-collar": "Kerah biru",
    "entrepreneur": "Wirausaha",
    "housemaid": "Asisten RT",
    "management": "Manajemen",
    "retired": "Pensiunan",
    "self-employed": "Pekerja lepas",
    "services": "Jasa",
    "student": "Pelajar/Mhs",
    "technician": "Teknisi",
    "unemployed": "Tidak bekerja",
    "unknown": "Tidak diketahui",
}
EDU_ID = {"primary": "Dasar", "secondary": "Menengah", "tertiary": "Tinggi", "unknown": "Tidak diketahui"}
POUT_ID = {"success": "Sukses", "other": "Lainnya", "failure": "Gagal", "unknown": "Belum pernah"}
CONTACT_ID = {"cellular": "Seluler", "telephone": "Telepon rumah", "unknown": "Tidak tercatat"}
AGE_ORDER = ["18–25", "26–30", "31–40", "41–50", "51–60", "60+"]
ATTEMPT_ORDER = ["Kontak ke-1", "Kontak ke-2–3", "Kontak ke-4+"]


# =============================================================================
# 2. CSS
# =============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, button, input, label {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
}
.stApp { background: #f6f5f1; }
.block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1320px; }
h1, h2, h3, h4 { font-family: 'Plus Jakarta Sans', sans-serif !important; letter-spacing: -0.02em; color: #0b0b0b; }

/* ---------- HERO ---------- */
.hero {
    background: radial-gradient(1200px 400px at 85% -20%, #2a78d6 0%, rgba(42,120,214,0) 60%),
                linear-gradient(135deg, #0d366b 0%, #104281 55%, #184f95 100%);
    border-radius: 24px; padding: 2.4rem 2.6rem 2.2rem; color: #fff; margin-bottom: 1.4rem;
    box-shadow: 0 20px 40px -24px rgba(13,54,107,.55);
}
.hero .eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; color: #9ec5f4; }
.hero h1 { color: #fff !important; font-size: 2.55rem; font-weight: 800; line-height: 1.12; margin: .45rem 0 .7rem; }
.hero p { color: #dbe8f8; font-size: 1.02rem; max-width: 820px; margin: 0; line-height: 1.6; }

/* ---------- KPI ---------- */
.kpi-row { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 14px; margin: 1.6rem 0 0; }
.kpi { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.16); border-radius: 16px; padding: 1rem 1.15rem; }
.kpi .lbl { font-size: .78rem; color: #b7d3f6; font-weight: 600; }
.kpi .val { font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.15; margin-top: .25rem; }
.kpi .sub { font-size: .78rem; color: #cde2fb; margin-top: .2rem; }
@media (max-width: 900px) { .kpi-row { grid-template-columns: repeat(2, minmax(0,1fr)); } .hero h1 { font-size: 1.9rem; } }

/* ---------- TABS ---------- */
[data-testid="stTabs"] [role="tablist"] { gap: 6px; background: #ebe9e2; padding: 6px; border-radius: 14px; border: none; }
[data-testid="stTabs"] [role="tab"] { min-height: 44px; padding: 0 18px; border-radius: 10px; background: transparent; border: none; }
[data-testid="stTabs"] [role="tab"] p { font-weight: 600; color: #52514e; font-size: .95rem; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] p { color: #104281; font-weight: 800; }
[data-testid="stTabs"] [role="tablist"] > div:not([role="tab"]) { display: none; }
[data-testid="stTabs"] [role="tab"]::after, [data-testid="stTabs"] [role="tab"]::before { display: none !important; }
[data-testid="stTabPanel"] { padding-top: 1.2rem; }

/* ---------- CARDS (st.container(border=True)) ---------- */
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .card-head),
[data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .action) {
    background: #ffffff; border-radius: 20px !important; border: 1px solid rgba(11,11,11,.07) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.03), 0 8px 24px -16px rgba(0,0,0,.12); padding: 1.4rem 1.5rem 1.2rem !important;
}
.card-head { margin-bottom: .2rem; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: .55rem; }
.badge { display: inline-block; font-size: .7rem; font-weight: 700; letter-spacing: .04em; padding: 3px 10px; border-radius: 999px; }
.b-l3 { background: #fdeee6; color: #a8431a; }
.b-l4 { background: #e3f5ec; color: #0d6e4a; }
.b-l5 { background: #ece9fb; color: #4a3aa7; }
.b-ch { background: #f0efec; color: #52514e; }
.card-title { font-size: 1.32rem; font-weight: 800; color: #0b0b0b; line-height: 1.3; letter-spacing: -.015em; }
.card-sub { font-size: .9rem; color: #6b6a65; margin-top: .3rem; line-height: 1.5; }

/* ---------- INSIGHT ---------- */
.insight { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: .4rem; }
.ins-box { border-radius: 14px; padding: .9rem 1.05rem; font-size: .9rem; line-height: 1.55; color: #2b2a27; }
.ins-box ul { margin: 0; padding-left: 1.1rem; }
.ins-box li { margin-bottom: .25rem; }
.ins-box b.t { display: block; font-size: .7rem; letter-spacing: .12em; text-transform: uppercase; margin-bottom: .3rem; }
.ins-find { background: #f4f8fe; border-left: 4px solid #2a78d6; }
.ins-find b.t { color: #1c5cab; }
.ins-act { background: #fff6ef; border-left: 4px solid #eb6834; }
.ins-act b.t { color: #a8431a; }
@media (max-width: 900px) { .insight { grid-template-columns: 1fr; } }

/* ---------- MINI STATS ---------- */
.stat { background: #f8f8f6; border-radius: 14px; padding: .85rem 1rem; margin-bottom: 10px; border: 1px solid #efeee9; }
.stat .v { font-size: 1.65rem; font-weight: 800; color: #104281; line-height: 1.1; }
.stat .v.warn { color: #c43c3b; }
.stat .l { font-size: .82rem; color: #52514e; margin-top: .2rem; line-height: 1.4; }

/* ---------- CHAPTER INTRO ---------- */
.chapter { margin: .6rem 0 1rem; }
.chapter .num { font-size: .75rem; font-weight: 800; letter-spacing: .14em; color: #2a78d6; text-transform: uppercase; }
.chapter h2 { font-size: 1.75rem; font-weight: 800; margin: .2rem 0 .3rem; }
.chapter p { color: #52514e; max-width: 900px; margin: 0; }

/* ---------- STORY CARDS ---------- */
.story { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; }
.story-card { background: #fff; border-radius: 18px; padding: 1.2rem 1.25rem; border: 1px solid rgba(11,11,11,.07); }
.story-card .n { width: 30px; height: 30px; border-radius: 9px; background: #104281; color: #fff; font-weight: 800; display: grid; place-items: center; font-size: .9rem; }
.story-card h4 { margin: .7rem 0 .35rem; font-size: 1.05rem; font-weight: 800; }
.story-card p { font-size: .88rem; color: #52514e; margin: 0; line-height: 1.55; }
.story-card .big { font-size: 1.5rem; font-weight: 800; color: #104281; margin-top: .6rem; }
@media (max-width: 900px) { .story { grid-template-columns: 1fr; } }

.tier-card { border-radius: 16px; padding: 1rem 1.1rem; color: #fff; height: 100%; }
.tier-card h4 { margin: 0 0 .35rem; font-size: 1.05rem; }
.tier-card .r { font-size: 1.8rem; font-weight: 800; }
.tier-card p { font-size: .82rem; margin: .35rem 0 0; opacity: .92; line-height: 1.5; }

[data-testid="stMetricValue"] { font-weight: 800; color: #104281; }
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# 3. DATA
# =============================================================================
@st.cache_data
def load_data():
    path = Path(__file__).parent / "bank-full.csv"
    if not path.exists():
        path = Path("bank-full.csv")
    df = pd.read_csv(path, sep=";")
    df["subscribe"] = (df["y"] == "yes").astype(int)

    # Data UCI diurutkan kronologis (Mei 2008 – Nov 2010) tetapi tanpa kolom tahun.
    # Tahun direkonstruksi: setiap kali urutan bulan "mundur", berarti masuk tahun baru.
    m_idx = df["month"].map({m: i for i, m in enumerate(MONTHS)})
    df["m_idx"] = m_idx
    df["year"] = 2008 + (m_idx.diff() < 0).cumsum()

    df["age_band"] = pd.cut(df["age"], [17, 25, 30, 40, 50, 60, 120], labels=AGE_ORDER)
    df["attempt"] = pd.cut(df["campaign"], [0, 1, 3, 1000], labels=ATTEMPT_ORDER)
    df["loan_status"] = np.where(
        (df["housing"] == "no") & (df["loan"] == "no"), "Tanpa pinjaman", "Punya pinjaman"
    )
    df["job_id"] = df["job"].map(JOB_ID)
    df["edu_id"] = df["education"].map(EDU_ID)

    q = pd.qcut(df["balance"], 5, retbins=True)[1]
    bal_labels = [
        f"Q1 · ≤ €{q[1]:,.0f}",
        f"Q2 · €{q[1] + 1:,.0f}–{q[2]:,.0f}",
        f"Q3 · €{q[2] + 1:,.0f}–{q[3]:,.0f}",
        f"Q4 · €{q[3] + 1:,.0f}–{q[4]:,.0f}",
        f"Q5 · > €{q[4]:,.0f}",
    ]
    bal_labels = [b.replace(",", ".") for b in bal_labels]
    df["bal_q"] = pd.qcut(df["balance"], 5, labels=bal_labels)

    # Segmentasi prioritas (dibangun dari temuan Bab 1–3)
    hot = (
        (df["poutcome"] == "success")
        | (df["age"] > 60)
        | (df["age"] <= 25)
        | df["job"].isin(["student", "retired"])
    )
    warm = ~hot & (df["housing"] == "no") & (df["loan"] == "no") & (df["contact"] != "unknown")
    df["tier"] = np.select([hot, warm], ["Prioritas", "Potensial"], "Umum")
    return df, bal_labels


df, BAL_LABELS = load_data()
N = len(df)
BASE = df["subscribe"].mean()
N_YES = int(df["subscribe"].sum())
JOB_ORDER = df.groupby("job_id")["subscribe"].mean().sort_values(ascending=False).index.tolist()


# =============================================================================
# 4. HELPERS
# =============================================================================
def nf(x):
    """Angka format Indonesia: 45.211"""
    return f"{x:,.0f}".replace(",", ".")


def pf(x, d=1):
    """Persen format Indonesia: 11,7%"""
    return f"{x * 100:.{d}f}".replace(".", ",") + "%"


def style(fig, height=440, legend=True):
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family=FONT, size=13, color=INK2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=36, b=8),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=12, color=INK)),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, font=dict(size=12)),
        bargap=0.22,
        separators=",.",  # format angka Indonesia: desimal koma, ribuan titik
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED, size=11),
                     title_font=dict(size=12, color=INK2))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED, size=11),
                     title_font=dict(size=12, color=INK2))
    return fig


def show(fig):
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


def card_head(title):
    st.markdown(f'<div class="card-head"><div class="card-title">{title}</div></div>', unsafe_allow_html=True)


def insight(points, act):
    items = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(
        f'<div class="insight"><div class="ins-box ins-find"><b class="t">Temuan</b><ul>{items}</ul></div>'
        f'<div class="ins-box ins-act"><b class="t">Business insight</b>{act}</div></div>',
        unsafe_allow_html=True,
    )


def lift(rate):
    return f"{rate / BASE:.1f}×".replace(".", ",")


def stat(value, label, warn=False):
    st.markdown(
        f'<div class="stat"><div class="v{" warn" if warn else ""}">{value}</div><div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )


def chapter(title):
    st.markdown(f'<div class="chapter"><h2>{title}</h2></div>', unsafe_allow_html=True)


def slog(x):
    """Signed log10 — untuk data sangat miring yang juga punya nilai negatif (balance)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log10(1 + np.abs(x))


# =============================================================================
# 5. PRE-COMPUTED NUMBERS (dipakai di narasi supaya selalu konsisten dengan data)
# =============================================================================
by_year = df.groupby("year")["subscribe"].agg(rate="mean", n="size")
r08, r10 = by_year.loc[2008, "rate"], by_year.loc[2010, "rate"]
n08, n10 = by_year.loc[2008, "n"], by_year.loc[2010, "n"]

calls_total = df["campaign"].sum()
calls_extra = (df["campaign"] - 3).clip(lower=0).sum()
yes_le3 = (df.loc[df["subscribe"] == 1, "campaign"] <= 3).mean()
rate_4plus = df.loc[df["campaign"] > 3, "subscribe"].mean()

tier = df.groupby("tier")["subscribe"].agg(n="size", yes="sum", rate="mean").reindex(["Prioritas", "Potensial", "Umum"])
tier["share_n"] = tier["n"] / N
tier["share_yes"] = tier["yes"] / N_YES


# =============================================================================
# 6. HERO
# =============================================================================
st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">Deposito Berjangka · Bank Portugal 2008–2010</div>
  <h1>Analisis Kampanye Telemarketing</h1>
  <div class="kpi-row">
    <div class="kpi"><div class="lbl">Nasabah dihubungi</div><div class="val">{nf(N)}</div><div class="sub">{nf(calls_total)} total panggilan</div></div>
    <div class="kpi"><div class="lbl">Subscription rate</div><div class="val">{pf(BASE)}</div><div class="sub">{nf(N_YES)} nasabah subscribe</div></div>
    <div class="kpi"><div class="lbl">Conversion 2008 → 2010</div><div class="val">{pf(r08)} → {pf(r10, 0)}</div><div class="sub">volume turun {nf(n08)} → {nf(n10)}</div></div>
    <div class="kpi"><div class="lbl">Segmen “Prioritas”</div><div class="val">{pf(tier.loc['Prioritas', 'rate'], 0)}</div><div class="sub">{lift(tier.loc['Prioritas', 'rate'])} rata-rata · {pf(tier.loc['Prioritas', 'share_n'], 0)} nasabah</div></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    ["🧭  Ringkasan", "①  Cara menelepon", "②  Siapa targetnya", "③  Sinyal terkuat", "🎯  Rekomendasi"]
)

# =============================================================================
# TAB 0 — RINGKASAN
# =============================================================================
with tabs[0]:
    chapter("Strategi Bisnis")
    pri = tier.loc["Prioritas"]
    st.markdown(
        f"""
<div class="story">
  <div class="story-card"><div class="n">1</div><h4>Cara menelepon</h4>
    <p>Bank berubah dari panggilan massal (2008) ke panggilan tertarget (2010). Kontak ke-4 dan seterusnya
    menghabiskan hampir sepertiga upaya dengan conversion di bawah rata-rata.</p>
    <div class="big">{pf(calls_extra / calls_total, 0)} panggilan terbuang</div></div>
  <div class="story-card"><div class="n">2</div><h4>Siapa targetnya</h4>
    <p>Volume telepon terkonsentrasi di usia 31–50 &amp; kerah biru, padahal conversion tertinggi ada di
    nasabah muda, lansia, pelajar, pensiunan, dan nasabah tanpa pinjaman.</p>
    <div class="big">Usia 60+: {pf(df.loc[df.age > 60, 'subscribe'].mean(), 0)} conversion</div></div>
  <div class="story-card"><div class="n">3</div><h4>Sinyal terkuat</h4>
    <p>Nasabah yang pernah sukses di kampanye lalu adalah “warm lead” terbaik. Saldo tinggi dan bebas
    pinjaman memperkuat peluang di semua kelompok usia.</p>
    <div class="big">Riwayat sukses: {pf(df.loc[df.poutcome == 'success', 'subscribe'].mean(), 0)}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.expander("🗂️ Data & catatan metodologi"):
        st.markdown(
            f"""
- **Dataset**: UCI Bank Marketing `bank-full.csv`, {nf(N)} baris × 17 kolom, target `y` = subscribe deposito berjangka.
- **Tahun direkonstruksi**: data diurutkan kronologis (Mei 2008 – Nov 2010) tapi tidak punya kolom tahun.
  Tahun diturunkan dari urutan baris: setiap kali bulan “mundur” (mis. Des → Jan) dihitung sebagai tahun baru.
- **`campaign`** = jumlah kontak ke nasabah selama kampanye ini (termasuk kontak terakhir).
  Total panggilan = jumlah seluruh `campaign` = {nf(calls_total)}.
- **`duration`** baru diketahui setelah telepon selesai, jadi hanya dipakai untuk evaluasi kualitas panggilan,
  bukan untuk memilih nasabah (menghindari *data leakage*).
- Segmen dengan sampel kecil diberi catatan jumlah `n` di tooltip agar tidak disalahartikan.
"""
        )
        st.dataframe(
            df.drop(columns=["subscribe", "m_idx", "age_band", "attempt", "loan_status", "job_id", "edu_id", "bal_q", "tier"]).head(200),
            use_container_width=True, height=280,
        )


# =============================================================================
# TAB 1 — CARA MENELEPON
# =============================================================================
with tabs[1]:
    chapter("Cara menelepon: lebih sedikit, lebih tepat")

    # ---------- 1A. Timeline: stacked bars (amounts) + 100% stacked bars (proportions) ----------
    with st.container(border=True):
        card_head("Volume Panggilan vs Subscription Rate per Bulan (2008–2010)")
        tl = (
            df.groupby(["year", "m_idx"])["subscribe"].agg(n="size", yes="sum").reset_index()
            .sort_values(["year", "m_idx"]).reset_index(drop=True)
        )
        tl["no"] = tl["n"] - tl["yes"]
        tl["rate"] = tl["yes"] / tl["n"]
        tl["label"] = [f"{BULAN[m]}<br>{str(y)[2:]}" for y, m in zip(tl["year"], tl["m_idx"])]
        tl["full"] = [f"{BULAN[m]} {y}" for y, m in zip(tl["year"], tl["m_idx"])]
        x = list(range(len(tl)))

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.07)
        cd = np.stack([tl["full"], tl["n"], tl["rate"] * 100], axis=1)
        fig.add_bar(x=x, y=tl["no"], name="Tidak subscribe", marker_color=NO, row=1, col=1,
                    marker_line=dict(color="white", width=0.5), customdata=cd,
                    hovertemplate="<b>%{customdata[0]}</b><br>Tidak subscribe: %{y:,}<extra></extra>")
        fig.add_bar(x=x, y=tl["yes"], name="Subscribe", marker_color=YES, row=1, col=1,
                    marker_line=dict(color="white", width=0.5), customdata=cd,
                    hovertemplate="<b>%{customdata[0]}</b><br>Subscribe: %{y:,}<br>Total: %{customdata[1]:,}<extra></extra>")
        fig.add_bar(x=x, y=tl["rate"] * 100, marker_color=YES, showlegend=False, row=2, col=1,
                    marker_line=dict(color="white", width=0.5), customdata=cd,
                    hovertemplate="<b>%{customdata[0]}</b><br>Subscription rate: %{y:.1f}%<br>n = %{customdata[1]:,}<extra></extra>")
        fig.add_bar(x=x, y=(1 - tl["rate"]) * 100, marker_color=NO, showlegend=False, row=2, col=1,
                    marker_line=dict(color="white", width=0.5), hoverinfo="skip")
        fig.add_hline(y=BASE * 100, line_dash="dot", line_color=INK, line_width=1, row=2, col=1,
                      annotation_text=f"rata-rata {pf(BASE)}", annotation_position="top left",
                      annotation_font=dict(size=11, color=INK))

        # pemisah tahun + label era
        bounds = tl.groupby("year").apply(lambda g: (g.index.min(), g.index.max()), include_groups=False)
        eras = {2008: "2008 · Telepon massal", 2009: "2009 · Transisi", 2010: "2010 · Tertarget"}
        for yr, (a, b) in bounds.items():
            if a > 0:
                for r in (1, 2):
                    fig.add_vline(x=a - 0.5, line_color=INK2, line_width=1, line_dash="dash", row=r, col=1)
            yr_rate = by_year.loc[yr, "rate"]
            fig.add_annotation(x=(a + b) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                               text=f"<b>{eras[yr]}</b>  ·  conversion {pf(yr_rate)}",
                               font=dict(size=12, color=INK), bgcolor="#f0efec", borderpad=5)
        fig.update_layout(barmode="stack")
        style(fig, 560)
        fig.update_layout(legend=dict(y=1.1), margin=dict(t=70))
        fig.update_xaxes(tickvals=x, ticktext=tl["label"], showgrid=False, tickfont=dict(size=10))
        fig.update_yaxes(title_text="Jumlah nasabah", row=1, col=1, tickformat=",")
        fig.update_yaxes(title_text="Proporsi (%)", row=2, col=1, range=[0, 100], ticksuffix="%", showgrid=False)
        show(fig)
        may08 = tl[(tl.year == 2008) & (tl.m_idx == 4)].iloc[0]
        insight(
            [
                f"Mei 2008: <b>{nf(may08['n'])}</b> panggilan, conversion hanya <b>{pf(may08['rate'])}</b>.",
                f"Volume per tahun turun {nf(n08)} → {nf(n10)}, conversion naik {pf(r08)} → {pf(r10)}.",
                "Sejak pertengahan 2009 conversion stabil di 40–55% tiap bulan, bukan efek musiman.",
            ],
            "Hasil datang dari <b>memilih nasabah yang tepat</b>, bukan dari menambah volume telepon.",
        )

    # ---------- 1B. ECDF ----------
    with st.container(border=True):
        card_head("Distribusi Kumulatif Jumlah Kontak: Subscribe vs Tidak Subscribe")
        c1, c2 = st.columns([2.3, 1])
        with c1:
            fig = go.Figure()
            for key, color, name in [("no", NO_DARK, "Tidak subscribe"), ("yes", YES, "Subscribe")]:
                v = np.sort(df.loc[df.y == key, "campaign"].values)
                xs, cnt = np.unique(v, return_counts=True)
                ys = np.cumsum(cnt) / len(v)
                xs = np.r_[xs, 63]
                ys = np.r_[ys, 1.0]
                fig.add_scatter(x=xs, y=ys * 100, mode="lines", line=dict(color=color, width=3, shape="hv"),
                                name=name, hovertemplate="≤ %{x} kontak: %{y:.1f}%<extra>" + name + "</extra>")
            fig.add_vrect(x0=3.5, x1=63, fillcolor=RED, opacity=0.06, line_width=0)
            fig.add_vline(x=3.5, line_dash="dash", line_color=RED, line_width=1.5)
            no_le3 = (df.loc[df.y == "no", "campaign"] <= 3).mean()
            fig.add_annotation(x=np.log10(3), y=yes_le3 * 100, ax=-90, ay=-40, text=f"<b>{pf(yes_le3)}</b> subscriber",
                               font=dict(color=YES_DARK, size=12), arrowcolor=YES, arrowwidth=1.5)
            fig.add_annotation(x=np.log10(3), y=no_le3 * 100, ax=-70, ay=45, text=f"{pf(no_le3)} non-subscriber",
                               font=dict(color=INK2, size=12), arrowcolor=NO_DARK, arrowwidth=1.5)
            fig.add_annotation(x=np.log10(12), y=40, showarrow=False, align="left",
                               text="<b>Zona kontak ke-4+</b><br>conversion hanya " + pf(rate_4plus),
                               font=dict(color="#a32e2d", size=12))
            style(fig, 420)
            fig.update_xaxes(type="log", tickvals=[1, 2, 3, 4, 5, 10, 20, 40, 63], title="Jumlah kontak di kampanye ini (skala log)")
            fig.update_yaxes(title="Proporsi kumulatif nasabah", ticksuffix="%", range=[0, 102])
            show(fig)
        with c2:
            st.write("")
            stat(pf(calls_extra / calls_total), f"dari {nf(calls_total)} panggilan adalah kontak ke-4 dst. ({nf(calls_extra)} panggilan)", warn=True)
            stat(pf(rate_4plus), f"conversion nasabah yang dikontak ≥ 4 kali, di bawah rata-rata {pf(BASE)}", warn=True)
            stat(pf(1 - yes_le3), "subscriber yang “hilang” jika kontak dibatasi maksimal 3 kali")
        insight(
            [
                f"<b>{pf(yes_le3, 0)}</b> subscriber sudah didapat dalam 1–3 kontak pertama.",
                f"Kontak ke-4 dst. = <b>{pf(calls_extra / calls_total, 0)}</b> total panggilan, conversion hanya {pf(rate_4plus)}.",
            ],
            f"Batasi maksimal <b>3 kontak</b> per nasabah: hemat ±{pf(calls_extra / calls_total, 0)} panggilan, hanya kehilangan {pf(1 - yes_le3, 0)} subscriber.",
        )

    # ---------- 1C. Split violin + sina ----------
    with st.container(border=True):
        card_head("Durasi Panggilan per Urutan Kontak: Subscribe vs Tidak Subscribe")
        dd = df[df["duration"] > 0].copy()
        dd["ld"] = np.log10(dd["duration"])
        rng = np.random.default_rng(7)
        fig = go.Figure()
        for side, key, color, dot, name in [
            ("negative", "no", NO, NO_DARK, "Tidak subscribe"),
            ("positive", "yes", YES, YES_DARK, "Subscribe"),
        ]:
            sub = dd[dd["y"] == key]
            xs = sub["attempt"].map({a: i for i, a in enumerate(ATTEMPT_ORDER)}).astype(float)
            fig.add_violin(
                x=xs, y=sub["ld"], side=side, name=name, legendgroup=key, scalegroup=key,
                line=dict(color=dot, width=1.5), fillcolor=color, opacity=0.55, width=0.85,
                points=False, meanline_visible=False, box_visible=True,
                box=dict(width=0.12, fillcolor="white", line=dict(color=INK, width=1)),
                spanmode="hard", bandwidth=0.06, hoverinfo="skip",
            )
            # sina points
            px_all, py_all, pd_all = [], [], []
            for i, att in enumerate(ATTEMPT_ORDER):
                g = sub.loc[sub["attempt"] == att, "ld"].values
                if len(g) == 0:
                    continue
                s = rng.choice(g, size=min(450, len(g)), replace=False)
                kde = gaussian_kde(g, bw_method=0.15)
                grid = np.linspace(g.min(), g.max(), 200)
                dmax = kde(grid).max()
                w = kde(s) / dmax * 0.40 * rng.uniform(0.05, 1, len(s))
                px_all.append(i + (w if side == "positive" else -w))
                py_all.append(s)
                pd_all.append(10 ** s)
            fig.add_scatter(
                x=np.concatenate(px_all), y=np.concatenate(py_all), mode="markers", legendgroup=key,
                showlegend=False, marker=dict(size=3.2, color=dot, opacity=0.45),
                customdata=np.concatenate(pd_all), hovertemplate=name + ": %{customdata:.0f} detik<extra></extra>",
            )
        med = dd.groupby(["attempt", "y"], observed=True)["duration"].median().unstack()
        for i, att in enumerate(ATTEMPT_ORDER):
            for key, dx, col in [("no", -0.47, INK2), ("yes", 0.47, YES_DARK)]:
                fig.add_annotation(x=i + dx, y=np.log10(med.loc[att, key]), showarrow=False,
                                   text=f"<b>{med.loc[att, key]:.0f} dtk</b>", xanchor="right" if key == "no" else "left",
                                   font=dict(size=11, color=col), bgcolor="rgba(255,255,255,.85)")
        ticks = [5, 15, 30, 60, 120, 300, 600, 1200, 3000]
        tick_txt = ["5 dtk", "15 dtk", "30 dtk", "1 mnt", "2 mnt", "5 mnt", "10 mnt", "20 mnt", "50 mnt"]
        fig.update_layout(violingap=0, violinmode="overlay")
        style(fig, 520)
        fig.update_xaxes(tickvals=[0, 1, 2], ticktext=ATTEMPT_ORDER, range=[-0.75, 2.75], showgrid=False,
                         tickfont=dict(size=13, color=INK))
        fig.update_yaxes(tickvals=np.log10(ticks), ticktext=tick_txt, title="Durasi panggilan (skala log)", range=[0.5, 3.7])
        show(fig)
        dbin = pd.cut(df["duration"], [-1, 120, 600, 1e6], labels=["<2", "2-10", ">10"])
        drate = df.groupby(dbin, observed=True)["subscribe"].mean()
        insight(
            [
                f"Panggilan < 2 menit: conversion <b>{pf(drate['<2'])}</b>; panggilan > 10 menit: <b>{pf(drate['>10'])}</b>.",
                f"Di kontak ke-4+, panggilan gagal makin pendek (median {med.loc['Kontak ke-4+', 'no']:.0f} dtk).",
            ],
            "Jadikan durasi <b>KPI kualitas agen</b>: percakapan < 2 menit = berhenti, jangan telepon ulang.",
        )


# =============================================================================
# TAB 2 — SIAPA TARGETNYA
# =============================================================================
with tabs[2]:
    chapter("Siapa targetnya: telepon kita salah alamat")

    # ---------- 2A. Treemap ----------
    with st.container(border=True):
        card_head("Komposisi Nasabah per Pekerjaan & Subscription Rate")
        lvl_name = st.radio(
            "Pecah tiap pekerjaan berdasarkan:", ["Kelompok usia", "Status pinjaman", "Pendidikan"],
            horizontal=True, key="tm_lvl",
        )
        lvl = {"Kelompok usia": "age_band", "Status pinjaman": "loan_status", "Pendidikan": "edu_id"}[lvl_name]
        agg = df.groupby(["job_id", lvl], observed=True)["subscribe"].agg(n="size", rate="mean").reset_index()
        agg["rate_pct"] = agg["rate"] * 100
        jobs = agg.groupby("job_id").apply(
            lambda g: pd.Series({"n": g["n"].sum(), "rate": (g["rate"] * g["n"]).sum() / g["n"].sum()}),
            include_groups=False,
        ).reset_index()
        ids = ["root"] + [f"j/{j}" for j in jobs["job_id"]] + [f"j/{j}/{v}" for j, v in zip(agg["job_id"], agg[lvl])]
        labels = ["Semua nasabah"] + list(jobs["job_id"]) + [str(r[lvl]) for _, r in agg.iterrows()]
        parents = [""] + ["root"] * len(jobs) + [f"j/{j}" for j in agg["job_id"]]
        values = [N] + list(jobs["n"]) + list(agg["n"])
        rates = [BASE * 100] + list(jobs["rate"] * 100) + list(agg["rate_pct"])
        fig = go.Figure(go.Treemap(
            ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
            customdata=rates, sort=True,
            marker=dict(colors=rates, colorscale=DIVERGING, cmin=0, cmid=BASE * 100, cmax=BASE * 200,
                        line=dict(color="white", width=2), cornerradius=6, showscale=True,
                        colorbar=dict(title=dict(text="Rate", font=dict(size=11)), thickness=12, len=0.8)),
            texttemplate="<b>%{label}</b><br>%{customdata:.1f}%<br>%{value:,} nasabah",
            hovertemplate="<b>%{label}</b><br>Nasabah: %{value:,}<br>Subscription rate: %{customdata:.1f}%<extra></extra>",
            tiling=dict(pad=3), root_color="#f6f5f1", textfont=dict(family=FONT, size=13),
        ))
        b = BASE * 100
        fig.update_traces(marker_colorbar=dict(
            tickvals=[0, b / 2, b, b * 1.5, b * 2],
            ticktext=["0%", f"{b / 2:.0f}%", f"{b:.1f}% (rata-rata)".replace(".", ","), f"{b * 1.5:.0f}%", f"≥ {b * 2:.0f}%"],
        ))
        style(fig, 600, legend=False)
        fig.update_layout(margin=dict(t=10, l=0, r=0, b=0))
        show(fig)
        share_big = df["job"].isin(["blue-collar", "management", "technician"]).mean()
        insight(
            [
                f"Kerah biru, manajemen & teknisi = <b>{pf(share_big, 0)}</b> volume telepon; kerah biru conversion hanya {pf(df.loc[df.job == 'blue-collar', 'subscribe'].mean())}.",
                f"Pelajar ({pf(df.loc[df.job == 'student', 'subscribe'].mean(), 0)}) dan pensiunan ({pf(df.loc[df.job == 'retired', 'subscribe'].mean(), 0)}) conversion tertinggi, tapi volumenya kecil.",
                "Di setiap pekerjaan, nasabah tanpa pinjaman conversion-nya lebih tinggi.",
            ],
            "Alokasikan kuota telepon berdasarkan <b>peluang</b>, bukan jumlah nasabah.",
        )

    # ---------- 2B. Age: parts of total + stacked density ----------
    with st.container(border=True):
        card_head("Distribusi Usia Nasabah & Proporsi Subscribe per Usia")
        age_ct = df.groupby(["age", "y"]).size().unstack(fill_value=0).reindex(range(18, 89), fill_value=0)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.08)
        fig.add_bar(x=age_ct.index, y=age_ct.sum(axis=1), marker_color="#dddcd5", name="Semua nasabah ditelepon",
                    hovertemplate="Usia %{x}: %{y:,} nasabah<extra></extra>", row=1, col=1)
        fig.add_bar(x=age_ct.index, y=age_ct["yes"], marker_color=YES, name="Subscriber (bagian dari total)",
                    hovertemplate="Usia %{x}: %{y:,} subscriber<extra></extra>", row=1, col=1)

        grid = np.linspace(18, 88, 400)
        ay, an = df.loc[df.y == "yes", "age"].values, df.loc[df.y == "no", "age"].values
        fy = gaussian_kde(ay, bw_method=0.08)(grid) * len(ay)
        fn = gaussian_kde(an, bw_method=0.08)(grid) * len(an)
        share = fy / (fy + fn) * 100
        fig.add_scatter(x=grid, y=share, fill="tozeroy", mode="lines", line=dict(color=YES_DARK, width=1.5),
                        fillcolor=YES, showlegend=False, hovertemplate="Usia %{x:.0f}: %{y:.1f}% subscribe<extra></extra>",
                        row=2, col=1)
        fig.add_scatter(x=grid, y=np.full_like(grid, 100), fill="tonexty", mode="lines", line=dict(width=0),
                        fillcolor=NO, showlegend=False, hoverinfo="skip", row=2, col=1)
        fig.add_hline(y=BASE * 100, line_dash="dot", line_color=INK, line_width=1, row=2, col=1,
                      annotation_text=f"rata-rata {pf(BASE)}", annotation_position="top left",
                      annotation_font=dict(size=11, color=INK))
        young, old = df.age <= 25, df.age > 60
        for x0, x1, lab in [(17.5, 25.5, f"≤25 th · {pf(young.mean(), 1)} panggilan<br><b>{pf(df.loc[young, 'subscribe'].mean(), 0)} conversion</b>"),
                            (60.5, 88.5, f">60 th · {pf(old.mean(), 1)} panggilan · <b>{pf(df.loc[old, 'subscribe'].mean(), 0)} conversion</b>")]:
            fig.add_vrect(x0=x0, x1=x1, fillcolor="#eb6834", opacity=0.07, line_width=0)
            fig.add_annotation(x=(x0 + x1) / 2, y=1, yref="paper", yanchor="bottom", text=lab, showarrow=False,
                               font=dict(size=11, color="#a8431a"))
        mid = (df.age > 30) & (df.age <= 50)
        fig.add_annotation(x=40, y=1, yref="paper", yanchor="bottom", showarrow=False,
                           text=f"31–50 th · <b>{pf(mid.mean(), 0)}</b> panggilan · {pf(df.loc[mid, 'subscribe'].mean(), 0)} conversion",
                           font=dict(size=11, color=INK2))
        fig.update_layout(barmode="overlay", bargap=0.05)
        style(fig, 560)
        fig.update_layout(margin=dict(t=80), legend=dict(y=-0.12, yanchor="top"))
        fig.update_xaxes(title_text="Usia (tahun)", row=2, col=1, dtick=5, range=[17.5, 88.5])
        fig.update_yaxes(title_text="Jumlah nasabah", row=1, col=1)
        fig.update_yaxes(title_text="Proporsi", row=2, col=1, range=[0, 100], ticksuffix="%", showgrid=False)
        show(fig)
        insight(
            [
                f"Usia 31–50 = <b>{pf(mid.mean(), 0)}</b> panggilan, tapi conversion di bawah rata-rata.",
                f"Usia >60: conversion <b>{pf(df.loc[old, 'subscribe'].mean(), 0)}</b>; usia ≤25: <b>{pf(df.loc[young, 'subscribe'].mean(), 0)}</b>.",
            ],
            "Buat <b>kampanye khusus</b> untuk usia ≤25 (tabungan pertama) dan >60 (dana pensiun aman).",
        )

    # ---------- 2C. Dumbbell dot plot ----------
    _g = df.groupby(["job_id", "loan_status"])["subscribe"].mean().unstack()
    _r = _g["Tanpa pinjaman"] / _g["Punya pinjaman"]
    DUMB_TITLE = (f"Di setiap pekerjaan, nasabah tanpa pinjaman {_r.min():.1f}–{_r.max():.1f}× lebih mungkin subscribe").replace(".", ",")
    with st.container(border=True):
        card_head("Subscription Rate per Pekerjaan: Tanpa Pinjaman vs Punya Pinjaman")
        g = df.groupby(["job_id", "loan_status"])["subscribe"].agg(["mean", "size"]).unstack()
        g = g.sort_values(("mean", "Tanpa pinjaman"))
        fig = go.Figure()
        for job in g.index:
            a, b2 = g.loc[job, ("mean", "Punya pinjaman")] * 100, g.loc[job, ("mean", "Tanpa pinjaman")] * 100
            fig.add_scatter(x=[a, b2], y=[job, job], mode="lines", line=dict(color="#d8d7d0", width=6),
                            showlegend=False, hoverinfo="skip")
        for ls, color in [("Punya pinjaman", ACCENT), ("Tanpa pinjaman", YES)]:
            fig.add_scatter(
                x=g[("mean", ls)] * 100, y=g.index, mode="markers", name=ls,
                marker=dict(size=15, color=color, line=dict(color="white", width=2)),
                customdata=g[("size", ls)],
                hovertemplate="<b>%{y}</b><br>" + ls + ": %{x:.1f}%<br>n = %{customdata:,}<extra></extra>",
            )
        ratio = g[("mean", "Tanpa pinjaman")] / g[("mean", "Punya pinjaman")]
        fig.add_scatter(x=[g[("mean", "Tanpa pinjaman")].max() * 100 + 4.5] * len(g), y=g.index, mode="text",
                        text=[f"{r:.1f}×".replace(".", ",") for r in ratio], textfont=dict(size=12, color=INK2),
                        showlegend=False, hoverinfo="skip")
        fig.add_vline(x=BASE * 100, line_dash="dot", line_color=INK, line_width=1,
                      annotation_text=f"rata-rata {pf(BASE)}", annotation_position="bottom right",
                      annotation_font=dict(size=11))
        style(fig, 520)
        fig.update_xaxes(title="Subscription rate (%)", ticksuffix="%", range=[0, 42])
        fig.update_yaxes(showgrid=True, tickfont=dict(size=12, color=INK))
        show(fig)
        insight(
            [
                f"Di semua {len(g)} pekerjaan, nasabah tanpa pinjaman conversion-nya lebih tinggi.",
                f"Tertinggi: pelajar tanpa pinjaman <b>{pf(g.loc['Pelajar/Mhs', ('mean', 'Tanpa pinjaman')])}</b>, pensiunan <b>{pf(g.loc['Pensiunan', ('mean', 'Tanpa pinjaman')])}</b>.",
                "Nasabah yang punya pinjaman hampir semuanya di bawah rata-rata.",
            ],
            "Pakai status pinjaman sebagai <b>filter awal</b>; nasabah bercicilan ditawari produk lain.",
        )


# =============================================================================
# TAB 3 — SINYAL TERKUAT
# =============================================================================
with tabs[3]:
    chapter("Sinyal terkuat: riwayat dan kondisi finansial")

    # ---------- 3A. Parallel sets ----------
    with st.container(border=True):
        card_head("Alur Nasabah: Hasil Kampanye Lalu → Saluran Kontak → KPR → Keputusan")
        pc = df.assign(
            p=df["poutcome"].map(POUT_ID), c=df["contact"].map(CONTACT_ID),
            h=np.where(df["housing"] == "yes", "Punya KPR", "Tanpa KPR"),
            yy=np.where(df["y"] == "yes", "Subscribe", "Tidak"),
        ).groupby(["p", "c", "h", "yy"]).size().reset_index(name="n")
        fig = go.Figure(go.Parcats(
            dimensions=[
                dict(values=pc["p"], label="Hasil kampanye lalu", categoryorder="array",
                     categoryarray=["Sukses", "Lainnya", "Gagal", "Belum pernah"]),
                dict(values=pc["c"], label="Saluran kontak", categoryorder="array",
                     categoryarray=["Seluler", "Telepon rumah", "Tidak tercatat"]),
                dict(values=pc["h"], label="KPR (housing loan)", categoryorder="array",
                     categoryarray=["Tanpa KPR", "Punya KPR"]),
                dict(values=pc["yy"], label="Keputusan", categoryorder="array", categoryarray=["Subscribe", "Tidak"]),
            ],
            counts=pc["n"],
            line=dict(color=(pc["yy"] == "Subscribe").astype(int), colorscale=[[0, "#d9d8d1"], [1, YES]], shape="hspline"),
            hoveron="color", hoverinfo="count+probability",
            labelfont=dict(size=14, color=INK, family=FONT), tickfont=dict(size=12, color=INK2, family=FONT),
            arrangement="freeform", bundlecolors=True,
        ))
        style(fig, 520, legend=False)
        fig.update_layout(margin=dict(l=110, r=80, t=40, b=20))
        show(fig)
        ps = df.groupby("poutcome")["subscribe"].mean()
        unk = df.loc[df.contact == "unknown", "subscribe"].mean()
        insight(
            [
                f"Riwayat sukses kampanye lalu: conversion <b>{pf(ps['success'], 0)}</b>, padahal hanya {pf((df.poutcome == 'success').mean())} nasabah.",
                f"{pf((df.poutcome == 'unknown').mean(), 0)} nasabah belum pernah dihubungi, conversion {pf(ps['unknown'])}.",
                f"Saluran kontak tidak tercatat: conversion hanya {pf(unk)}.",
            ],
            "Prioritaskan <b>warm lead</b> (riwayat sukses) dan wajibkan pencatatan nomor seluler.",
        )

    # ---------- 3B. Heatmap ----------
    with st.container(border=True):
        card_head("Subscription Rate per Kelompok Usia × Kuintil Saldo")
        hm_rate = df.pivot_table(index="age_band", columns="bal_q", values="subscribe", aggfunc="mean", observed=False) * 100
        hm_n = df.pivot_table(index="age_band", columns="bal_q", values="subscribe", aggfunc="size", observed=False)
        hm_rate = hm_rate.iloc[::-1]
        hm_n = hm_n.iloc[::-1]
        fig = go.Figure(go.Heatmap(
            z=hm_rate.values, x=[c.replace(" · ", "<br>") for c in hm_rate.columns], y=list(hm_rate.index),
            colorscale=[[0, "#c43c3b"], [0.125, "#ee9d97"], [0.25, "#f0efec"], [0.55, "#86b6ef"], [1, "#104281"]],
            zmin=0, zmax=BASE * 400,  # rata-rata tepat di posisi 0,25 → abu-abu
            text=[[f"{v:.0f}%" for v in row] for row in hm_rate.values], texttemplate="<b>%{text}</b>",
            textfont=dict(size=14, family=FONT), customdata=hm_n.values, xgap=3, ygap=3,
            hovertemplate="Usia %{y}<br>Saldo %{x}<br>Rate: %{z:.1f}%<br>n = %{customdata:,}<extra></extra>",
            colorbar=dict(title=dict(text="Rate", font=dict(size=11)), ticksuffix="%", thickness=12, len=0.85),
        ))
        style(fig, 470, legend=False)
        fig.update_xaxes(side="top", showgrid=False, tickfont=dict(size=11, color=INK2))
        fig.update_yaxes(showgrid=False, title="Kelompok usia", tickfont=dict(size=12, color=INK))
        fig.update_layout(margin=dict(t=60))
        show(fig)
        insight(
            [
                "Semakin tinggi saldo, semakin tinggi conversion di semua kelompok usia.",
                f"Tertinggi: usia 60+ saldo teratas <b>{hm_rate.loc['60+'].iloc[-1]:.0f}%</b>, usia 18–25 saldo teratas {hm_rate.loc['18–25'].iloc[-1]:.0f}%.",
                "Usia 31–50 tetap rendah meski saldonya tinggi.",
            ],
            "Pakai saldo sebagai <b>skor kedua</b> setelah usia.",
        )

    # ---------- 3C. Ridgeline ----------
    with st.container(border=True):
        card_head("Distribusi Saldo per Pekerjaan: Subscribe vs Tidak Subscribe")
        x_lo, x_hi = -2000, 40000
        grid = np.linspace(slog(x_lo), slog(x_hi), 320)
        order = JOB_ORDER[::-1]  # tertinggi di atas
        H = 1.7
        fig = go.Figure()
        meds = []
        for i in reversed(range(len(order))):
            job = order[i]
            sub = df[df.job_id == job]
            for key, line_c, fill_c in [("yes", YES_DARK, "rgba(42,120,214,0.38)"), ("no", "#6f6e69", "rgba(195,194,183,0.25)")]:
                v = slog(sub.loc[sub.y == key, "balance"].clip(x_lo, x_hi))
                d = gaussian_kde(v, bw_method=0.18)(grid)
                d = d / d.max() * H * 0.5
                fig.add_scatter(x=grid, y=np.full_like(grid, i), mode="lines", line=dict(width=0),
                                showlegend=False, hoverinfo="skip")
                fig.add_scatter(x=grid, y=i + d, mode="lines", fill="tonexty", fillcolor=fill_c,
                                line=dict(color=line_c, width=1.4), showlegend=False, hoverinfo="skip")
                m = sub.loc[sub.y == key, "balance"].median()
                meds.append((job, key, m, i))
                fig.add_scatter(x=[slog(m)] * 2, y=[i, i + 0.35], mode="lines",
                                line=dict(color=line_c, width=2.5), showlegend=False, hoverinfo="skip")
        mdf = pd.DataFrame(meds, columns=["job", "y", "m", "i"])
        for key, color, name in [("no", NO_DARK, "Tidak subscribe"), ("yes", YES, "Subscribe")]:
            s = mdf[mdf.y == key]
            fig.add_scatter(x=slog(s["m"]), y=s["i"] + 0.35, mode="markers", name=name,
                            marker=dict(size=7, color=color, symbol="line-ns-open" if False else "circle"),
                            customdata=np.stack([s["job"], s["m"]], axis=1),
                            hovertemplate="<b>%{customdata[0]}</b><br>Median saldo " + name.lower() + ": €%{customdata[1]:,.0f}<extra></extra>")
        tv = [-1000, -100, 0, 100, 1000, 10000, 40000]
        style(fig, 760)
        fig.update_xaxes(tickvals=slog(tv), ticktext=["−€1rb", "−€100", "€0", "€100", "€1rb", "€10rb", "€40rb"],
                         title="Saldo rata-rata tahunan (signed-log)", range=[slog(-1500), slog(x_hi)])
        fig.update_yaxes(tickvals=list(range(len(order))), ticktext=order, showgrid=False,
                         tickfont=dict(size=12, color=INK), range=[-0.3, len(order) - 1 + H * 0.6])
        fig.add_vline(x=0, line_color=AXIS, line_width=1)
        show(fig)
        mp = mdf.pivot(index="job", columns="y", values="m")
        higher = (mp["yes"] > mp["no"]).sum()
        insight(
            [
                f"Median saldo subscriber lebih tinggi di <b>{higher} dari {len(mp)}</b> pekerjaan.",
                "Non-subscriber menumpuk di saldo ≈ €0 dan negatif.",
                "Pengecualian: pelajar, median saldonya sama.",
            ],
            "Kecualikan nasabah bersaldo <b>≤ €0</b> dari kampanye deposito.",
        )


# =============================================================================
# TAB 4 — REKOMENDASI
# =============================================================================
with tabs[4]:
    chapter("Strategi segmentasi: telepon yang tepat, dengan batas yang jelas")
    cols = st.columns(3)
    desc = {
        "Prioritas": "Pernah sukses di kampanye lalu, ATAU usia ≤25 / >60, ATAU pelajar/pensiunan.",
        "Potensial": "Di luar Prioritas, tanpa KPR & pinjaman pribadi, dan punya nomor kontak tercatat.",
        "Umum": "Sisanya: mayoritas usia produktif dengan cicilan aktif.",
    }
    for col, t in zip(cols, ["Prioritas", "Potensial", "Umum"]):
        r = tier.loc[t]
        txt = "#fff" if t != "Umum" else INK
        col.markdown(
            f'<div class="tier-card" style="background:{TIER_COLOR[t]};color:{txt}">'
            f'<h4 style="color:{txt}">{t}</h4><div class="r">{pf(r["rate"])}</div>'
            f'<p>conversion · {lift(r["rate"])} rata-rata<br>{pf(r["share_n"], 0)} nasabah → {pf(r["share_yes"], 0)} subscriber</p>'
            f'<p style="opacity:.85"><i>{desc[t]}</i></p></div>',
            unsafe_allow_html=True,
        )
    st.write("")

    with st.container(border=True):
        top2 = tier.loc[["Prioritas", "Potensial"]].sum()
        card_head("Komposisi Panggilan vs Subscriber per Segmen")
        fig = go.Figure()
        rows = ["Porsi nasabah<br>yang ditelepon", "Porsi subscriber<br>yang didapat"]
        for t in ["Prioritas", "Potensial", "Umum"]:
            vals = [tier.loc[t, "share_n"] * 100, tier.loc[t, "share_yes"] * 100]
            fig.add_bar(y=rows, x=vals, name=t, orientation="h", marker_color=TIER_COLOR[t],
                        marker_line=dict(color="white", width=2),
                        text=[f"<b>{t}</b><br>{v:.0f}%".replace(".", ",") for v in vals], textposition="inside",
                        insidetextanchor="middle", textfont=dict(color="white" if t != "Umum" else INK, size=13),
                        hovertemplate=f"{t}: %{{x:.1f}}%<extra></extra>")
        # garis penghubung antar batang (perubahan porsi)
        cum_n = (tier["share_n"].cumsum() * 100).values[:-1]
        cum_y = (tier["share_yes"].cumsum() * 100).values[:-1]
        for a, b2 in zip(cum_n, cum_y):
            fig.add_shape(type="line", x0=a, x1=b2, y0=0.32, y1=0.68, xref="x", yref="y",
                          line=dict(color=INK2, width=1, dash="dot"))
        fig.update_layout(barmode="stack", bargap=0.36)
        style(fig, 280)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(legend=dict(traceorder="normal"))
        fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False)
        fig.update_yaxes(showgrid=False, tickfont=dict(size=13, color=INK))
        show(fig)

    with st.container(border=True):
        card_head("Simulasi Batas Kontak & Segmen: Panggilan vs Subscriber")
        s1, s2 = st.columns([1, 1.4])
        with s1:
            cap = st.slider("Batas maksimal kontak per nasabah", 1, 10, 3)
            segs = st.multiselect("Segmen yang ditelepon", ["Prioritas", "Potensial", "Umum"],
                                  default=["Prioritas", "Potensial"])
        sel = df[df["tier"].isin(segs)]
        calls_new = sel["campaign"].clip(upper=cap).sum()
        yes_new = ((sel["subscribe"] == 1) & (sel["campaign"] <= cap)).sum()
        rate_new = yes_new / calls_new if calls_new else 0
        rate_old = N_YES / calls_total
        with s2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Panggilan", nf(calls_new), f"-{pf(1 - calls_new / calls_total, 0)}", delta_color="inverse")
            m2.metric("Subscriber", nf(yes_new), f"{pf(yes_new / N_YES, 0)} dari semula", delta_color="off")
            m3.metric("Subscriber / 100 panggilan", f"{rate_new * 100:.1f}".replace(".", ","),
                      f"{rate_new / rate_old:.1f}× semula".replace(".", ",") if rate_old else None)
            st.caption(f"Kondisi awal: {nf(calls_total)} panggilan → {nf(N_YES)} subscriber "
                       f"({f'{rate_old * 100:.1f}'.replace('.', ',')} per 100 panggilan).")

    st.markdown("#### Rencana aksi")
    a1, a2, a3 = st.columns(3)
    with a1, st.container(border=True):
        st.markdown(f"**① Batasi 3 kontak**  \nKontak ke-4+ = {pf(calls_extra / calls_total, 0)} beban kerja dengan "
                    f"conversion {pf(rate_4plus)}. Alihkan kapasitasnya ke nasabah baru.")
    with a2, st.container(border=True):
        st.markdown("**② Telepon segmen Prioritas lebih dulu**  \nWarm lead (riwayat sukses), usia ≤25/>60, pelajar & pensiunan. "
                    "Siapkan skrip berbeda untuk “tabungan pertama” dan “dana pensiun aman”.")
    with a3, st.container(border=True):
        st.markdown("**③ Filter & KPI kualitas**  \nSkip nasabah saldo ≤ €0 dan tanpa nomor seluler. "
                    "Pantau durasi: percakapan < 2 menit = sinyal berhenti, bukan sinyal untuk menelepon ulang.")