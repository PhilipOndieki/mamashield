"""
MamaShield: The Digital Shield
GEGIS Hackathon 2026 - Track 5: Interactive Dashboard

Research Question:
    Can a mother's education and information access protect her child
    even when living in a remote or high-malaria zone of Kenya?
"""

import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MamaShield – The Digital Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────
DATA_DIR = Path("hackathon_data")   # local dev fallback

# Google Drive direct-download URLs
GDRIVE_URLS = {
    "kenya_dhs_dataset_complete.csv": (
        "https://drive.google.com/uc?export=download"
        "&id=1p8P7p3ONBevyoccZVj5GylVp_s3DYI95"
    ),
    "kenya_dhs_covariates.csv": (
        "https://drive.google.com/uc?export=download"
        "&id=1npGQPMoR1yXMWQ_sukVvapuHF64CI0H6"
    ),
    "kenya_dhs_dataset_gps.geojson": (
        "https://drive.google.com/uc?export=download"
        "&id=1ce8PF2n0q5vr2PVKnlJK4RevEf1cAwB7"
    ),
}

# Only 6 of 1,319 columns needed — keeps full dataset at ~11 MB RAM
SURVEY_COLS = ["V001", "V106", "V119", "V120", "V121", "B5"]

EDU_MAP = {
    "No education": 0,
    "Primary":      1,
    "Secondary":    2,
    "Higher":       3,
}

BINARY_MAP = {"Yes": 1, "No": 0, "Not a dejure resident": 0}

# ──────────────────────────────────────────────────────────────────
# CSS
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .banner-title {
        font-size: 2.8rem;
        font-weight: 900;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.1rem;
    }
    .banner-sub {
        font-size: 1.1rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        padding: 0.8rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# GOOGLE DRIVE FETCHER
# ──────────────────────────────────────────────────────────────────

def _fetch_gdrive(url: str) -> io.BytesIO:
    """
    Download a file from Google Drive, handling the large-file
    virus-scan confirmation page if it appears.
    """
    session = requests.Session()
    resp = session.get(url, timeout=180)
    resp.raise_for_status()

    # Google Drive shows an HTML confirmation page for large files
    content_type = resp.headers.get("Content-Type", "")
    if "text/html" in content_type:
        token_match = re.search(r'confirm=([0-9A-Za-z_\-]+)', resp.text)
        if token_match:
            confirmed_url = url + "&confirm=" + token_match.group(1)
            resp = session.get(confirmed_url, timeout=180)
            resp.raise_for_status()

    return io.BytesIO(resp.content)


def _open_file(filename: str):
    """
    Return a file-like object for `filename`.
    Checks local hackathon_data/ first (dev mode), then falls back
    to downloading from Google Drive (Streamlit Cloud).
    """
    local_path = DATA_DIR / filename
    if local_path.exists():
        return local_path          # local: return Path so pandas can memory-map
    return _fetch_gdrive(GDRIVE_URLS[filename])


# ──────────────────────────────────────────────────────────────────
# DATA LOADING  (cached — runs once per session)
# ──────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_raw() -> tuple:
    """
    Load, merge, encode, and score all three source files.

    Returns
    -------
    df         : pd.DataFrame  — 77,381 rows, analysis-ready
    geojson    : dict          — raw GeoJSON for choropleth map
    thresholds : dict          — data-driven cut-points for risk & education
    """

    # 1 ── Survey (only required columns)
    survey = pd.read_csv(
        _open_file("kenya_dhs_dataset_complete.csv"),
        usecols=SURVEY_COLS,
        low_memory=False,
    )

    # 2 ── Covariates
    cov_raw = _open_file("kenya_dhs_covariates.csv")
    cov = pd.read_csv(cov_raw)[
        ["DHSCLUST", "Travel_Times", "Malaria_Prevalence_2020",
         "Nightlights_Composite"]
    ]

    # 3 ── GeoJSON + GPS table
    geo_raw = _open_file("kenya_dhs_dataset_gps.geojson")
    if isinstance(geo_raw, Path):
        with open(geo_raw) as fh:
            geojson = json.load(fh)
    else:
        geojson = json.loads(geo_raw.read().decode("utf-8"))

    gps = pd.DataFrame([
        {
            "DHSCLUST": f["properties"]["DHSCLUST"],
            "ADM1NAME": f["properties"]["ADM1NAME"],
            "LATNUM":   f["properties"]["LATNUM"],
            "LONGNUM":  f["properties"]["LONGNUM"],
        }
        for f in geojson["features"]
    ])

    # 4 ── Merge
    df = (
        survey
        .merge(cov, left_on="V001", right_on="DHSCLUST", how="left")
        .merge(gps,  on="DHSCLUST",                       how="left")
    )

    # 5 ── Encode string columns → int8  (tiny memory)
    df["edu_level"]   = df["V106"].map(EDU_MAP).fillna(0).astype("int8")
    df["mobile"]      = df["V119"].map(BINARY_MAP).fillna(0).astype("int8")
    df["tv"]          = df["V120"].map(BINARY_MAP).fillna(0).astype("int8")
    df["radio"]       = df["V121"].map(BINARY_MAP).fillna(0).astype("int8")
    df["child_alive"] = df["B5"].map({"Yes": 1, "No": 0}).astype("float32")

    # Drop original string columns
    df.drop(columns=["V106", "V119", "V120", "V121", "B5", "DHSCLUST"],
            inplace=True)

    # 6 ── Education Score  0–10
    #   edu_level 0-3 → scaled to 0-5 | mobile +2 | tv +2 | radio +1
    df["edu_score"] = (
        df["edu_level"].astype("float32") / 3 * 5
        + df["mobile"] * 2
        + df["tv"]     * 2
        + df["radio"]  * 1
    ).clip(0, 10).astype("float32")

    # 7 ── Risk Score  0–10  (data-driven normalisation)
    def norm10(s: pd.Series) -> pd.Series:
        """Normalise series to [0, 10]."""
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series(0.0, index=s.index, dtype="float32")
        return ((s - lo) / (hi - lo) * 10).astype("float32")

    travel  = df["Travel_Times"].fillna(0)
    malaria = df["Malaria_Prevalence_2020"].fillna(0)

    # Normalise each component to [0, 10], then average → combined 0-10
    df["travel_score"]  = norm10(travel)
    df["malaria_score"] = norm10(malaria)
    df["risk_score"]    = (
        (df["travel_score"] + df["malaria_score"]) / 2
    ).astype("float32")

    # Component scores for "Risk Type" sidebar filter
    df["malaria_risk"] = df["malaria_score"]
    df["remote_risk"]  = df["travel_score"]

    # 8 ── Data-driven thresholds (percentile-based)
    #   Using 33rd and 67th percentiles so each bin has ~equal population
    edu_p33  = float(df["edu_score"].quantile(0.33))
    edu_p67  = float(df["edu_score"].quantile(0.67))
    risk_p33 = float(df["risk_score"].quantile(0.33))
    risk_p67 = float(df["risk_score"].quantile(0.67))

    thresholds = {
        "edu_lo":  edu_p33,
        "edu_hi":  edu_p67,
        "risk_lo": risk_p33,
        "risk_hi": risk_p67,
    }

    # 9 ── Categorical labels using data-driven thresholds
    df["risk_cat"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, risk_p33, risk_p67, 10.01],
        labels=["Low", "Medium", "High"],
    )
    df["edu_cat"] = pd.cut(
        df["edu_score"],
        bins=[-0.01, edu_p33, edu_p67, 10.01],
        labels=["Low", "Medium", "High"],
    )

    return df, geojson, thresholds


@st.cache_data(show_spinner=False)
def build_county_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    County-level aggregations.
    Shield effect = (survival rate of edu='High', risk='High')
                  - (survival rate of edu='Low',  risk='High')
    Min 10 obs per sub-group; otherwise NaN.
    """
    valid = df.dropna(subset=["ADM1NAME", "child_alive"])

    agg = valid.groupby("ADM1NAME", sort=False).agg(
        survival_rate=("child_alive", "mean"),
        sample_size  =("child_alive", "count"),
        avg_education=("edu_score",   "mean"),
        avg_risk     =("risk_score",  "mean"),
        latitude     =("LATNUM",      "first"),
        longitude    =("LONGNUM",     "first"),
    ).reset_index()
    agg["survival_rate"] = (agg["survival_rate"] * 100).round(2)

    effects = []
    for county in agg["ADM1NAME"]:
        sub = valid[valid["ADM1NAME"] == county]
        hi_risk = sub["risk_cat"] == "High"
        g_edu   = sub[hi_risk & (sub["edu_cat"] == "High")]["child_alive"]
        g_base  = sub[hi_risk & (sub["edu_cat"] == "Low") ]["child_alive"]
        if len(g_edu) >= 10 and len(g_base) >= 10:
            effects.append(round((g_edu.mean() - g_base.mean()) * 100, 2))
        else:
            effects.append(np.nan)

    agg["shield_effect"] = effects
    return agg


@st.cache_data(show_spinner=False)
def build_groups(df: pd.DataFrame, edu_threshold: float,
                 thresholds: dict) -> dict:
    """
    Compute survival stats for the four comparison groups.
    edu_threshold is the sidebar slider value (0-10).
    Risk threshold is the data-driven 67th percentile.
    """
    valid = df.dropna(subset=["child_alive"])
    risk_hi = thresholds["risk_hi"]
    risk_lo = thresholds["risk_lo"]

    hi_risk = valid["risk_score"] >= risk_hi
    lo_risk = valid["risk_score"] <= risk_lo
    hi_edu  = valid["edu_score"]  >= edu_threshold
    lo_edu  = valid["edu_score"]  <= thresholds["edu_lo"]

    def _s(mask):
        sub = valid[mask]
        rate = float(sub["child_alive"].mean() * 100) if len(sub) else 0.0
        return {"rate": round(rate, 2), "n": len(sub)}

    return {
        "hrhe": _s(hi_risk & hi_edu),
        "hrle": _s(hi_risk & lo_edu),
        "lrhe": _s(lo_risk & hi_edu),
        "lrle": _s(lo_risk & lo_edu),
    }


# ──────────────────────────────────────────────────────────────────
# CHART FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def chart_shield_bar(grps: dict) -> go.Figure:
    shield = grps["hrhe"]["rate"] - grps["hrle"]["rate"]

    plot_df = pd.DataFrame({
        "Group":   [f"Educated\n(n={grps['hrhe']['n']:,})",
                    f"Uneducated\n(n={grps['hrle']['n']:,})"],
        "Rate":    [grps["hrhe"]["rate"], grps["hrle"]["rate"]],
        "Colour":  ["Educated", "Uneducated"],
    })

    fig = px.bar(
        plot_df, x="Group", y="Rate", color="Colour", text="Rate",
        color_discrete_map={"Educated": "#27ae60", "Uneducated": "#e74c3c"},
        title="The Shield Effect: Child Survival in High-Risk Zones",
        labels={"Rate": "Child Survival Rate (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        showlegend=False,
        yaxis=dict(range=[0, 105], title="Child Survival Rate (%)"),
        xaxis_title="",
        height=460,
        font=dict(size=14),
    )
    color = "#27ae60" if shield >= 0 else "#c0392b"
    fig.add_annotation(
        text=f"<b>Shield Strength: {shield:+.1f}%</b>",
        xref="paper", yref="paper", x=0.5, y=1.10,
        showarrow=False,
        font=dict(size=18, color=color),
        bgcolor="#eafaf1" if shield >= 0 else "#fdedec",
        bordercolor=color, borderwidth=2, borderpad=8,
    )
    return fig


def chart_four_groups(grps: dict) -> go.Figure:
    plot_df = pd.DataFrame({
        "Group":    ["High-Risk\nEducated", "High-Risk\nUneducated",
                     "Low-Risk\nEducated",  "Low-Risk\nUneducated"],
        "Rate":     [grps["hrhe"]["rate"], grps["hrle"]["rate"],
                     grps["lrhe"]["rate"], grps["lrle"]["rate"]],
        "n":        [grps["hrhe"]["n"],    grps["hrle"]["n"],
                     grps["lrhe"]["n"],    grps["lrle"]["n"]],
        "Category": ["High Risk", "High Risk", "Low Risk", "Low Risk"],
    })

    fig = px.bar(
        plot_df, x="Group", y="Rate",
        color="Category", text="Rate",
        hover_data={"n": True, "Category": False},
        color_discrete_map={"High Risk": "#e74c3c", "Low Risk": "#3498db"},
        title="All Groups: Education × Geographic Risk",
        labels={"Rate": "Child Survival Rate (%)"},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        yaxis=dict(range=[0, 105], title="Child Survival Rate (%)"),
        xaxis_title="", height=460, font=dict(size=13),
    )
    return fig


def chart_county_map(cs: pd.DataFrame, geojson: dict = None) -> go.Figure:
    """
    Bubble map of shield effect by county.
    Uses scatter_mapbox with county centroids (lat/lon) because the
    GeoJSON contains cluster points, not county polygons.
    Counties with valid shield effect are shown as coloured bubbles.
    Counties with NaN shield effect are shown as small grey markers.
    """
    valid   = cs.dropna(subset=["shield_effect", "latitude", "longitude"])
    invalid = cs[cs["shield_effect"].isna()].dropna(subset=["latitude","longitude"])

    if valid.empty and invalid.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No county data available with current filters.<br>"
                 "Select <b>All Counties</b> to see the map.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=15),
        )
        fig.update_layout(height=500)
        return fig

    fig = go.Figure()

    # Grey markers for counties without enough data
    if not invalid.empty:
        fig.add_trace(go.Scattermapbox(
            lat=invalid["latitude"],
            lon=invalid["longitude"],
            mode="markers",
            marker=dict(size=10, color="#cccccc", opacity=0.5),
            text=invalid["ADM1NAME"] + "<br>Insufficient data",
            hoverinfo="text",
            name="Insufficient data",
            showlegend=True,
        ))

    # Coloured bubbles for counties with valid shield effect
    if not valid.empty:
        # Normalise bubble size: bigger = more births
        max_n  = valid["sample_size"].max()
        min_sz, max_sz = 15, 50
        sizes  = (valid["sample_size"] / max_n * (max_sz - min_sz) + min_sz).round()

        hover_text = (
            "<b>" + valid["ADM1NAME"] + "</b><br>"
            + "Shield Effect: "   + valid["shield_effect"].round(1).astype(str) + "%<br>"
            + "Survival Rate: "   + valid["survival_rate"].round(1).astype(str) + "%<br>"
            + "Avg Education: "   + valid["avg_education"].round(2).astype(str) + "<br>"
            + "Avg Risk: "        + valid["avg_risk"].round(2).astype(str) + "<br>"
            + "Sample Size: "     + valid["sample_size"].astype(int).astype(str)
        )

        fig.add_trace(go.Scattermapbox(
            lat=valid["latitude"],
            lon=valid["longitude"],
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=valid["shield_effect"],
                colorscale="RdYlGn",
                cmin=-20,
                cmax=20,
                colorbar=dict(
                    title="Shield<br>Effect (%)",
                    thickness=15,
                    len=0.6,
                ),
                opacity=0.85,
            ),
            text=valid["ADM1NAME"],
            textposition="top center",
            textfont=dict(size=9, color="#333333"),
            hovertext=hover_text,
            hoverinfo="text",
            name="Shield Effect",
            showlegend=False,
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": 0.5, "lon": 37.9},
            zoom=5.2,
        ),
        height=620,
        margin=dict(l=0, r=0, t=40, b=0),
        title="Shield Effect by County  (Green = Education Protects Most · Bubble size = sample size)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=0.01,
            xanchor="right",  x=0.99,
        ),
    )
    return fig


def chart_scatter(df: pd.DataFrame, county: str) -> go.Figure:
    plot_df = df.dropna(subset=["child_alive", "edu_score", "risk_cat"])
    if county != "All Counties":
        plot_df = plot_df[plot_df["ADM1NAME"] == county]

    agg = (
        plot_df
        .groupby(["V001", "risk_cat"], observed=True)
        .agg(
            edu_score   =("edu_score",   "mean"),
            survival_pct=("child_alive", "mean"),
            n           =("child_alive", "count"),
            county      =("ADM1NAME",    "first"),
        )
        .reset_index()
    )
    agg["survival_pct"] = agg["survival_pct"] * 100

    fig = px.scatter(
        agg,
        x="edu_score", y="survival_pct",
        color="risk_cat", size="n",
        hover_data=["county", "n"],
        trendline="ols",
        color_discrete_map={
            "Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"
        },
        category_orders={"risk_cat": ["Low", "Medium", "High"]},
        title="Education Score vs Child Survival Rate by Risk Level",
        labels={
            "edu_score":    "Education Score (0–10)",
            "survival_pct": "Child Survival Rate (%)",
            "risk_cat":     "Risk Level",
        },
    )
    fig.update_layout(
        height=480,
        xaxis=dict(range=[0, 10]),
        yaxis=dict(range=[0, 100]),
        font=dict(size=12),
    )
    return fig


def chart_edu_by_risk(df: pd.DataFrame) -> go.Figure:
    valid = df.dropna(subset=["risk_cat", "edu_cat"])
    counts = (
        valid.groupby(["risk_cat", "edu_cat"], observed=True)
        .size().reset_index(name="n")
    )
    totals = (
        valid.groupby("risk_cat", observed=True)
        .size().reset_index(name="total")
    )
    merged = counts.merge(totals, on="risk_cat")
    merged["pct"] = merged["n"] / merged["total"] * 100

    fig = px.bar(
        merged,
        x="risk_cat", y="pct", color="edu_cat",
        text="pct", barmode="stack",
        color_discrete_map={
            "Low": "#e74c3c", "Medium": "#f39c12", "High": "#27ae60"
        },
        category_orders={
            "risk_cat": ["Low", "Medium", "High"],
            "edu_cat":  ["Low", "Medium", "High"],
        },
        title="Education Level Distribution by Geographic Risk Zone",
        labels={
            "risk_cat": "Geographic Risk Zone",
            "pct":      "Percentage of Mothers (%)",
            "edu_cat":  "Education Level",
        },
    )
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
    fig.update_layout(height=400, font=dict(size=12))
    return fig


# ──────────────────────────────────────────────────────────────────
# AUTO-INSIGHTS
# ──────────────────────────────────────────────────────────────────

def make_insights(df, grps, cs):
    pos, warn, info = [], [], []

    shield = grps["hrhe"]["rate"] - grps["hrle"]["rate"]

    if shield > 10:
        pos.append(
            f"💪 **Strong shield:** Educated mothers in high-risk zones "
            f"have **{shield:.1f}%** higher child survival "
            f"(n={grps['hrhe']['n']:,} vs n={grps['hrle']['n']:,})."
        )
    elif shield > 0:
        info.append(
            f"ℹ️ **Moderate shield:** Education provides a **{shield:.1f}%** "
            f"protective effect in high-risk zones."
        )
    else:
        warn.append(
            f"⚠️ **Weak shield ({shield:.1f}%):** "
            "Try adjusting the education threshold or risk type."
        )

    hrhe_r, lrle_r = grps["hrhe"]["rate"], grps["lrle"]["rate"]
    if hrhe_r >= lrle_r:
        pos.append(
            f"🌟 **Education overcomes geography:** Educated mothers in "
            f"high-risk zones ({hrhe_r:.1f}%) match or exceed uneducated "
            f"mothers in safe zones ({lrle_r:.1f}%)."
        )
    else:
        warn.append(
            f"📍 **Geography still matters:** Educated mothers in high-risk "
            f"zones lag **{lrle_r - hrhe_r:.1f}%** behind uneducated mothers "
            f"in safe zones."
        )

    overall = df["child_alive"].mean() * 100
    info.append(
        f"📊 Overall child survival: **{overall:.1f}%** across "
        f"**{len(df):,}** births in **{df['ADM1NAME'].nunique()}** counties."
    )

    valid_cs = cs.dropna(subset=["shield_effect"])
    if not valid_cs.empty:
        top3 = valid_cs.nlargest(3, "shield_effect")["ADM1NAME"].tolist()
        pos.append(
            f"🗺️ **Strongest shield counties:** {', '.join(top3)} — "
            "education makes the biggest difference here."
        )
        priority = valid_cs[
            (valid_cs["avg_risk"] > valid_cs["avg_risk"].median()) &
            (valid_cs["shield_effect"] < 5)
        ]
        if not priority.empty:
            pr3 = priority.nlargest(3, "avg_risk")["ADM1NAME"].tolist()
            warn.append(
                f"🚨 **Priority intervention counties:** {', '.join(pr3)} — "
                "high risk, weak shield — invest in education here first."
            )

    return pos, warn, info


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():

    # Banner
    st.markdown(
        '<h1 class="banner-title">🛡️ MamaShield: The Digital Shield</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="banner-sub">'
        "Can Education and Information Access Protect Mothers in High-Risk Zones?"
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Load ────────────────────────────────────────────────────
    with st.spinner(
        "⏳ Loading full dataset (77,381 births) — first run ~30–60 s…"
    ):
        try:
            df_full, geojson, thresholds = load_raw()
        except requests.exceptions.RequestException as exc:
            st.error(f"Network error downloading data: {exc}")
            st.info(
                "Check that the Google Drive files are publicly shared "
                "(Anyone with the link → Viewer)."
            )
            st.stop()
        except FileNotFoundError as exc:
            st.error(f"Data file not found: {exc}")
            st.info(
                "Ensure all four files are inside **hackathon_data/** "
                "in the same folder as app.py, or verify the Google Drive "
                "sharing permissions."
            )
            st.stop()
        except Exception as exc:
            st.error(f"Unexpected error loading data: {exc}")
            st.stop()

    # ── Sidebar ─────────────────────────────────────────────────
    with st.sidebar:
        st.title("📊 Filters")
        st.markdown("---")

        # Show user what the data-driven thresholds are
        st.caption(
            f"Data-driven risk threshold (67th pct): "
            f"{thresholds['risk_hi']:.2f}"
        )

        edu_threshold = st.slider(
            "Min Education Score (defines 'educated')",
            min_value=0.0, max_value=10.0,
            value=round(float(thresholds["edu_hi"]), 1),
            step=0.5,
            help="Mothers at or above this score are classed as 'educated'.",
        )

        risk_type = st.selectbox(
            "Risk Type",
            ["Combined Risk", "Malaria Risk Only", "Remoteness Only"],
        )

        all_counties = sorted(df_full["ADM1NAME"].dropna().unique().tolist())
        county_choice = st.selectbox(
            "County Filter (optional)",
            ["All Counties"] + all_counties,
        )

        st.markdown("---")
        st.info(
            "**Track 5 — The Digital Shield**\n\n"
            "Analysing whether education and information access shield "
            "mothers from health risks in remote or high-malaria areas."
        )

    # ── Apply filters ────────────────────────────────────────────
    df = df_full.copy()

    if risk_type == "Malaria Risk Only":
        df["risk_score"] = df["malaria_risk"]
    elif risk_type == "Remoteness Only":
        df["risk_score"] = df["remote_risk"]

    # Re-compute data-driven thresholds for the active risk score
    rp33 = float(df["risk_score"].quantile(0.33))
    rp67 = float(df["risk_score"].quantile(0.67))
    active_thresholds = {**thresholds, "risk_lo": rp33, "risk_hi": rp67}

    df["risk_cat"] = pd.cut(
        df["risk_score"],
        bins=[-0.01, rp33, rp67, df["risk_score"].max() + 0.01],
        labels=["Low", "Medium", "High"],
    )

    if county_choice != "All Counties":
        df = df[df["ADM1NAME"] == county_choice]

    # ── Aggregations ────────────────────────────────────────────
    cs   = build_county_stats(df)
    grps = build_groups(df, edu_threshold, active_thresholds)

    # ── Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "🗺️ County Map",
        "🛡️ Shield Effect",
        "📊 Deep Dive",
        "💡 Insights",
    ])

    # ════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════
    with tab1:
        st.header("Overview")

        total         = len(df)
        survival_pct  = float(df["child_alive"].mean() * 100)
        high_risk_n   = int((df["risk_cat"] == "High").sum())
        high_risk_pct = high_risk_n / total * 100
        n_counties    = int(df["ADM1NAME"].nunique())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Births",        f"{total:,}")
        c2.metric("Child Survival Rate", f"{survival_pct:.1f}%")
        c3.metric("In High-Risk Zones",  f"{high_risk_pct:.1f}%")
        c4.metric("Counties Covered",    str(n_counties))

        st.markdown("---")
        st.subheader("Who Needs Protection? Education Levels by Risk Zone")
        st.plotly_chart(chart_edu_by_risk(df), use_container_width=True)
        st.info(
            "💡 High-risk zones have the highest share of low-education "
            "mothers — exactly where the shield is needed most."
        )

    # ════════════════════════════════════════════════════════════
    # TAB 2 — COUNTY MAP
    # ════════════════════════════════════════════════════════════
    with tab2:
        st.header("Geographic Shield Effect by County")
        st.markdown(
            "**Shield Effect** = child survival rate of *educated* mothers "
            "in high-risk zones **minus** *uneducated* mothers in the same zones.\n\n"
            "🟢 Green = strong shield &nbsp;|&nbsp; "
            "🟡 Yellow = moderate &nbsp;|&nbsp; "
            "🔴 Red = weak or negative"
        )
        st.plotly_chart(chart_county_map(cs, geojson=None), use_container_width=True)

        with st.expander("📋 County Statistics Table"):
            want = ["ADM1NAME", "survival_rate", "shield_effect",
                    "avg_education", "avg_risk", "sample_size"]
            disp = cs[[c for c in want if c in cs.columns]].rename(columns={
                "ADM1NAME":      "County",
                "survival_rate": "Survival Rate (%)",
                "shield_effect": "Shield Effect (%)",
                "avg_education": "Avg Education Score",
                "avg_risk":      "Avg Risk Score",
                "sample_size":   "Sample Size",
            })
            if "Shield Effect (%)" in disp.columns and \
                    disp["Shield Effect (%)"].notna().any():
                disp = disp.sort_values("Shield Effect (%)", ascending=False)
            st.dataframe(disp, hide_index=True, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 3 — SHIELD EFFECT
    # ════════════════════════════════════════════════════════════
    with tab3:
        st.header("The Shield Effect — Core Analysis")
        st.markdown(
            "**Critical question:** In high-risk zones, do educated mothers "
            "have better child survival than uneducated mothers?"
        )

        st.plotly_chart(chart_shield_bar(grps), use_container_width=True)

        shield_val = grps["hrhe"]["rate"] - grps["hrle"]["rate"]
        min_n      = min(grps["hrhe"]["n"], grps["hrle"]["n"])

        if min_n < 30:
            st.warning(
                "⚠️ **Small sample warning:** one or more groups has < 30 obs. "
                "Select **All Counties** or adjust the education threshold."
            )
        elif shield_val > 10:
            st.success(f"✅ **Strong shield: +{shield_val:.1f}%** — "
                       "education makes a meaningful difference in high-risk zones!")
        elif shield_val > 0:
            st.info(f"ℹ️ **Moderate shield: +{shield_val:.1f}%** — "
                    "education helps but the effect is modest.")
        else:
            st.warning(f"⚠️ **No shield detected ({shield_val:.1f}%)** — "
                       "try different filters.")

        st.markdown("---")
        st.subheader("Full Comparison: All Four Groups")
        st.plotly_chart(chart_four_groups(grps), use_container_width=True)

        st.markdown("---")
        st.subheader("Can Education Overcome Geography Entirely?")

        hrhe_r = grps["hrhe"]["rate"]
        lrle_r = grps["lrle"]["rate"]
        col1, col2 = st.columns(2)
        col1.metric("Educated — HIGH-Risk Zones",
                    f"{hrhe_r:.1f}%",
                    delta=f"n = {grps['hrhe']['n']:,}")
        col2.metric("Uneducated — LOW-Risk Zones",
                    f"{lrle_r:.1f}%",
                    delta=f"n = {grps['lrle']['n']:,}")

        if hrhe_r >= lrle_r:
            st.success(
                "✅ **YES — education can overcome geography.** "
                "Educated mothers in dangerous zones achieve equal or better "
                "survival than uneducated mothers in safe zones."
            )
        else:
            st.warning(
                f"⚠️ **Partially.** Educated mothers in high-risk zones "
                f"still trail uneducated safe-zone mothers by "
                f"{lrle_r - hrhe_r:.1f}%. Both education AND location matter."
            )

    # ════════════════════════════════════════════════════════════
    # TAB 4 — DEEP DIVE
    # ════════════════════════════════════════════════════════════
    with tab4:
        st.header("Deep Dive Analysis")

        st.subheader("Education Score vs Child Survival Rate by Risk Level")
        st.plotly_chart(
            chart_scatter(df, county_choice), use_container_width=True
        )
        st.caption(
            "Each bubble = one DHS cluster · "
            "Bubble size = number of births · "
            "Colour = geographic risk · "
            "Lines = OLS trend per risk group"
        )

        st.markdown("---")
        st.subheader("Group Statistics Summary")
        summary_df = pd.DataFrame({
            "Group": [
                "High-Risk + High-Education  ← shield group",
                "High-Risk + Low-Education   ← baseline",
                "Low-Risk  + High-Education",
                "Low-Risk  + Low-Education",
            ],
            "Child Survival Rate (%)": [
                grps["hrhe"]["rate"], grps["hrle"]["rate"],
                grps["lrhe"]["rate"], grps["lrle"]["rate"],
            ],
            "Sample Size": [
                grps["hrhe"]["n"], grps["hrle"]["n"],
                grps["lrhe"]["n"], grps["lrle"]["n"],
            ],
        })
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # ════════════════════════════════════════════════════════════
    # TAB 5 — INSIGHTS
    # ════════════════════════════════════════════════════════════
    with tab5:
        st.header("Key Insights & Recommendations")

        pos, warn, info = make_insights(df, grps, cs)

        if pos:
            st.subheader("✅ Positive Findings")
            for m in pos:
                st.success(m)
        if warn:
            st.subheader("⚠️ Areas of Concern")
            for m in warn:
                st.warning(m)
        if info:
            st.subheader("ℹ️ Context")
            for m in info:
                st.info(m)

        st.markdown("---")
        col1, col2 = st.columns(2)
        valid_cs = cs.dropna(subset=["shield_effect"])

        with col1:
            st.subheader("🎯 Priority Counties for Investment")
            if not valid_cs.empty:
                priority = (
                    valid_cs[valid_cs["sample_size"] >= 20]
                    .nlargest(5, "avg_risk")
                    [["ADM1NAME", "avg_risk", "shield_effect",
                      "avg_education", "sample_size"]]
                    .rename(columns={
                        "ADM1NAME":      "County",
                        "avg_risk":      "Risk Score",
                        "shield_effect": "Shield Effect (%)",
                        "avg_education": "Avg Education",
                        "sample_size":   "n",
                    })
                )
                st.dataframe(priority, hide_index=True,
                             use_container_width=True)
                st.caption(
                    "Highest-risk counties — invest in education here first."
                )
            else:
                st.info("Insufficient data.")

        with col2:
            st.subheader("🌟 Success Stories")
            if not valid_cs.empty:
                success = (
                    valid_cs[
                        (valid_cs["avg_risk"] >
                         valid_cs["avg_risk"].median()) &
                        (valid_cs["shield_effect"] > 5) &
                        (valid_cs["sample_size"] >= 20)
                    ]
                    .nlargest(3, "shield_effect")
                    [["ADM1NAME", "avg_risk", "shield_effect",
                      "avg_education", "sample_size"]]
                    .rename(columns={
                        "ADM1NAME":      "County",
                        "avg_risk":      "Risk Score",
                        "shield_effect": "Shield Effect (%)",
                        "avg_education": "Avg Education",
                        "sample_size":   "n",
                    })
                )
                if not success.empty:
                    st.dataframe(success, hide_index=True,
                                 use_container_width=True)
                    st.caption(
                        "High-risk counties where education is "
                        "already working — learn from them!"
                    )
                else:
                    st.info("No success stories with current filters.")
            else:
                st.info("Insufficient data.")

        st.markdown("---")
        st.subheader("📋 Policy Recommendations")
        st.markdown("""
1. **Invest in secondary education** in high-risk counties with low current
   education levels — the data shows it saves lives.
2. **Expand mobile phone access** — it contributes directly to the shield
   score and is far cheaper than building schools alone.
3. **Study success counties** — understand what enables education to protect
   there, and replicate those conditions elsewhere.
4. **Deploy community health workers** in remote areas — travel time is the
   other half of the risk equation that education alone cannot fully overcome.
5. **Combine interventions** — the strongest outcomes appear where education,
   information access, and healthcare proximity all improve together.
        """)


if __name__ == "__main__":
    main()