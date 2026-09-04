"""
EDA Module for MindPredict AI
Generates interactive visualizations and analytical insights.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from typing import List, Optional, Dict


PREVALENCE_COLS = {
    "schizophrenia_prev": "Schizophrenia",
    "depressive_prev": "Depressive Disorders",
    "anxiety_prev": "Anxiety Disorders",
    "bipolar_prev": "Bipolar Disorders",
    "eating_prev": "Eating Disorders",
}

DALY_COLS = {
    "schizophrenia_daly": "Schizophrenia DALY",
    "depressive_daly": "Depressive DALY",
    "anxiety_daly": "Anxiety DALY",
    "bipolar_daly": "Bipolar DALY",
    "eating_daly": "Eating Disorders DALY",
}


def plot_global_trend(df: pd.DataFrame, indicator: str = "depressive_prev") -> go.Figure:
    """Plot the global average trend of a mental health indicator over time."""
    if "Entity" in df.columns:
        non_country = ["World", "High-income countries", "Low-income countries",
                       "Lower-middle-income countries", "Upper-middle-income countries"]
        world_df = df[df["Entity"].isin(non_country)]
        if not world_df.empty:
            trend_df = world_df.groupby("Year")[indicator].mean().reset_index()
        else:
            trend_df = df.groupby("Year")[indicator].mean().reset_index()
    else:
        trend_df = df.groupby("Year")[indicator].mean().reset_index()

    fig = px.line(
        trend_df, x="Year", y=indicator,
        title=f"Global Average Trend: {PREVALENCE_COLS.get(indicator, indicator)}",
        markers=True,
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Prevalence (share of population)",
        template="plotly_white", height=450,
    )
    return fig


def plot_country_comparison(df: pd.DataFrame, indicator: str = "depressive_prev",
                           year: int = 2019, top_n: int = 15) -> go.Figure:
    """Plot top/bottom countries for a given indicator and year."""
    year_df = df[df["Year"] == year].copy()
    non_countries = ["World", "High-income countries", "Low-income countries",
                     "Lower-middle-income countries", "Upper-middle-income countries"]
    year_df = year_df[~year_df["Entity"].isin(non_countries)]

    top = year_df.nlargest(top_n, indicator)
    bottom = year_df.nsmallest(top_n, indicator)
    combined = pd.concat([top, bottom]).drop_duplicates()

    fig = px.bar(
        combined.sort_values(indicator, ascending=True),
        x=indicator, y="Entity",
        orientation="h",
        title=f"Country Comparison: {PREVALENCE_COLS.get(indicator, indicator)} ({year})",
        color=indicator,
        color_continuous_scale="RdYlBu_r",
    )
    fig.update_layout(
        xaxis_title="Prevalence (share of population)",
        yaxis_title="", template="plotly_white", height=600,
    )
    return fig


def plot_disorder_distribution(df: pd.DataFrame, year: int = 2019) -> go.Figure:
    """Plot the distribution of prevalence across all disorders for a given year."""
    year_df = df[df["Year"] == year].copy()
    non_countries = ["World", "High-income countries", "Low-income countries",
                     "Lower-middle-income countries", "Upper-middle-income countries"]
    year_df = year_df[~year_df["Entity"].isin(non_countries)]

    cols = list(PREVALENCE_COLS.keys())
    available = [c for c in cols if c in year_df.columns]

    data = []
    for col in available:
        data.append(go.Box(y=year_df[col], name=PREVALENCE_COLS[col], boxmean=True))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=f"Distribution of Mental Disorder Prevalence ({year})",
        yaxis_title="Prevalence (share of population)",
        template="plotly_white", height=450, showlegend=False,
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plot correlation heatmap of all numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    cols_of_interest = [c for c in list(PREVALENCE_COLS.keys()) + list(DALY_COLS.keys())
                        if c in numeric_df.columns]
    if len(cols_of_interest) < 2:
        cols_of_interest = list(numeric_df.columns[:10])

    corr = numeric_df[cols_of_interest].corr()

    nice_names = {c: PREVALENCE_COLS.get(c, DALY_COLS.get(c, c)) for c in cols_of_interest}
    corr.index = [nice_names.get(c, c) for c in corr.index]
    corr.columns = [nice_names.get(c, c) for c in corr.columns]

    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=600, width=800)
    return fig


def plot_daly_vs_prevalence(df: pd.DataFrame, disorder: str = "depressive") -> go.Figure:
    """Scatter plot of DALY rate vs prevalence for a specific disorder."""
    prev_col = f"{disorder}_prev"
    daly_col = f"{disorder}_daly"

    if prev_col not in df.columns or daly_col not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Data not available for this disorder")
        return fig

    fig = px.scatter(
        df, x=prev_col, y=daly_col,
        color="Entity", hover_data=["Year"],
        title=f"{PREVALENCE_COLS.get(prev_col, disorder)}: Prevalence vs Disease Burden (DALY)",
        opacity=0.6,
        color_discrete_sequence=px.colors.qualitative.Light24,
    )
    fig.update_layout(
        xaxis_title="Prevalence (share of population)",
        yaxis_title="DALY Rate",
        template="plotly_white", height=500, showlegend=False,
    )
    return fig


def plot_regional_heatmap_trend(df: pd.DataFrame, indicator: str = "depressive_prev") -> go.Figure:
    """Heatmap of indicator values across regions over time."""
    non_countries = ["World", "High-income countries", "Low-income countries",
                     "Lower-middle-income countries", "Upper-middle-income countries"]
    region_df = df[df["Entity"].isin(non_countries)]

    if region_df.empty:
        pivot = df.pivot_table(values=indicator, index="Entity", columns="Year", aggfunc="mean")
    else:
        pivot = region_df.pivot_table(values=indicator, index="Entity", columns="Year", aggfunc="mean")

    fig = px.imshow(
        pivot, color_continuous_scale="YlOrRd",
        title=f"Prevalence Heatmap: {PREVALENCE_COLS.get(indicator, indicator)}",
        aspect="auto",
    )
    fig.update_layout(template="plotly_white", height=400)
    return fig


def plot_missing_values(df: pd.DataFrame) -> go.Figure:
    """Plot missing values as a bar chart."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        fig = go.Figure()
        fig.update_layout(title="No Missing Values Found", template="plotly_white")
        return fig

    fig = px.bar(
        x=missing.index, y=missing.values,
        title="Missing Values by Feature",
        labels={"x": "Feature", "y": "Missing Count"},
        color=missing.values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(template="plotly_white", height=400, showlegend=False)
    return fig


def plot_treatment_gap(treatment_df: pd.DataFrame) -> go.Figure:
    """Plot anxiety treatment gap data."""
    if treatment_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Treatment Data Available")
        return fig

    countries_to_show = treatment_df[
        ~treatment_df["Entity"].isin(["High-income countries", "Low-income countries"])
    ].nlargest(15, "Untreated, conditional")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Untreated", x=countries_to_show["Entity"],
        y=countries_to_show["Untreated, conditional"],
        marker_color="#EF553B",
    ))
    fig.add_trace(go.Bar(
        name="Other Treatment", x=countries_to_show["Entity"],
        y=countries_to_show["Other treatments, conditional"],
        marker_color="#636EFA",
    ))
    fig.add_trace(go.Bar(
        name="Adequate Treatment", x=countries_to_show["Entity"],
        y=countries_to_show["Potentially adequate treatment, conditional"],
        marker_color="#00CC96",
    ))
    fig.update_layout(
        barmode="stack", title="Anxiety Disorders Treatment Gap by Country",
        yaxis_title="Percentage (%)", template="plotly_white", height=500,
    )
    return fig


def plot_us_symptoms(us_df: pd.DataFrame) -> go.Figure:
    """Plot US depressive symptoms data."""
    if us_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No US Symptoms Data Available")
        return fig

    freq_cols = ["Nearly every day", "More than half the days", "Several days", "Not at all"]
    available = [c for c in freq_cols if c in us_df.columns]

    fig = go.Figure()
    colors = ["#EF553B", "#FFA15A", "#FECB52", "#00CC96"]
    for i, col in enumerate(available):
        fig.add_trace(go.Bar(
            name=col, x=us_df["Entity"], y=us_df[col],
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="group", title="Depressive Symptom Frequency in US Population (2014)",
        yaxis_title="Percentage (%)", template="plotly_white", height=500,
    )
    return fig


def plot_feature_target_relationship(df: pd.DataFrame, feature: str,
                                      target: str = "depressive_prev") -> go.Figure:
    """Scatter plot of a feature against the target variable."""
    if feature not in df.columns or target not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"Feature '{feature}' not found")
        return fig

    fig = px.scatter(
        df, x=feature, y=target, color="Year",
        title=f"{PREVALENCE_COLS.get(feature, feature)} vs {PREVALENCE_COLS.get(target, target)}",
        opacity=0.5, color_continuous_scale="viridis",
    )
    fig.update_layout(template="plotly_white", height=450)
    return fig


def plot_country_timeseries(df: pd.DataFrame, countries: List[str],
                            indicator: str = "depressive_prev") -> go.Figure:
    """Plot time series for selected countries."""
    filtered = df[df["Entity"].isin(countries)]
    fig = px.line(
        filtered, x="Year", y=indicator, color="Entity",
        title=f"{PREVALENCE_COLS.get(indicator, indicator)} Over Time",
        markers=True,
    )
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Prevalence (share of population)",
        template="plotly_white", height=450,
    )
    return fig


def generate_insights(df: pd.DataFrame) -> List[Dict]:
    """Generate automated data-driven insights."""
    insights = []

    non_countries = ["World", "High-income countries", "Low-income countries",
                     "Lower-middle-income countries", "Upper-middle-income countries"]
    country_df = df[~df["Entity"].isin(non_countries)]

    for col, name in PREVALENCE_COLS.items():
        if col not in df.columns:
            continue

        if len(country_df) > 0:
            latest = country_df[country_df["Year"] == country_df["Year"].max()]
            earliest = country_df[country_df["Year"] == country_df["Year"].min()]

            if not latest.empty and not earliest.empty:
                latest_avg = latest[col].mean()
                earliest_avg = earliest[col].mean()

                if latest_avg > earliest_avg * 1.05:
                    trend = "increased"
                elif latest_avg < earliest_avg * 0.95:
                    trend = "decreased"
                else:
                    trend = "remained relatively stable"

                insights.append({
                    "category": "Trend",
                    "title": f"{name} Trend",
                    "finding": f"Global average {name.lower()} prevalence has {trend} over the study period "
                              f"(from {earliest_avg:.4f} to {latest_avg:.4f} of population).",
                    "importance": "high",
                })

        if not latest.empty:
            top3 = latest.nlargest(3, col)
            if not top3.empty:
                countries_str = ", ".join(top3["Entity"].tolist())
                insights.append({
                    "category": "Geographic",
                    "title": f"Highest {name} Prevalence",
                    "finding": f"The countries with the highest {name.lower()} prevalence in "
                              f"{latest['Year'].iloc[0]} are: {countries_str}.",
                    "importance": "medium",
                })

    for col, name in PREVALENCE_COLS.items():
        daly_col = col.replace("_prev", "_daly")
        if col in df.columns and daly_col in df.columns:
            corr = df[[col, daly_col]].dropna().corr().iloc[0, 1]
            insights.append({
                "category": "Correlation",
                "title": f"{name}: Prevalence vs Burden",
                "finding": f"Correlation between {name.lower()} prevalence and disease burden (DALY): {corr:.3f}.",
                "importance": "high" if abs(corr) > 0.7 else "medium",
            })

    for col, name in PREVALENCE_COLS.items():
        if col not in country_df.columns:
            continue
        latest = country_df[country_df["Year"] == country_df["Year"].max()]
        if latest.empty:
            continue
        std = latest[col].std()
        mean = latest[col].mean()
        cv = (std / mean * 100) if mean > 0 else 0
        insights.append({
            "category": "Variability",
            "title": f"{name} Global Variability",
            "finding": f"Cross-country coefficient of variation for {name.lower()}: {cv:.1f}%. "
                      f"{'High' if cv > 50 else 'Moderate' if cv > 25 else 'Low'} variability across nations.",
            "importance": "medium",
        })

    return insights
