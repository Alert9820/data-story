"""
DataStory — Production-ready AI-style Data Storytelling Dashboard
No external AI APIs. Pure Python analytics + Flask + Plotly.
"""

import os, io, json, traceback, uuid, re
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from scipy import stats as scipy_stats
from flask import Flask, jsonify, render_template, request, send_file

# ── App Setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

UPLOAD_DIR = Path("uploads")
CLEANED_DIR = Path("cleaned")
UPLOAD_DIR.mkdir(exist_ok=True)
CLEANED_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {"csv", "xlsx", "xls"}

# ── Plotly Theme ──────────────────────────────────────────────────────────────
PALETTE = ["#38bdf8", "#818cf8", "#34d399", "#fb923c", "#f472b6", "#a78bfa", "#fbbf24"]
CHART_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.04)"
FONT = dict(family="'Syne', sans-serif", color="#cbd5e1", size=12)


def _theme(fig, title="") -> dict:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#94a3b8"), x=0),
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=FONT,
        colorway=PALETTE,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#f8fafc")),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zeroline=False, showline=False, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False, showline=False, tickfont=dict(size=10))
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


# ── File Helpers ──────────────────────────────────────────────────────────────
def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def _load(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(path, encoding=enc)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, encoding="utf-8", errors="replace")
    return pd.read_excel(path)


# ── Phase 1 — Data Cleaning ───────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    before_rows, before_cols = len(df), len(df.columns)

    # Normalize column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w\s]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # Drop fully empty rows/cols
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)

    # Duplicates
    dups = int(df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    report["duplicates_removed"] = dups

    # Detect & convert date columns
    date_cols_found = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(30)
        try:
            conv = pd.to_datetime(sample, errors="coerce")
            if conv.notna().sum() >= len(sample) * 0.75:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                date_cols_found.append(col)
        except Exception:
            pass
    report["date_columns"] = date_cols_found

    # String-encoded numerics
    for col in df.select_dtypes(include="object").columns:
        cleaned = df[col].str.replace(r"[\$,€£₹%]", "", regex=True).str.strip()
        num = pd.to_numeric(cleaned, errors="coerce")
        if num.notna().sum() > len(df) * 0.6:
            df[col] = num

    # Missing values
    missing_total = int(df.isnull().sum().sum())
    for col in df.columns:
        if col in date_cols_found:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == object:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if len(mode) else "Unknown")
    report["missing_filled"] = missing_total

    # Outlier detection (IQR capping)
    outlier_cols, total_outliers = [], 0
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (df[col] < lo) | (df[col] > hi)
        n = int(mask.sum())
        if n:
            df[col] = df[col].clip(lo, hi)
            outlier_cols.append(col)
            total_outliers += n
    report["outliers_detected"] = total_outliers
    report["outlier_cols"] = outlier_cols

    report["rows_before"] = before_rows
    report["rows_after"] = len(df)
    report["cols_before"] = before_cols
    report["cols_after"] = len(df.columns)
    report["rows_removed"] = before_rows - len(df)

    # Quality score (0-100)
    completeness = max(0, 1 - missing_total / max(before_rows * before_cols, 1))
    dup_penalty = max(0, 1 - dups / max(before_rows, 1))
    outlier_penalty = max(0, 1 - total_outliers / max(before_rows * 2, 1))
    report["quality_score"] = round((completeness * 0.5 + dup_penalty * 0.3 + outlier_penalty * 0.2) * 100, 1)

    return df, report


# ── Column Detection ──────────────────────────────────────────────────────────
def detect_columns(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include="object").columns.tolist()
    date = df.select_dtypes(include=["datetime64"]).columns.tolist()
    return {
        "numeric": numeric,
        "categorical": categorical,
        "date": date,
        "best_num": numeric[0] if numeric else None,
        "best_cat": categorical[0] if categorical else None,
        "best_date": date[0] if date else None,
    }


# ── Phase 2 — Visualizations ──────────────────────────────────────────────────
def generate_visualizations(df: pd.DataFrame) -> dict:
    cols = detect_columns(df)
    num = cols["numeric"]
    cat = cols["categorical"]
    date = cols["date"]
    charts = {}

    # 1. Bar chart
    if cols["best_cat"] and cols["best_num"]:
        agg = df.groupby(cols["best_cat"])[cols["best_num"]].sum().nlargest(12).reset_index()
        fig = px.bar(
            agg, x=cols["best_cat"], y=cols["best_num"],
            color=cols["best_num"], color_continuous_scale="Blues",
            text=cols["best_num"],
        )
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside",
                          marker_line_width=0, opacity=0.9)
        charts["bar"] = _theme(fig, f"Top {cols['best_cat'].replace('_',' ').title()} by {cols['best_num'].replace('_',' ').title()}")

    # 2. Line / trend chart
    if date and cols["best_num"]:
        ldf = df.sort_values(cols["best_date"])
        fig = px.line(ldf, x=cols["best_date"], y=cols["best_num"], markers=True,
                      color_discrete_sequence=["#38bdf8"])
        fig.update_traces(line_width=2.5, marker_size=5)
        charts["line"] = _theme(fig, f"{cols['best_num'].replace('_',' ').title()} Trend Over Time")
    elif cols["best_cat"] and cols["best_num"]:
        grp = df.groupby(cols["best_cat"])[cols["best_num"]].mean().reset_index()
        fig = px.line(grp, x=cols["best_cat"], y=cols["best_num"], markers=True,
                      color_discrete_sequence=["#38bdf8"])
        fig.update_traces(line_width=2.5, marker_size=6)
        charts["line"] = _theme(fig, f"Avg {cols['best_num'].replace('_',' ').title()} Across Categories")

    # 3. Pie / Donut
    if cols["best_cat"] and cols["best_num"]:
        pdf = df.groupby(cols["best_cat"])[cols["best_num"]].sum().nlargest(8).reset_index()
        fig = px.pie(pdf, names=cols["best_cat"], values=cols["best_num"],
                     hole=0.5, color_discrete_sequence=PALETTE)
        fig.update_traces(textposition="inside", textinfo="percent+label",
                          marker_line_width=2, marker_line_color="rgba(0,0,0,0.3)")
        charts["pie"] = _theme(fig, f"Share of {cols['best_num'].replace('_',' ').title()}")

    # 4. Histogram
    if cols["best_num"]:
        fig = px.histogram(df, x=cols["best_num"], nbins=30,
                           color_discrete_sequence=["#818cf8"])
        fig.update_traces(marker_line_width=0, opacity=0.85)
        charts["histogram"] = _theme(fig, f"Distribution of {cols['best_num'].replace('_',' ').title()}")

    # 5. Scatter
    if len(num) >= 2:
        color_col = cat[0] if cat else None
        sdf = df.sample(min(500, len(df)), random_state=42)
        fig = px.scatter(sdf, x=num[0], y=num[1], color=color_col,
                         opacity=0.72, size_max=8,
                         color_discrete_sequence=PALETTE)
        fig.update_traces(marker_size=6)
        charts["scatter"] = _theme(fig, f"{num[0].replace('_',' ').title()} vs {num[1].replace('_',' ').title()}")

    # 6. Correlation heatmap
    if len(num) >= 3:
        corr = df[num[:10]].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1)
        fig.update_traces(textfont_size=10)
        charts["heatmap"] = _theme(fig, "Correlation Matrix")

    # 7. Box plot for spread analysis
    if cols["best_num"] and cols["best_cat"]:
        bdf = df.copy()
        top_cats = bdf[cols["best_cat"]].value_counts().nlargest(6).index
        bdf = bdf[bdf[cols["best_cat"]].isin(top_cats)]
        fig = px.box(bdf, x=cols["best_cat"], y=cols["best_num"],
                     color=cols["best_cat"], color_discrete_sequence=PALETTE)
        fig.update_traces(boxmean=True)
        charts["box"] = _theme(fig, f"{cols['best_num'].replace('_',' ').title()} Distribution by Category")

    return charts


# ── Phase 3 — Narrative (Pure Python) ────────────────────────────────────────
def generate_narrative(df: pd.DataFrame, clean_report: dict) -> dict:
    cols = detect_columns(df)
    num = cols["numeric"]
    cat = cols["categorical"]
    date = cols["date"]
    narrative = {}

    # ── Executive Summary
    quality = clean_report.get("quality_score", 0)
    quality_label = "Excellent" if quality >= 85 else "Good" if quality >= 65 else "Fair"
    rows, total_cols = clean_report["rows_after"], clean_report["cols_after"]

    exec_lines = [
        f"This dataset comprises **{rows:,} records** across **{total_cols} dimensions** — cleaned and validated with a data quality score of **{quality}% ({quality_label})**.",
        f"During preprocessing, **{clean_report['duplicates_removed']} duplicate entries** were eliminated and **{clean_report['missing_filled']} missing values** were imputed using statistical methods.",
    ]
    if clean_report["outliers_detected"]:
        exec_lines.append(f"**{clean_report['outliers_detected']} outliers** across {len(clean_report['outlier_cols'])} columns were identified and capped using the IQR method to ensure robust analysis.")
    if date:
        exec_lines.append(f"Time-series analysis is enabled through **{len(date)} date column(s)**: {', '.join(date)}.")
    narrative["executive_summary"] = " ".join(exec_lines)

    # ── Numeric Insights
    num_insights = []
    for col in num[:4]:
        series = df[col].dropna()
        mean, med, std = series.mean(), series.median(), series.std()
        skew = float(series.skew())
        top_val = series.max()
        skew_label = "positively skewed (right tail — high outliers exist)" if skew > 0.5 else \
                     "negatively skewed (left tail — low values dominate)" if skew < -0.5 else \
                     "approximately normally distributed"
        pct_above_mean = (series > mean).mean() * 100

        insight = f"**{col.replace('_',' ').title()}**: Mean = {mean:,.2f}, Median = {med:,.2f}, Std Dev = {std:,.2f}. "
        insight += f"The distribution is {skew_label}. "
        insight += f"{pct_above_mean:.1f}% of records exceed the average, with a peak value of {top_val:,.2f}."
        num_insights.append(insight)
    narrative["numeric_insights"] = num_insights

    # ── Category Performance
    cat_insights = []
    for col in cat[:3]:
        vc = df[col].value_counts()
        top_cat, top_count = vc.index[0], int(vc.iloc[0])
        bottom_cat, bottom_count = vc.index[-1], int(vc.iloc[-1])
        total = len(df)
        share = top_count / total * 100
        n_unique = int(vc.shape[0])

        insight = f"**{col.replace('_',' ').title()}** has **{n_unique} unique categories**. "
        insight += f"The dominant category is **'{top_cat}'** with {top_count:,} records ({share:.1f}% share). "
        if n_unique > 1:
            insight += f"The least represented is **'{bottom_cat}'** ({bottom_count:,} records). "
        if num and n_unique <= 20:
            try:
                agg = df.groupby(col)[num[0]].sum()
                top_performer = agg.idxmax()
                top_val = agg.max()
                insight += f"By {num[0].replace('_',' ')}, **'{top_performer}'** leads with {top_val:,.2f}."
            except Exception:
                pass
        cat_insights.append(insight)
    narrative["category_insights"] = cat_insights

    # ── Trend & Correlation Insights
    trend_insights = []

    if len(num) >= 2:
        corr_matrix = df[num].corr()
        pairs = []
        for i in range(len(num)):
            for j in range(i + 1, len(num)):
                c = corr_matrix.iloc[i, j]
                pairs.append((num[i], num[j], c))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        for a, b, c in pairs[:3]:
            if abs(c) >= 0.4:
                strength = "strong" if abs(c) >= 0.7 else "moderate"
                direction = "positive" if c > 0 else "negative"
                trend_insights.append(
                    f"**{a.replace('_',' ').title()}** and **{b.replace('_',' ').title()}** share a {strength} {direction} correlation (r = {c:.2f}). "
                    + (f"As {a.replace('_',' ')} increases, {b.replace('_',' ')} tends to {'increase' if c > 0 else 'decrease'} proportionally."
                       if abs(c) >= 0.6 else "")
                )

    if date and num:
        try:
            ts = df.sort_values(cols["best_date"])[[cols["best_date"], num[0]]].dropna()
            ts = ts.set_index(cols["best_date"]).resample("ME").sum()[num[0]].dropna()
            if len(ts) >= 3:
                x = np.arange(len(ts))
                slope, _, r, p, _ = scipy_stats.linregress(x, ts.values)
                direction = "upward" if slope > 0 else "downward"
                significance = "statistically significant" if p < 0.05 else "not statistically significant"
                trend_insights.append(
                    f"Monthly time-series analysis of **{num[0].replace('_',' ').title()}** reveals a **{direction} trend** (slope = {slope:,.2f}/month, p={p:.3f}). "
                    f"This trend is {significance} at the 95% confidence level."
                )
        except Exception:
            pass

    narrative["trend_insights"] = trend_insights if trend_insights else [
        "Insufficient numeric columns for correlation analysis. Enrich the dataset with more quantitative dimensions."
    ]

    # ── Top / Bottom Performers
    performers = []
    if cols["best_cat"] and cols["best_num"]:
        try:
            agg = df.groupby(cols["best_cat"])[cols["best_num"]].agg(["sum", "mean", "count"])
            agg.columns = ["total", "avg", "count"]
            agg = agg.sort_values("total", ascending=False)
            top3 = agg.head(3)
            bot3 = agg.tail(3)

            top_str = ", ".join([f"**{idx}** ({row['total']:,.2f})" for idx, row in top3.iterrows()])
            bot_str = ", ".join([f"**{idx}** ({row['total']:,.2f})" for idx, row in bot3.iterrows()])

            performers.append(f"🏆 **Top Performers** by {cols['best_num'].replace('_',' ')}: {top_str}")
            performers.append(f"⚠️ **Lowest Performers**: {bot_str} — these segments warrant attention or strategic investment.")
        except Exception:
            pass
    narrative["performers"] = performers

    # ── Statistical Anomalies
    anomalies = []
    for col in num[:3]:
        series = df[col].dropna()
        z = np.abs(scipy_stats.zscore(series))
        extreme = int((z > 3).sum())
        if extreme > 0:
            anomalies.append(f"**{col.replace('_',' ').title()}**: {extreme} data points fall beyond 3 standard deviations — flag for manual review.")
    if not anomalies:
        anomalies.append("No extreme statistical anomalies detected. Dataset passes the Z-score sanity check (no values beyond ±3σ).")
    narrative["anomalies"] = anomalies

    # ── Recommendations
    recs = []
    if clean_report["duplicates_removed"] > rows * 0.05:
        recs.append("🔍 **Duplicate Rate Alert**: Over 5% of records were duplicates. Audit your data ingestion pipeline for deduplication logic.")
    if clean_report["missing_filled"] > rows * total_cols * 0.1:
        recs.append("📋 **High Missingness**: More than 10% of values were missing. Consider improving data collection at the source.")
    if len(num) >= 2:
        recs.append(f"📈 **Predictive Modeling**: With {len(num)} numeric features, this dataset is ready for regression or classification models using `{num[0]}` as the target variable.")
    if cat:
        recs.append(f"🎯 **Segmentation Opportunity**: Leverage `{cat[0]}` for cohort analysis and targeted business strategy.")
    if date:
        recs.append("📅 **Time-Series Forecasting**: Date columns detected. Consider building ARIMA or Prophet forecasting models for future trend prediction.")
    recs.append("🔄 **Refresh Cadence**: Establish a weekly data refresh pipeline to keep insights current and actionable.")
    narrative["recommendations"] = recs

    return narrative


# ── KPI Generator ─────────────────────────────────────────────────────────────
def generate_kpis(df: pd.DataFrame) -> list:
    kpis = []
    for col in df.select_dtypes(include=[np.number]).columns[:4]:
        s = df[col].dropna()
        total = s.sum()
        mean = s.mean()
        # Trend: compare first half vs second half
        half = len(s) // 2
        if half > 0:
            trend = ((s.iloc[half:].mean() - s.iloc[:half].mean()) / (s.iloc[:half].mean() + 1e-9)) * 100
        else:
            trend = 0
        kpis.append({
            "label": col.replace("_", " ").title(),
            "value": f"{total:,.1f}" if total < 1e7 else f"{total/1e6:,.2f}M",
            "avg": f"{mean:,.2f}",
            "trend": round(trend, 1),
            "trend_up": trend >= 0,
        })
    return kpis


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        file = request.files["file"]
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400
        if not _allowed(file.filename):
            return jsonify({"error": "Only CSV and Excel files accepted"}), 400

        uid = uuid.uuid4().hex[:10]
        ext = file.filename.rsplit(".", 1)[1].lower()
        fname = f"{uid}.{ext}"
        raw_path = UPLOAD_DIR / fname
        file.save(raw_path)

        # Load
        df_raw = _load(raw_path)
        original_shape = df_raw.shape

        # Phase 1 — Clean
        df, clean_report = clean_data(df_raw.copy())

        # Save cleaned
        cleaned_path = CLEANED_DIR / f"cleaned_{uid}.csv"
        df.to_csv(cleaned_path, index=False)

        # Phase 2 — Visualizations
        charts = generate_visualizations(df)

        # KPIs
        kpis = generate_kpis(df)

        # Phase 3 — Narrative
        narrative = generate_narrative(df, clean_report)

        # Preview (first 100 rows)
        preview_df = df.head(100).fillna("").astype(str)

        return jsonify({
            "success": True,
            "original_name": file.filename,
            "original_shape": list(original_shape),
            "clean_report": clean_report,
            "charts": charts,
            "kpis": kpis,
            "narrative": narrative,
            "cleaned_file": f"cleaned_{uid}.csv",
            "preview": {
                "columns": preview_df.columns.tolist(),
                "rows": preview_df.values.tolist(),
            },
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/download/<filename>")
def download(filename):
    path = CLEANED_DIR / filename
    if not path.exists():
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
