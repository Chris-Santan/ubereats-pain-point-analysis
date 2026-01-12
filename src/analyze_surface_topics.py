import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# Paths
INPUT_PATH = Path("data/processed/uber_eats_bertopic_labeled.csv")
OUTPUT_STATS_PATH = Path("data/processed/topic_summary_stats.csv")
OUTPUT_TRENDS_PATH = Path("data/processed/topic_trends_monthly.csv")
VISUALS_PATH = Path("visuals")

def main():
    print("Loading labeled review data...")
    df = pd.read_csv(INPUT_PATH)

    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df.get("at")):
        df["at"] = pd.to_datetime(df["at"], errors="coerce")
    df = df.dropna(subset=["at"])

    # A. Prevalence
    prevalence = (
        df.groupby("pain_point_label")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    total = prevalence["count"].sum()
    prevalence["percent"] = prevalence["count"] / total * 100

    OUTPUT_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    prevalence.to_csv(OUTPUT_STATS_PATH, index=False)
    print(f"Saved prevalence stats to {OUTPUT_STATS_PATH}")

    # Create visuals directory
    VISUALS_PATH.mkdir(parents=True, exist_ok=True)

    # Plot 1: Prevalence bar chart (all topics)
    plt.figure(figsize=(12, 6))
    plt.barh(prevalence["pain_point_label"], prevalence["count"])
    plt.xlabel("Number of Reviews")
    plt.title("Prevalence of Pain Point Categories (All Topics)")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(VISUALS_PATH / "prevalence_by_pain_point.png", dpi=300)
    plt.close()

    # >>> NEW: Plot 1b – Top 15 most common topics
    top15 = prevalence.head(15).copy()
    plt.figure(figsize=(12, 6))
    plt.barh(top15["pain_point_label"], top15["count"])
    plt.xlabel("Number of Reviews")
    plt.title("Top 15 Most Common Pain Point Categories")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(VISUALS_PATH / "prevalence_top15_by_pain_point.png", dpi=300)
    plt.close()
    # <<< END NEW

    # B. Temporal trends
    df["month"] = df["at"].dt.to_period("M")
    trends = df.groupby(["month", "pain_point_label"]).size().reset_index(name="count")
    trends["month"] = trends["month"].dt.to_timestamp()
    trends.to_csv(OUTPUT_TRENDS_PATH, index=False)
    print(f"Saved monthly trends to {OUTPUT_TRENDS_PATH}")

    trend_pivot = trends.pivot(index="month", columns="pain_point_label", values="count").fillna(0)

    # Plot 2: Time series
    plt.figure(figsize=(14, 6))
    for col in trend_pivot.columns:
        plt.plot(trend_pivot.index, trend_pivot[col], label=col)
    plt.xlabel("Month")
    plt.ylabel("Number of Reviews")
    plt.title("Monthly Complaint Trends by Pain Point Category")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(VISUALS_PATH / "monthly_trends_by_pain_point.png", dpi=300)
    plt.close()

    print(f"Saved all graphs to {VISUALS_PATH}")

if __name__ == "__main__":
    main()
