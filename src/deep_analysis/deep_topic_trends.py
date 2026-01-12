import os
import pandas as pd
import matplotlib.pyplot as plt

# Paths
BASE_DIR = r"C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project"
DATA_PATH = os.path.join(BASE_DIR, r"data\deep_analysis\uber_eats_topics_2_7_24_deep_analysis.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, r"visuals\deep_analysis")

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Topics we are interested in
TARGET_TOPICS = [2, 7, 24]

def plot_topic_trend(df_topic, topic_id, topic_label):
    """
    Plot monthly prevalence over time for a single topic.
    df_topic: dataframe filtered to one topic
    topic_id: numeric topic id
    topic_label: string label for the topic
    """
    # Ensure datetime
    df_topic = df_topic.copy()
    df_topic["at"] = pd.to_datetime(df_topic["at"], errors="coerce")
    df_topic = df_topic.dropna(subset=["at"])

    if df_topic.empty:
        print(f"No valid dates for topic {topic_id}, skipping plot.")
        return

    # Group by month
    df_topic["month"] = df_topic["at"].dt.to_period("M").dt.to_timestamp()
    monthly_counts = (
        df_topic.groupby("month")
        .size()
        .reset_index(name="count")
        .sort_values("month")
    )

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_counts["month"], monthly_counts["count"], marker="o")
    plt.xlabel("Month")
    plt.ylabel("Number of reviews")
    plt.title(f"Monthly prevalence over time - Topic {topic_id}: {topic_label}")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Safe filename: remove characters that can cause issues
    safe_label = "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in topic_label)
    safe_label = "_".join(safe_label.split())  # replace spaces with underscores

    filename = f"topic_{topic_id}_monthly_trend_{safe_label}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(output_path)
    plt.close()

    print(f"Saved trend plot for topic {topic_id} to:")
    print(f"  {output_path}")


def main():
    # Load deep analysis subset
    df = pd.read_csv(DATA_PATH)

    # Ensure topic is numeric
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")

    # Optional: sanity check
    available_topics = sorted(df["topic"].dropna().unique())
    print("Available topics in deep analysis file:", available_topics)

    # Loop through each target topic and plot
    for topic_id in TARGET_TOPICS:
        df_topic = df[df["topic"] == topic_id].copy()
        if df_topic.empty:
            print(f"No reviews found for topic {topic_id}, skipping.")
            continue

        # Use the first non-null pain_point_label as label
        topic_label = df_topic["pain_point_label"].dropna().iloc[0] if "pain_point_label" in df_topic.columns else f"Topic {topic_id}"

        print(f"\nProcessing topic {topic_id}: {topic_label}")
        print(f"Number of reviews: {len(df_topic)}")

        plot_topic_trend(df_topic, topic_id, topic_label)


if __name__ == "__main__":
    main()
