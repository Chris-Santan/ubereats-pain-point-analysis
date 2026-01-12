import os
import pandas as pd

# Paths (adjust if your folder structure changes)
LABELED_PATH = r"C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\data\processed\uber_eats_bertopic_labeled.csv"
OUTPUT_DIR = r"C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project\data\deep_analysis"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "uber_eats_topics_2_7_24_deep_analysis.csv")

# Make sure the output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Topics you want to keep
TARGET_TOPICS = {2, 7, 24}

def main():
    # Load the full labeled review dataset
    df = pd.read_csv(LABELED_PATH)

    # Ensure topic column is numeric (sometimes it can be read as string)
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")

    # Filter to only the chosen topics
    df_filtered = df[df["topic"].isin(TARGET_TOPICS)].copy()

    # (Optional) sort by topic and date to make it easier to inspect
    if "at" in df_filtered.columns:
        df_filtered["at"] = pd.to_datetime(df_filtered["at"], errors="coerce")
        df_filtered = df_filtered.sort_values(["topic", "at"])
    else:
        df_filtered = df_filtered.sort_values("topic")

    # Save to CSV without index
    df_filtered.to_csv(OUTPUT_PATH, index=False)

    # Print some quick stats so you can see what happened
    print(f"Saved {len(df_filtered)} reviews to:")
    print(f"  {OUTPUT_PATH}")
    print("\nCounts per topic:")
    print(df_filtered["topic"].value_counts().sort_index())

if __name__ == "__main__":
    main()
