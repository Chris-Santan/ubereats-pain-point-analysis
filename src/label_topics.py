from pathlib import Path
import ast
import pandas as pd

# Paths aligned with bertopic_model.py
TOPIC_INFO_PATH = Path("data/processed/uber_eats_bertopic_topic_info.csv")
INPUT_TOPICS_PATH = Path("data/processed/uber_eats_bertopic_topics.csv")
OUTPUT_LABELED_PATH = Path("data/processed/uber_eats_bertopic_labeled.csv")


def build_topic_map(topic_info_df: pd.DataFrame) -> dict:
    """
    Build a mapping from numeric topic id -> human-readable label,
    using only the top 3 words from the Representation field.
    No assumptions beyond those words.
    """
    def label_from_rep(rep_str: str) -> str:
        # rep_str is like "['food', 'cold', 'driver', ...]"
        try:
            tokens = ast.literal_eval(rep_str)
            tokens = [str(t) for t in tokens]
        except Exception:
            return "general issues"

        # keep up to 3 top words
        tokens = tokens[:3]
        if not tokens:
            return "general issues"

        if len(tokens) == 1:
            core = tokens[0]
        elif len(tokens) == 2:
            core = f"{tokens[0]} and {tokens[1]}"
        else:
            core = f"{tokens[0]}, {tokens[1]} and {tokens[2]}"

        # purely descriptive, no extra inference
        return f"{core} related issues"

    topic_map = {}
    for _, row in topic_info_df.iterrows():
        topic_id = int(row["Topic"])
        rep = row["Representation"]
        topic_map[topic_id] = label_from_rep(rep)

    return topic_map


def main():
    print(f"Loading topic info from {TOPIC_INFO_PATH}...")
    topic_info_df = pd.read_csv(TOPIC_INFO_PATH)

    # Build automatic labels for all topics (including -1)
    topic_map = build_topic_map(topic_info_df)
    print(f"Built labels for {len(topic_map)} topics.")
    print("Sample labels:")
    for k in sorted(list(topic_map.keys()))[:10]:
        print(f"  Topic {k}: {topic_map[k]}")

    print(f"\nLoading per-review topic assignments from {INPUT_TOPICS_PATH}...")
    df = pd.read_csv(INPUT_TOPICS_PATH)

    # Detect the topic column name: 'topic' (from your bertopic_model.py) or 'Topic'
    if "topic" in df.columns:
        topic_col = "topic"
    elif "Topic" in df.columns:
        topic_col = "Topic"
    else:
        raise ValueError(
            f"Expected a 'topic' or 'Topic' column in {INPUT_TOPICS_PATH}, "
            f"but found columns: {list(df.columns)}"
        )

    # Map numeric topic id -> human-readable label
    df["pain_point_label"] = df[topic_col].map(topic_map)

    # Save labeled file
    OUTPUT_LABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_LABELED_PATH, index=False)
    print(f"Saved labeled reviews to {OUTPUT_LABELED_PATH}")


if __name__ == "__main__":
    main()
