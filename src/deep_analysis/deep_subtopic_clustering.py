import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = r"C:\Users\NutSplitter\Desktop\Uber Eats Pain Point Project"
INPUT_PATH = os.path.join(BASE_DIR, r"data\deep_analysis\uber_eats_topics_2_7_24_deep_analysis.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, r"data\deep_analysis")

# topics you are deep diving
MAIN_TOPICS = [2, 7, 24]

def build_topic_labels(topic_model):
    """
    Build a map: subtopic_id -> short human readable label
    based on the first 3 representative words.
    """
    info = topic_model.get_topic_info()
    label_map = {}

    for _, row in info.iterrows():
        topic_id = row["Topic"]
        # skip outlier topic -1 if it exists
        if topic_id == -1:
            continue

        rep = row["Representation"]
        # rep is like "['word1', 'word2', 'word3', ...]" or list depending on version
        if isinstance(rep, str):
            # crude but robust parse: strip brackets and split on commas
            cleaned = rep.strip("[]")
            words = [w.strip().strip("'\"") for w in cleaned.split(",")]
        else:
            words = list(rep)

        top_words = [w for w in words if w][:3]
        if not top_words:
            label = f"subtopic_{topic_id}"
        else:
            if len(top_words) == 1:
                label = f"{top_words[0]} related issues"
            elif len(top_words) == 2:
                label = f"{top_words[0]} and {top_words[1]} related issues"
            else:
                label = f"{top_words[0]}, {top_words[1]} and {top_words[2]} related issues"

        label_map[topic_id] = label

    return label_map

def cluster_subtopics_for_topic(df, main_topic_id, embedding_model):
    """
    Run BERTopic subtopic clustering for a single main topic.
    Returns the df for that topic with added subtopic_id and subtopic_label columns.
    """
    df_topic = df[df["topic"] == main_topic_id].copy()
    df_topic = df_topic.dropna(subset=["processed_content"])

    if df_topic.empty:
        print(f"No reviews found for topic {main_topic_id}, skipping.")
        return df_topic

    texts = df_topic["processed_content"].astype(str).tolist()

    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=8,   # smaller to find finer subtopics
        top_n_words=10,
        verbose=True
    )

    print(f"\nFitting subtopic model for main topic {main_topic_id} on {len(texts)} reviews...")
    subtopics, probs = topic_model.fit_transform(texts)

    df_topic["subtopic_id"] = subtopics

    # build labels
    label_map = build_topic_labels(topic_model)
    df_topic["subtopic_label"] = df_topic["subtopic_id"].map(label_map).fillna("other / outlier")

    # save per topic file
    filename = f"topic_{main_topic_id}_subtopics.csv"
    out_path = os.path.join(OUTPUT_DIR, filename)
    df_topic.to_csv(out_path, index=False)
    print(f"Saved subtopics for main topic {main_topic_id} to:")
    print(f"  {out_path}")

    # quick summary
    print("\nSubtopic counts:")
    print(df_topic["subtopic_label"].value_counts())

    return df_topic

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    df["topic"] = pd.to_numeric(df["topic"], errors="coerce")

    # load embedding model once
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    dfs_with_subtopics = []

    for main_topic_id in MAIN_TOPICS:
        df_topic_with_sub = cluster_subtopics_for_topic(df, main_topic_id, embedding_model)
        dfs_with_subtopics.append(df_topic_with_sub)

    # merge back into one combined file
    if dfs_with_subtopics:
        combined = pd.concat(dfs_with_subtopics, axis=0)
        # keep any rows not in 2, 7, 24 unchanged (optional)
        others = df[~df["topic"].isin(MAIN_TOPICS)].copy()
        combined_full = pd.concat([combined, others], axis=0)

        out_combined = os.path.join(
            OUTPUT_DIR,
            "uber_eats_deep_analysis_topics_2_7_24_with_subtopics.csv"
        )
        combined_full.to_csv(out_combined, index=False)
        print("\nSaved combined file with subtopics to:")
        print(f"  {out_combined}")

if __name__ == "__main__":
    main()
