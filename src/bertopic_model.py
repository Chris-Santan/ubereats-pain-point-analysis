import pandas as pd
from pathlib import Path
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

# adjust this to your actual preprocessed file name
INPUT_PATH = Path("data/processed/preprocessed_cleaned_batch.csv")
TOPIC_ASSIGNMENTS_PATH = Path("data/processed/uber_eats_bertopic_topics.csv")
TOPIC_INFO_PATH = Path("data/processed/uber_eats_bertopic_topic_info.csv")


def main():
    print(f"Loading preprocessed reviews from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    texts = df["processed_content"].astype(str).tolist()
    print(f"Number of reviews: {len(texts)}")

    # Same embedding model family as before
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Better vectorizer for topic words
    vectorizer_model = CountVectorizer(
        stop_words="english",      # remove common stopwords
        ngram_range=(1, 2),        # unigrams + bigrams
        min_df=3                   # ignore super-rare terms
    )

    print("Fitting BERTopic model with custom vectorizer...")
    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        verbose=True,
        min_topic_size=10,         # allow more, smaller topics
        top_n_words=10             # show 10 words per topic
    )

    topics, probs = topic_model.fit_transform(texts)

    # Save topic assignments per review
    df["topic"] = topics
    TOPIC_ASSIGNMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TOPIC_ASSIGNMENTS_PATH, index=False)
    print(f"Saved topic assignments to {TOPIC_ASSIGNMENTS_PATH}")

    # Save topic summary info
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(TOPIC_INFO_PATH, index=False)
    print(f"Saved topic summary info to {TOPIC_INFO_PATH}")

    # Save the model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    topic_model.save(models_dir / "uber_eats_bertopic")
    print(f"Saved BERTopic model to {models_dir / 'uber_eats_bertopic'}")


if __name__ == "__main__":
    main()
