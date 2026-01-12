import pandas as pd
import re
from pathlib import Path

INPUT_PATH = Path("data/processed/cleaned_batch.csv")
OUTPUT_PATH = Path("data/processed/preprocessed_cleaned_batch.csv")


def clean_text(text: str) -> str:
    """Normalize review text for embeddings."""
    # ensure string
    text = str(text)

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # remove all characters that are not letters, numbers, or whitespace
    # this strips punctuation, emojis, and most other symbols
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    print(f"Loading cleaned review sample from {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)

    print("Applying text preprocessing (lowercase, remove URLs/punctuation/emojis, normalize spaces)...")
    df["processed_content"] = df["content"].fillna("").astype(str).apply(clean_text)

    # drop rows where processed_content ended up empty
    before_len = len(df)
    df = df[df["processed_content"] != ""].copy()
    after_len = len(df)
    print(f"Removed {before_len - after_len} rows with empty processed text.")

    # enforce >= 6 words *after* cleaning, just in case cleaning shortened some
    df["processed_word_count"] = df["processed_content"].str.split().str.len()
    before_len = len(df)
    df = df[df["processed_word_count"] >= 6].copy()
    after_len = len(df)
    print(f"Removed {before_len - after_len} rows with < 6 words after cleaning.")

    # drop duplicates on processed text
    before_len = len(df)
    df = df.drop_duplicates(subset=["processed_content"])
    after_len = len(df)
    print(f"Removed {before_len - after_len} duplicate reviews based on processed_content.")

    # save and drop helper column
    df = df.drop(columns=["processed_word_count"])
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved preprocessed dataset to {OUTPUT_PATH}")
    print(f"Final row count: {len(df)}")


if __name__ == "__main__":
    main()
