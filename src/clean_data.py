import pandas as pd
from langdetect import detect, LangDetectException
from pathlib import Path

RAW_SAMPLE_PATH = Path("data/raw/raw_batch.csv")
OUT_PATH = Path("data/processed/cleaned_batch.csv")


def is_english(text: str) -> bool:
    """Return True if the text is detected as English, False otherwise."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def word_count(text: str) -> int:
    """Return number of words in the text."""
    return len(text.split())


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to 1–3 star, non-empty, English reviews with >= 6 words."""
    
    # Convert content to string and strip whitespace
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    
    # Filter by star rating (1–3)
    mask_score = df["score"].between(1, 3)
    
    # Filter out empty content
    mask_non_empty = df["content"] != ""
    
    # Filter by English language
    mask_english = df["content"].apply(is_english)
    
    # Filter out reviews with fewer than 6 words
    mask_word_count = df["content"].apply(word_count) >= 6
    
    cleaned = df[mask_score & mask_non_empty & mask_english & mask_word_count].copy()
    return cleaned


def main():
    print(f"Loading raw sample from {RAW_SAMPLE_PATH}...")
    df = pd.read_csv(RAW_SAMPLE_PATH)

    print(f"Raw reviews: {len(df)}")
    cleaned = clean_reviews(df)
    print(f"Filtered reviews (1–3 star, non-empty, English, >=6 words): {len(cleaned)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned reviews to {OUT_PATH}")


if __name__ == "__main__":
    main()
