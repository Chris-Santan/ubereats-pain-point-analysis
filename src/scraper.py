import pandas as pd
from google_play_scraper import reviews, Sort

APP_ID = "com.ubercab.eats"

def fetch_reviews(count=80000):
    result, _ = reviews(
        APP_ID,
        lang='en',
        country='us',
        sort=Sort.NEWEST,
        count=count
    )
    df = pd.DataFrame(result)
    df.to_csv("data/raw/raw_batch.csv", index=False)
    print(f"Saved {len(df)} samples reviews.")

if __name__ == "__main__":
    fetch_reviews(80000)