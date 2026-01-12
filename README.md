# Uber Eats Pain Point Analysis

## Project Overview

This project performs large-scale analysis of English language Uber Eats reviews sourced from the Google Play Store. The goal is to identify recurring negative user pain points at scale, reduce them into meaningful categories, and produce actionable summaries using modern NLP tooling. 

The dataset contained approximately 80,000 reviews after scraping, which was reduced to 38,120 eligible negative reviews after filtering for sentiment and length. All analysis was performed locally using reproducible Python scripts. The final outcome is a structured breakdown of major complaint categories supported by topic modeling, clustering, visualization, and human validation.

---

## Table of Contents

1. [Data Collection & Filtering](#data-collection--filtering)
2. [NLP Processing Pipeline](#nlp-processing-pipeline)
3. [Topic Modeling & Clustering](#topic-modeling--clustering)
4. [Deep Analysis Results](#deep-analysis-results)
5. [Visualization](#visualization)
6. [Repository Structure](#repository-structure)
7. [Technical Stack](#technical-stack)
8. [Usage Instructions](#usage-instructions)
9. [Current Scope and Future Work](#current-scope-and-future-work)

---

## Data Collection & Filtering

### Scraping Review Data

Reviews were scraped using a custom scraper (`src/scraper.py`) which collected approximately 80,000 Uber Eats reviews from the Google Play Store. Only English language reviews were collected because they are the most consistent for downstream embedding models and because sentiment heuristics tend to perform better.

### Filtering Criteria

A cleaning script (`src/clean_data.py`) removed neutral or positive reviews and filtered for meaningful content. The following filters were applied:

- Rating less than 4 stars
- Minimum length of 7 words
- English text only (using langdetect)

After filtering, the dataset contained approximately 38,120 eligible negative reviews suitable for NLP analysis. This approach ensured that downstream topic extraction would focus on user complaints rather than general praise or noise.

---

## NLP Processing Pipeline

### Text Preprocessing

Standard NLP preprocessing was applied (`src/preprocess.py`) including:
- Lowercasing
- Punctuation removal
- Stopword removal
- Normalization of whitespace

This stage was designed to reduce noise without damaging semantic meaning for transformer-based embeddings.

### Topic Modeling with BERTopic

BERTopic (`src/bertopic_model.py`) was used as the primary tool to extract coherent topic clusters from the filtered dataset. The pipeline produced:

- Sentence Transformers based embeddings (using `all-MiniLM-L6-v2`)
- HDBSCAN clustering
- UMAP dimensionality reduction
- Topic labeling based on word frequency and TF-IDF

The model configuration:
- Embedding model: `all-MiniLM-L6-v2` (SentenceTransformer)
- Vectorizer: CountVectorizer with unigrams and bigrams, min_df=3
- Minimum topic size: 10 reviews
- Top N words per topic: 10

The initial BERTopic pass produced dozens of interpretable topics including complaints about promotions, fees, restaurant issues, account issues, refunds, and delivery problems.

### Topic Labeling

Human-readable labels were assigned to topics (`src/label_topics.py`) based on the most representative words and manual review. Topics were then analyzed for surface-level patterns (`src/analyze_surface_topics.py`).

---

## Topic Modeling & Clustering

### Subtopic Discovery

To further resolve granular complaints, additional clustering was performed inside major topics (`src/deep_analysis/deep_subtopic_clustering.py`). For example, the promotion-related topic was decomposed into subtopics such as unusable promo codes, expiring promotions, fake discounts, and eligibility errors.

The subtopic clustering process:
- Applied BERTopic iteratively to each major topic
- Used smaller minimum topic size (8 reviews) to find finer-grained subtopics
- Generated human-readable subtopic labels based on top representative words
- Saved subtopic assignments and labels for each review

### Human in the Loop Validation

After clustering, representative samples from selected subtopics were reviewed manually to ensure semantic correctness. Human validation helped interpret raw cluster labels into meaningful business-level pain points. Examples of validated interpretations include:

- Unauthorized or incorrect tip charges
- Promotion codes being expired or invalid
- Account suspensions tied to gift card redemptions

This hybrid method aligns unsupervised ML with practical business insights.

---

## Deep Analysis Results

Three major pain point categories were selected for deep analysis: Topic 2 (Promotions), Topic 7 (Tipping), and Topic 24 (Gift Cards). Each topic underwent detailed subtopic clustering and manual analysis.

### Topic 2: Promo, Promotion and Promos Related Issues

**Total Reviews Analyzed:** 155 reviews

**Key Findings:**

1. **Broken or Unreliable Promo Codes** (56% - 87 reviews)
   - Promo codes simply do not work when applied
   - Users enter codes and get generic errors like "cannot be applied," "invalid," or "doesn't work"
   - New-user codes that never work on "first order" despite it being the first order
   - Email/postcard/in-app promos that cannot be claimed at checkout
   - App shows promo as available in UI, but fails at checkout

2. **Promo Codes Revoked, Removed, or Expiring Early** (18% - 28 reviews)
   - Code exists and may show as applied, but is then taken away
   - Code active when starting order, but "deactivated" or "cancelled" during order placement
   - Promo disappears at checkout or after basket modification
   - Codes disappear from account before published end date

3. **Promo Marked as Already Used** (4% - 6 reviews)
   - Users never used the code but get "already used" or "max redemptions reached" errors
   - Especially damaging for direct mail/postcard campaigns and referral codes

4. **Promo Appears Applied but User Charged Full Price** (9% - 14 reviews)
   - Promo looks accepted in UI, but final charge is full amount with no discount
   - Users only realize after checking bank statements
   - Strong language (scam, theft) due to perceived deception

5. **Not Enough Promos / Stopped Receiving Them** (5% - 8 reviews)
   - Promo strategy and fairness complaints
   - Users stopped receiving promos after complaints
   - Perceived favoritism toward new customers

6. **Eligibility and Account Restrictions** (3% - 5 reviews)
   - Users blocked from promos due to vague or unexplained restrictions
   - "Your account is not eligible for new promotion codes" on brand-new accounts
   - Messages about "irregular activities" after trying to use promo

7. **UX / Process Complaints** (4% - 6 reviews)
   - Too many steps to complete order with promo
   - Cannot see where to enter code
   - Confusing visual design for promo tags

**Monthly Trend Visualization:** `visuals/deep_analysis/topic_2_monthly_trend_promo__promotion_and_promos_related_issues.png`

This visualization shows how promotion-related complaints have evolved over time, revealing patterns in when users experience the most issues with promo codes.

### Topic 7: Tip, Tips and Tipping Related Issues

**Total Reviews Analyzed:** 70 reviews

**Key Findings:**

1. **Double, Triple, or Unauthorized Tip/Charge Issues** (36% - 25 reviews)
   - Tip charged twice, order charged multiple times
   - Tip taken after $0 + cash tip
   - PayPal double charges, surprise extra charges after the fact
   - Most call it "theft" or "scam"
   - Often mention having to dispute charges with bank or being refused refund by support

2. **Forced or Manipulative Tipping and High Default Tips** (20% - 14 reviews)
   - Tip added by default, cannot place order without minimum tip
   - App choosing higher percentages (e.g., switching 15% to 18%)
   - Constant pressure to tip, feeling forced to tip instead of using cash
   - App enforces tip levels, pushes large tips (e.g., $7 on $20 order)
   - Quietly changes selected percentage after delivery

3. **Cannot Edit, Add, or Remove Tip** (20% - 14 reviews)
   - User wants to change tip (up or down), remove mistaken tip, or add tip for driver later
   - App flow does not allow it or option is missing
   - One hour edit window not actually usable
   - No way to lower tip for poor service
   - Only preset percentages with no real control

4. **Tip UI Bugs or Glitches** (9% - 6 reviews)
   - Custom tip field not working
   - Keyboard not responding
   - Screen disappearing when trying to tip
   - Tip UI button doing nothing
   - Scrolling bugs that prevent saving

5. **Nontransparent Totals, Tip Amounts, or Calculators** (7% - 5 reviews)
   - Cannot see full total with tip
   - Percentage shown without dollar amount
   - Tip calculator "wrong" or confusing holds and totals
   - Users have to calculate grand total outside app

6. **General High Cost or Fee Complaints** (7% - 5 reviews)
   - Overall cost is "criminal"
   - Service fees plus tip feel excessive
   - Discounts do not reduce total
   - App makes orders far more expensive than food value

**Monthly Trend Visualization:** `visuals/deep_analysis/topic_7_monthly_trend_tip__tips_and_tipping_related_issues.png`

This visualization tracks tipping-related complaints over time, showing when users experience the most issues with tip charges and tip functionality.

### Topic 24: Gift, Gift Card and Card Related Issues

**Total Reviews Analyzed:** 110 reviews

**Key Findings:**

1. **Gift Card Credit is Unusable / Stuck** (25% - 28 reviews)
   - "Have a gift card I can't use"
   - Balance shows in account but not as payment option
   - Order will not process when GC is selected
   - GC only works for some services / not in user's context

2. **Verification, Bans, and Eligibility Blockers** (25% - 28 reviews)
   - Forced to upload driver's license and selfie to use a GC
   - Accounts suspended or banned right after loading a GC
   - Under 18 / family account restrictions making GC balance unusable
   - Need "Uber Money account" or extra payment method to unlock GC usage
   - Country / currency ineligibility (e.g., USD card but user in CHF region)

3. **Gift Card Redemption / Loading Failures** (14% - 15 reviews)
   - Cannot redeem or add gift card number
   - App saying brand new GC is "already redeemed"
   - Scratched code situation where support gives code that is then "already used"
   - App not accepting gift card online or on website; sometimes forced into app
   - GC option/menu missing entirely

4. **UX Complexity / Painful Process / Too Many Hoops** (13% - 14 reviews)
   - "Took an hour to use a gift card"
   - Forced to install/uninstall app multiple times just to see GC option
   - "Most traumatic experience" or "absolute joke" onboarding just to redeem
   - Generally able to eventually use card, but flow feels hostile and exhausting

5. **Wrong Charges, Missing Value, or Suspicious Charge Behavior** (10% - 11 reviews)
   - GC charged but order not placed
   - Bank charged instead of GC, or both charged
   - Value partially missing after load ("doesn't load full amount", money disappeared)
   - Fraud-like charges reported when scanning a GC
   - Gift balance taken after cancellation and not restored

6. **No Proper Split Between Gift Card and Other Payment Methods** (5% - 5 reviews)
   - Must match order total exactly with GC or order will not go through
   - Cannot use GC for part of bill and card for the rest
   - Forced to drain whole GC and then pay extra on card
   - GC only works if it covers full cost

7. **High Costs Where Gift Card is Just the Reason to Use the App** (5% - 5 reviews)
   - "Only used because I had a gift card"
   - Complaints about astronomical fees and prices, with GC as only justification to try service

8. **Gifting Flow Issues** (4% - 4 reviews)
   - Gift not delivered or recipient not contacted
   - Buying digital gift cards failing
   - Complaints that gifter is contacted instead of recipient
   - Desire to see gift code beforehand or schedule properly

**Executive Summary:** Roughly half of the reviews (50%) say some version of "I have a gift card balance, but the app will not let me use it because of either technical failure or gating requirements like ID checks, bans, or eligibility rules." About a quarter (27%) focus on how confusing, fragile, or broken the redemption and usage flow is. Around 10% explicitly describe money seeming to vanish, double charges, or unclear allocation of payment.

**Monthly Trend Visualization:** `visuals/deep_analysis/topic_24_monthly_trend_gift__gift_card_and_card_related_issues.png`

This visualization shows how gift card-related complaints have changed over time, revealing patterns in when users experience the most issues with gift card functionality.

---

## Visualization

The project generates several types of visual outputs:

### Topic-Level Visualizations

- **Prevalence by Pain Point** (`visuals/prevalence_by_pain_point.png`): Bar chart showing the frequency of each major pain point category
- **Top 15 Prevalence by Pain Point** (`visuals/prevalence_top15_by_pain_point.png`): Focused view on the top 15 most common complaint categories
- **Monthly Trends by Pain Point** (`visuals/monthly_trends_by_pain_point.png`): Temporal analysis showing how different pain points have evolved over time

### Deep Analysis Visualizations

Monthly trend visualizations for each of the three deeply analyzed topics:

- **Topic 2 - Promotions**: `visuals/deep_analysis/topic_2_monthly_trend_promo__promotion_and_promos_related_issues.png`
- **Topic 7 - Tipping**: `visuals/deep_analysis/topic_7_monthly_trend_tip__tips_and_tipping_related_issues.png`
- **Topic 24 - Gift Cards**: `visuals/deep_analysis/topic_24_monthly_trend_gift__gift_card_and_card_related_issues.png`

These visualizations show how each subtopic's prevalence has changed over time, allowing identification of:
- Seasonal patterns in complaints
- Impact of app updates or policy changes
- Trends that may indicate systemic issues
- Effectiveness of fixes or improvements

### Temporal Analysis

Temporal trend analysis (`src/deep_analysis/deep_topic_trends.py`) groups reviews by month and tracks the number of reviews per month for each topic. This enables identification of:
- Spikes in complaints following specific events
- Gradual increases or decreases in complaint frequency
- Correlation between complaint trends and external factors

---

## Repository Structure

```
Uber Eats Pain Point Project/
├── data/
│   ├── raw/                    # Raw scraped review data
│   ├── processed/              # Cleaned and preprocessed data
│   │   ├── preprocessed_cleaned_batch.csv
│   │   ├── uber_eats_bertopic_topics.csv
│   │   ├── uber_eats_bertopic_topic_info.csv
│   │   └── ...
│   └── deep_analysis/          # Deep analysis data for topics 2, 7, 24
│       ├── topic_2_subtopics.csv
│       ├── topic_7_subtopics.csv
│       ├── topic_24_subtopics.csv
│       └── uber_eats_topics_2_7_24_deep_analysis.csv
│
├── src/
│   ├── scraper.py             # Web scraping for Google Play reviews
│   ├── clean_data.py           # Filtering and cleaning raw data
│   ├── preprocess.py           # Text preprocessing (lowercase, stopwords, etc.)
│   ├── bertopic_model.py       # Main BERTopic topic modeling
│   ├── label_topics.py         # Human-readable topic labeling
│   ├── analyze_surface_topics.py  # Surface-level topic analysis
│   ├── filters.py              # Data filtering utilities
│   ├── utils.py                # General utilities
│   ├── config.py               # Configuration constants
│   └── deep_analysis/
│       ├── deep_subtopic_clustering.py  # Subtopic clustering for major topics
│       ├── deep_topic_trends.py         # Temporal trend analysis
│       └── filter_topics.py             # Filter topics for deep analysis
│
├── models/                     # Saved BERTopic models (local only, not in Git)
│   └── uber_eats_bertopic/
│
├── results/                    # Analysis results and filtered reviews
│   ├── topic_2/               # Promo code analysis results
│   │   ├── filtered_reviews.csv
│   │   ├── review_content_only.csv
│   │   └── analysis
│   ├── topic_7/               # Tipping analysis results
│   │   ├── filtered_reviews.csv
│   │   ├── review_content_only.csv
│   │   └── analysis
│   └── topic_24/              # Gift card analysis results
│       ├── filtered_reviews.csv
│       ├── review_content_only.csv
│       └── analysis
│
├── visuals/                    # Generated visualizations
│   ├── prevalence_by_pain_point.png
│   ├── prevalence_top15_by_pain_point.png
│   ├── monthly_trends_by_pain_point.png
│   └── deep_analysis/
│       ├── topic_2_monthly_trend_promo__promotion_and_promos_related_issues.png
│       ├── topic_7_monthly_trend_tip__tips_and_tipping_related_issues.png
│       └── topic_24_monthly_trend_gift__gift_card_and_card_related_issues.png
│
├── filter_reviews_by_subtopic.py  # Filter reviews by subtopic label
├── extract_review_content.py      # Extract review content only
├── requirements.txt               # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## Technical Stack

### Core Libraries

- **Python 3.x**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Sentence Transformers**: Text embeddings (`all-MiniLM-L6-v2`)
- **BERTopic**: Topic modeling and clustering
- **Scikit-learn**: Machine learning utilities (HDBSCAN, CountVectorizer)
- **UMAP**: Dimensionality reduction for visualization
- **Matplotlib / Seaborn**: Data visualization
- **BeautifulSoup**: Web scraping utilities
- **langdetect**: Language detection for filtering
- **tqdm**: Progress bars for long-running operations

### Model Details

- **Embedding Model**: `all-MiniLM-L6-v2` (SentenceTransformer)
  - 384-dimensional embeddings
  - Optimized for semantic similarity tasks
  - Fast inference suitable for large-scale analysis

- **Clustering Algorithm**: HDBSCAN (via BERTopic)
  - Density-based clustering
  - Handles variable density clusters
  - Automatically identifies number of clusters

### Data Processing

- **Input**: ~80,000 raw reviews from Google Play Store
- **Filtered**: ~38,120 negative reviews (rating < 4, length >= 7 words, English only)
- **Topics Identified**: Dozens of interpretable complaint categories
- **Deep Analysis**: 3 major topics with detailed subtopic breakdowns

---

## Usage Instructions

### Prerequisites

1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

The analysis pipeline consists of several sequential steps:

1. **Scrape Reviews** (if needed):
   ```bash
   python src/scraper.py
   ```

2. **Clean and Filter Data**:
   ```bash
   python src/clean_data.py
   ```

3. **Preprocess Text**:
   ```bash
   python src/preprocess.py
   ```

4. **Run Topic Modeling**:
   ```bash
   python src/bertopic_model.py
   ```

5. **Label Topics**:
   ```bash
   python src/label_topics.py
   ```

6. **Analyze Surface Topics**:
   ```bash
   python src/analyze_surface_topics.py
   ```

7. **Deep Analysis - Filter Topics**:
   ```bash
   python src/deep_analysis/filter_topics.py
   ```

8. **Deep Analysis - Subtopic Clustering**:
   ```bash
   python src/deep_analysis/deep_subtopic_clustering.py
   ```

9. **Deep Analysis - Temporal Trends**:
   ```bash
   python src/deep_analysis/deep_topic_trends.py
   ```

### Utility Scripts

**Filter Reviews by Subtopic:**
```bash
python filter_reviews_by_subtopic.py
```
This script filters reviews from specific topics by subtopic label and saves them to the results directory.

**Extract Review Content Only:**
```bash
python extract_review_content.py
```
This script extracts only the review content column from filtered review files, creating numbered review lists.

---

## Current Scope and Future Work

### Current Scope Completed

- Data scraping from Google Play Store
- Sentiment-based filtering (negative reviews only)
- Text preprocessing and normalization
- Embedding generation using Sentence Transformers
- Topic modeling with BERTopic
- Subtopic clustering for major pain points
- Temporal trend analysis and visualization
- Human validation and interpretation
- Detailed analysis reports for three major topics
- Monthly trend visualizations

### Potential Future Extensions

1. **Temporal Trend Analysis Enhancement**
   - Correlation analysis with app version releases
   - Seasonal pattern identification
   - Predictive modeling for complaint trends

2. **Cross-Platform Comparison**
   - iOS vs Android complaint patterns
   - Platform-specific pain point identification
   - Comparative analysis of user experiences

3. **Geographic Segmentation**
   - Regional complaint patterns
   - Country-specific pain points
   - Cultural factors in complaint language

4. **Automated Pain Point Summarization**
   - LLM-based summary generation
   - Automated report creation
   - Natural language summaries of clusters

5. **Sentiment Scoring per Topic**
   - Fine-grained sentiment analysis within topics
   - Severity scoring for complaints
   - Prioritization based on sentiment intensity

6. **Integration into Live Dashboards**
   - Real-time monitoring of complaint trends
   - Alerting for spike detection
   - Continuous CX monitoring for product teams

7. **Advanced Clustering Techniques**
   - Hierarchical topic modeling
   - Multi-level topic decomposition
   - Dynamic topic modeling over time

8. **Customer Journey Analysis**
   - Complaint patterns by user lifecycle stage
   - First-time vs returning customer issues
   - Churn prediction based on complaint types

These extensions would make the analysis useful for continuous CX monitoring across product teams, support, and operations, enabling proactive identification and resolution of user pain points.

---

## Output Format

The final structured analysis produces:

- A set of major pain point categories with prevalence statistics
- Subtopics nested under those categories with detailed breakdowns
- Representative review excerpts for grounding and validation
- Statistical summaries including counts, percentages, and temporal patterns
- Visualizations showing prevalence and trends over time
- Human-validated interpretations of technical clusters

The final deliverable supports:
- Product planning and prioritization
- UX evaluation and improvement
- Customer experience strategy
- Support team training and resource allocation
- Executive reporting and decision-making

---

## Notes

- All analysis was performed locally
- Large model artifacts and CSV files are excluded from Git (see `.gitignore`)
- Data files exceed GitHub's 100 MB limit and are stored locally only
- The repository contains code and documentation only
- Results and visualizations are generated by running the pipeline scripts

---

## License

This project is for research and analysis purposes. Review data was collected from publicly available Google Play Store reviews.
