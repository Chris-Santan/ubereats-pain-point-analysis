Uber Eats Pain-Point Analysis
Project Overview

This project performs large scale analysis of English language Uber Eats reviews sourced from the Google Play Store. The goal was to identify recurring negative user pain points at scale, reduce them into meaningful categories, and produce actionable summaries using modern NLP tooling. The dataset contained roughly 80000 reviews after scraping, reduced to 38120 eligible negative reviews after filtering for sentiment and length. All analysis was done locally using reproducible Python scripts. The final outcome is a structured breakdown of major complaint categories supported by topic modeling, clustering, visualization, and human validation.

Data Collection & Filtering
Scraping Review Data

Reviews were scraped using a custom scraper which collected approximately 80000 Uber Eats reviews from the Google Play Store. Only English language reviews were collected because they are the most consistent for downstream embedding models and because sentiment heuristics tend to perform better.

Filtering Criteria

A cleaning script removed neutral or positive reviews and filtered for meaningful content. Filters applied:

Rating less than 4 stars

Minimum length 7 words

English text only
After filtering, the dataset contained roughly 38120 eligible negative reviews suitable for NLP analysis. This approach ensured that downstream topic extraction would focus on user complaints rather than general praise or noise.

NLP Processing Pipeline
Text Preprocessing

Standard NLP preprocessing was applied including lowercasing, punctuation removal, stopword removal, and normalization of whitespace. This stage was designed to reduce noise without damaging semantic meaning for transformer based embeddings.

Topic Modeling with BERTopic

BERTopic was used as the primary tool to extract coherent topic clusters from the filtered dataset. The pipeline produced:

Sentence Transformers based embeddings

HDBSCAN clustering

UMAP dimensionality reduction

Topic labeling based on word frequency and TF-IDF
The initial BERTopic pass produced dozens of interpretable topics including complaints about promotions, fees, restaurant issues, account issues, refunds, and delivery problems.

Subtopic Discovery

To further resolve granular complaints, additional clustering was performed inside major topics. For example, the promotion related topic was decomposed into subtopics such as unusable promo codes, expiring promotions, fake discounts, and eligibility errors. Subtopic work was done through both BERTopic iterative clustering and alternative embedding based clustering tools.

Visualization
Graphical Outputs

The project generated several visual diagnostic outputs including:

Topic frequency bar charts

Embedding clusters via UMAP

Distribution charts

Temporal patterns where relevant
These allowed quick validation of topic coherence and density.

KMeans Alternative Clustering

For topics where BERTopic treated many items as outliers due to HDBSCAN parameters, KMeans was used as an alternative. KMeans avoided outlier discarding, preserved semantic grouping, and allowed fixed K selection for clearer cluster labeling inside topics. This was especially useful for granular promotional complaints.

Human in the Loop Validation

After clustering, representative samples from selected subtopics were reviewed manually to ensure semantic correctness. Human validation helped interpret raw cluster labels into meaningful business level pain points. Examples of validated interpretations include:

Unauthorized or incorrect tip charges

Promotion codes being expired or invalid

Account suspensions tied to gift card redemptions
This hybrid method aligns unsupervised ML with practical business insights.

Final Output Format

The final structured analysis produces:

A set of major pain point categories

Subtopics nested under those categories

Representative review excerpts for grounding

Optional statistical summaries
The final deliverable supports product planning, UX evaluation, and customer experience strategy.

Technical Stack

Python

BeautifulSoup / scraping utilities

Sentence Transformers

BERTopic

HDBSCAN

UMAP

Scikit Learn

Pandas / NumPy

Matplotlib / Seaborn

All analysis was performed locally, and large model artifacts were excluded from Git to meet GitHub file size constraints.

Repository Structure
/src
    scraping
    preprocessing
    topic_modeling
    clustering
    visualization
/models
    (local only, not uploaded due to size)
/docs
    (optional)
README.md


The repository contains code only. Data and model files were not pushed because many exceed GitHub’s 100 MB limit.

Current Scope and Future Work
Current Scope Completed

Data scraping

Filtering

Embedding generation

Topic modeling

Subtopic clustering

Visualization

Human evaluation

Potential Future Extensions

Temporal trend analysis of complaints

Cross platform comparison (iOS vs Android)

Geographic segmentation

Automated pain point summarization

Sentiment scoring per topic

Integration into live dashboards

These extensions would make the analysis useful for continuous CX monitoring across product teams, support, and operations.