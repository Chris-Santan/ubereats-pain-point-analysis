Uber Eats Pain Point Analysis
Overview

This project analyzes user pain points in Uber Eats by extracting, cleaning, modeling, and evaluating large-scale consumer reviews from the Google Play Store. Over eighty thousand English reviews were scraped and processed to isolate negative experiences with a focus on product, pricing, promotions, and operational failures. Natural language processing, topic modeling, clustering, and manual qualitative review were used together to surface patterns that reflect real user frustrations. The outcome is a structured understanding of the most common UX and business pain points, supported by both quantitative and qualitative evidence.

Objectives

The project was built around three core goals.
First, identify the major categories of user frustration without pre-defining labels.
Second, isolate and analyze fine-grained subtopics within these categories.
Third, interpret findings through the lens of practical product recommendations rather than abstract sentiment labels.

Data Collection

A custom scraping script (scraper.py) was built to collect approximately 80,000 Uber Eats Google Play Store reviews. The scraper extracted review text, rating, date, and metadata fields. Only English-language content was retained for downstream modeling in this project.

Data Filtering and Cleaning

After scraping, the data passed through several processing stages using clean_data.py and preprocess.py.

Filtering decisions included:

Removing reviews with ratings above three stars in order to isolate negative feedback.

Removing short reviews under seven words to reduce noise and non-informative content.

Standardizing whitespace, punctuation, and casing.

Removing URLs, emails, and vendor-specific formatting artifacts.

After filtering, roughly 38,120 reviews remained from the initial 80,000.

Embeddings and Topic Modeling

The core modeling pipeline used BERTopic to discover latent high-level themes from user text. The bertopic.py module applied the following steps:

Transformed cleaned text into dense sentence embeddings using a transformer-based language model.

Reduced embedding dimensionality for clustering.

Assigned reviews to topics.

Generated topic labels using representation models.

This produced a set of distinct high-level topics representing major user complaints such as:

Promotional code failures.

Delivery fees and tipping frustrations.

Gift card and credit errors.

Customer service failures.

Order fulfillment problems.

Subtopic Discovery and Deep Analysis

High-level topics alone do not capture the full structure of user complaints. To address this, the project introduced a second analysis stage targeting specific topics for fine-grained clustering.

Two methods were added:

BERTopic on-topic subclustering for hierarchical themes.

KMeans clustering on sentence embeddings for partitioning without outliers.

These produced CSV outputs for each selected topic that contained subtopic labels and corresponding reviews. For example, the promotional code topic yielded clusters separating issues such as:

Codes advertised but not honored.

Eligibility or “account not eligible” errors.

Code disappears at checkout.

Expiration or deactivation complaints.

Referral codes not applying.

False advertising accusations.

The KMeans method assigned every review to a subcluster with no dropped outliers, which provided a more complete representation for product-oriented interpretation.

Manual Qualitative Layer

After computational modeling, selected subtopics underwent manual qualitative review. This step was essential because models alone cannot determine which issues matter most to users or businesses. Manual review focused on:

Extracting the core failure mechanism in user terms.

Identifying user expectations versus actual outcomes.

Extracting indicators of financial risk, support breakdown, or policy confusion.

Distinguishing authentic product failures from eligibility or policy misunderstandings.

An example subtopic, “codes, code and promo codes related issues”, contained 155 reviews describing scenarios such as codes disappearing at checkout, being deactivated before stated expiry, or being flagged as already used. The qualitative analysis revealed that many complaints described situations perceived as financial baiting or false advertising rather than minor technical bugs. That distinction is only visible through human interpretation.

Key Findings

Quantitative modeling and manual review together surfaced several concrete patterns.

Promotional codes:
Users frequently reported receiving codes by email, push, or mail that failed at checkout, disappeared upon order confirmation, or returned account eligibility errors despite satisfying requirements. Many users interpreted this as false advertising, baiting, or intentional withdrawal of promotions. Support channels compounded frustration by stating that promo deactivation is permitted at any time, which undermined trust.

Fees and tipping:
Review clusters revealed a mix of sticker shock at total cost, frustration with hidden or unexpected fees, and confusion about tipping defaults. Several users did not mind the product itself but rejected the business model of stacking fees.

Gift cards:
Gift card users experienced redemption failures, state mismatch issues, or confusing flows that prevented use and left credit stranded.

Customer support:
Many reviews described an inability to resolve issues because live support was missing, slow, or unhelpful. In several cases, this transformed minor technical failures into major emotional complaints.

These patterns suggest that a large share of user dissatisfaction stems not from order logistics alone but from price transparency, promotional integrity, billing confidence, and support responsiveness.

Repository Structure

The repository is organized to reflect the full processing workflow.

/data/                 (not included in repo due to size and privacy)
    raw/               (scraped output)
    processed/         (cleaned and filtered reviews)
    embeddings/        (vectorized document embeddings)
    deep_analysis/     (subtopics and clustering outputs)

/models/               (local LLM and BERTopic objects, excluded from GitHub)

/scripts/
    scraper.py
    clean_data.py
    preprocess.py
    bertopic.py
    kmeans_subtopics.py
    label_topics.py
    analyze_surface_topics.py

/notebooks/
    exploratory_analysis.ipynb
    manual_review.ipynb

README.md
requirements.txt


Large data artifacts and transformer models are intentionally excluded from GitHub due to size and licensing.

Running the Project

To reproduce the pipeline:

Install dependencies from requirements.txt.

Run scraper.py to collect reviews.

Run clean_data.py and preprocess.py to filter and normalize.

Run bertopic.py for topic modeling.

Optionally run kmeans_subtopics.py for dense subtopics.

Inspect outputs in /data/deep_analysis/.

Perform qualitative review via notebooks if desired.

GPU acceleration is optional but recommended for embedding-heavy stages.

Limitations and Future Work

This project does not attempt to classify sentiment beyond isolating negative reviews, and it does not cross-reference reviews against dates, app versions, geography, or business experiments. Future work may incorporate temporal tracking, cross-platform comparison, causal inference, and survey triangulation.

Another likely extension is using Retrieval Augmented Generation to summarize clusters and generate structured product requirements directly from user text.