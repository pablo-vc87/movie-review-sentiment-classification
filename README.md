# Movie Review Sentiment Classification

A Natural Language Processing project focused on automatically classifying IMDB movie reviews as **positive** or **negative**.

The project compares multiple text preprocessing strategies, TF-IDF representations, and machine learning models to identify the most effective approach for sentiment classification.

The final model, **LinearSVC with TF-IDF unigram and bigram features**, achieved an **F1-score of 0.9043 on the test set**, exceeding the project requirement of **F1 ≥ 0.85**.

---

## Project Overview

Film Junky Union, a community for classic movie enthusiasts, requires an automated system capable of filtering and categorizing movie reviews according to their sentiment.

The objective of this project was to build and evaluate machine learning models capable of classifying IMDB reviews as:

* Positive
* Negative

The main performance requirement was:

```text
F1-score ≥ 0.85
```

Several preprocessing strategies and machine learning models were evaluated to determine the best combination of predictive performance and computational efficiency.

---

## Project Workflow

```text
                                IMDB DATASET
                                     │
                                     ▼
                           Data Loading and Exploration
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
              Data Quality                    Exploratory Analysis
                    │                                 │
                    ▼                                 ▼
            Missing Values                     Class Balance
                                                      │
                                                      ▼
                                         Approximately balanced
                                         positive/negative classes
                                                      │
                                                      ▼
                                               Data Splitting
                                                      │
                                                      ▼
                                          Text Preprocessing
                                                      │
             ┌────────────────────────┬───────────────┴────────────────────┐
             ▼                        ▼                                    ▼
        Basic Cleaning          NLTK Lemmatization                  spaCy Lemmatization
             │                        │                                    │
             └────────────────────────┴──────────────────┬─────────────────┘
                                                          ▼
                                                 TF-IDF Vectorization
                                                          │
                        ┌─────────────────────────────────┼─────────────────────────────────┐
                        ▼                                 ▼                                 ▼
                    Unigrams                     Unigrams + Bigrams                  Model Comparison
                        │                                 │
            ┌───────────┴───────────┐         ┌───────────┴────────────┐
            ▼                       ▼         ▼                        ▼
     Logistic Regression       LinearSVC  Logistic Regression       LinearSVC
                                            │
                                            ▼
                                    LightGBM + Bigrams
                                            │
                                            ▼
                              Performance and Efficiency Comparison
                                            │
                                            ▼
                                  Best Model Selection
                                            │
                                            ▼
                            Retraining on the Training Dataset
                                            │
                                            ▼
                                  Final Test Evaluation
                                            │
                                            ▼
                              Classification of New Reviews
                                            │
                                            ▼
                                        Conclusions
```

---

# Dataset

The dataset used in this project contains IMDB movie reviews and their corresponding sentiment labels.

**File:**

```text
data/imdb_reviews.tsv
```

### Dataset characteristics

| Feature               |    Value |
| --------------------- | -------: |
| Records               |   47,331 |
| Columns               |       17 |
| Target variable       |    `pos` |
| Text feature          | `review` |
| Missing target values |        0 |
| Missing review values |        0 |

The target variable `pos` represents the sentiment classification:

```text
0 → Negative review
1 → Positive review
```

The dataset was approximately balanced, with close to 50% positive and 50% negative reviews. Therefore, no class balancing techniques were required.

Two missing values were found in:

* `average_rating`
* `votes`

These variables were not required for the text classification models.

---

# Data Splitting

The original dataset already included a predefined partition through the `ds_part` column.

The data was divided as follows:

```text
Complete Dataset
│
├── ds_part = train
│       │
│       ├── Internal Training Set → 80%
│       │
│       └── Validation Set → 20%
│
└── ds_part = test
        │
        └── Final Test Set
```

## Original Dataset Partition

The predefined dataset split contained:

| Dataset          |    Records |
| ---------------- | ---------: |
| Training dataset |     23,796 |
| Test dataset     |     23,535 |
| **Total**        | **47,331** |

The original test set was kept completely separate during model development and hyperparameter selection.

---

## Internal Train-Validation Split

The original training dataset was further divided using `train_test_split`:

```python
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The `stratify` parameter was used to preserve the original class distribution across the training and validation datasets.

The resulting split was:

| Dataset               | Records | Percentage of Original Training Set |
| --------------------- | ------: | ----------------------------------: |
| Internal training set |  19,036 |                                 80% |
| Validation set        |   4,760 |                                 20% |
| Original training set |  23,796 |                                100% |

The validation set was used to:

* Compare preprocessing strategies.
* Compare TF-IDF configurations.
* Evaluate unigram and bigram representations.
* Compare machine learning models.
* Test hyperparameter configurations.
* Select the final model.

---

## Final Training and Test Evaluation

After selecting the best configuration:

```text
LinearSVC
+
TF-IDF
+
Unigrams and Bigrams
```

the model was retrained using the complete original training dataset containing **23,796 reviews**.

The final model was then evaluated using the independent test set containing **23,535 reviews**.

This workflow ensured that the final test set was not used during model selection.

### Final Evaluation

```text
Training dataset: 23,796 reviews

Test dataset:     23,535 reviews

Final model:
LinearSVC + TF-IDF (Unigrams + Bigrams)

Final Test F1-score:
0.904334
```

---

# Text Preprocessing

Several preprocessing approaches were evaluated.

## Basic Cleaning

The basic preprocessing pipeline included:

* Converting text to lowercase.
* Removing non-alphabetic characters.
* Preserving apostrophes.
* Removing extra whitespace.

Example:

```text
"This movie wasn't very good!!!"
```

becomes approximately:

```text
"this movie wasn't very good"
```

---

## NLTK Lemmatization

A second preprocessing strategy used:

* Tokenization with NLTK.
* WordNet lemmatization.
* Reconstruction of the processed text.

---

## spaCy Lemmatization

A third approach used spaCy for linguistic preprocessing and lemmatization.

Although lemmatization can theoretically reduce vocabulary complexity, it did not improve model performance for this dataset.

---

# Feature Engineering

Text was transformed into numerical representations using **TF-IDF vectorization**.

Two main configurations were evaluated:

### Unigrams

```text
ngram_range=(1, 1)
```

Example:

```text
movie
good
boring
excellent
```

### Unigrams and Bigrams

```text
ngram_range=(1, 2)
```

This configuration allowed the models to capture short expressions such as:

```text
not good
not bad
very good
very bad
waste time
highly recommend
```

The inclusion of bigrams produced the largest improvement in predictive performance.

---

# Models Evaluated

The following models were compared:

* DummyClassifier
* Logistic Regression
* LinearSVC
* LightGBM

Different combinations of preprocessing strategies and TF-IDF configurations were evaluated.

---

# Model Performance

## Validation Results

| Model                         |           F1 |     Accuracy |      ROC-AUC | Training Time |
| ----------------------------- | -----------: | -----------: | -----------: | ------------: |
| **LinearSVC + Bigrams**       | **0.913855** | **0.913655** | **0.967661** |    **0.47 s** |
| Logistic Regression + Bigrams |     0.907567 |     0.907353 |     0.966184 |        4.50 s |
| Logistic Regression           |     0.898918 |     0.897899 |     0.960568 |        1.78 s |
| LinearSVC                     |     0.898829 |     0.898319 |     0.961366 |        0.21 s |
| Logistic Regression + NLTK    |     0.897329 |     0.896639 |     0.959479 |        1.51 s |
| Logistic Regression + spaCy   |     0.893244 |     0.892437 |     0.957768 |        2.48 s |
| LightGBM + Bigrams            |     0.878509 |     0.878151 |     0.950561 |       92.95 s |
| DummyClassifier               |     0.000000 |     0.500630 |     0.500000 |       0.001 s |

---

# Key Findings

## 1. Basic Cleaning vs. Lemmatization

For Logistic Regression using unigram TF-IDF features:

| Preprocessing Strategy |           F1 |
| ---------------------- | -----------: |
| **Basic Cleaning**     | **0.898918** |
| NLTK Lemmatization     |     0.897329 |
| spaCy Lemmatization    |     0.893244 |

Lemmatization did not improve model performance.

Additionally, spaCy required substantially more preprocessing time. Processing the training dataset required approximately 3 minutes and 32 seconds, while basic cleaning was significantly more efficient.

Therefore, **basic text cleaning was selected as the primary preprocessing strategy**.

---

## 2. Logistic Regression vs. LinearSVC

Using unigram TF-IDF features:

| Model               |           F1 |     Accuracy |      ROC-AUC | Training Time |
| ------------------- | -----------: | -----------: | -----------: | ------------: |
| Logistic Regression | **0.898918** |     0.897899 |     0.960568 |        2.21 s |
| LinearSVC           |     0.898829 | **0.898319** | **0.961366** |    **0.21 s** |

Both models achieved nearly identical predictive performance.

However, LinearSVC trained approximately ten times faster.

This result demonstrates that linear models are particularly effective for high-dimensional and sparse TF-IDF representations.

---

## 3. Impact of Bigrams

The largest improvement came from modifying the text representation.

| Model               | Unigrams | Unigrams + Bigrams |
| ------------------- | -------: | -----------------: |
| Logistic Regression | 0.898918 |           0.907567 |
| LinearSVC           | 0.898829 |       **0.913855** |

Adding bigrams improved:

* Logistic Regression by approximately **0.0086 F1 points**.
* LinearSVC by approximately **0.0150 F1 points**.

This improvement was greater than the gains obtained from additional linguistic preprocessing.

The results suggest that capturing short expressions such as:

```text
not good
not bad
very good
very bad
```

was more valuable than applying lemmatization.

---

## 4. LightGBM Results

LightGBM was also evaluated using TF-IDF features with unigram and bigram representations.

Its best result was:

```text
F1-score:          0.878509
Accuracy:          0.878151
ROC-AUC:           0.950561
Training time:     92.95 seconds
```

Compared with the linear models:

| Model                         |      Best F1 | Training Time |
| ----------------------------- | -----------: | ------------: |
| **LinearSVC + Bigrams**       | **0.913855** |    **0.47 s** |
| Logistic Regression + Bigrams |     0.907567 |        4.50 s |
| Logistic Regression           |     0.898918 |        1.78 s |
| LinearSVC                     |     0.898829 |        0.21 s |
| LightGBM + Bigrams            |     0.878509 |       92.95 s |

LightGBM required significantly more computational time while producing lower predictive performance.

For this dataset and TF-IDF representation, linear models proved to be a better fit.

---

# Final Model

The selected model was:

```text
LinearSVC
```

with:

```text
C = 0.5
max_iter = 5000
random_state = 42
```

The text representation used:

```text
TF-IDF
ngram_range = (1, 2)
```

This configuration includes both unigrams and bigrams.

---

# Final Results

After selecting the best configuration during validation, the model was retrained using the complete training dataset and evaluated on the test set.

```text
Model:
LinearSVC + TF-IDF (Unigrams + Bigrams)

Validation F1:
0.913855

Final Test F1:
0.904334

Project Requirement:
F1 ≥ 0.85
```

## Result

```text
✓ PROJECT REQUIREMENT ACHIEVED
```

The final model exceeded the required F1-score by more than 5 percentage points.

---

# Project Structure

```text
movie-review-sentiment-classification/
│
├── data/
│   └── imdb_reviews.tsv
│
├── notebooks/
│   └── movies_reviews_classification.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/pablo-vc87/movie-review-sentiment-classification.git
cd movie-review-sentiment-classification
```

## Option 1: Conda

Create the environment using:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate nlp_movie_reviews
```

Then start Jupyter:

```bash
jupyter notebook
```

---

## Option 2: pip

Create and activate a virtual environment.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then launch Jupyter:

```bash
jupyter notebook
```

---

# Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* scikit-learn
* LightGBM
* NLTK
* spaCy
* TF-IDF
* Jupyter Notebook

---

# Skills Demonstrated

This project demonstrates experience with:

* Natural Language Processing
* Text preprocessing
* TF-IDF vectorization
* N-gram analysis
* Sentiment classification
* Machine Learning model comparison
* Hyperparameter tuning
* Model evaluation
* F1-score optimization
* ROC-AUC analysis
* Performance benchmarking
* Data preprocessing
* Python-based data analysis

---

# Conclusions

The objective of this project was to develop an automated system capable of classifying movie reviews as positive or negative while achieving an F1-score of at least 0.85.

Several preprocessing strategies, linear models, and a gradient boosting model were evaluated.

The main conclusions were:

1. The dataset was approximately balanced, so class balancing techniques were not necessary.
2. Basic text cleaning performed better than NLTK and spaCy lemmatization.
3. Logistic Regression and LinearSVC achieved very similar performance with unigram TF-IDF features.
4. LinearSVC required significantly less training time than Logistic Regression.
5. Adding bigrams produced the largest improvement in model performance.
6. LinearSVC with unigram and bigram TF-IDF features achieved the best validation performance with an F1-score of **0.913855**.
7. After retraining and evaluating on the test set, the final model achieved an F1-score of **0.904334**.
8. LightGBM was less effective and substantially slower for this sparse TF-IDF representation.

The final solution demonstrates that, for high-dimensional sparse text data, a relatively simple linear model combined with an appropriate feature representation can outperform more computationally expensive machine learning algorithms.

---

## Author

**Pablo Andrés Vázquez Calva**

Data Analyst | Data Science

GitHub: [github.com/pablo-vc87](https://github.com/pablo-vc87)

LinkedIn: [linkedin.com/in/pablo-avc](https://www.linkedin.com/in/pablo-avc/)
