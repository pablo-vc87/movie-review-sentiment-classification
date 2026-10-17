# Movie Review Sentiment Classification

> Versión en español: [LEEME.md](LEEME.md)

A Natural Language Processing (NLP) project that classifies **IMDB movie reviews as positive or negative** from their text alone.

The project compares three text preprocessing strategies (basic cleaning, NLTK lemmatization, and spaCy lemmatization), two **TF-IDF** representations (unigrams and unigrams + bigrams), and four models. The selected model, **LinearSVC with TF-IDF unigrams + bigrams**, achieved an **F1-score of 0.9139 on validation** and **0.9043 F1 with 0.9663 ROC-AUC on the independent test set** (23,535 reviews), well above the requirement of **F1 ≥ 0.85**.

---

# Business Context

Film Junky Union, a community for classic movie enthusiasts, is building a system to filter and categorize movie reviews. The goal is to **automatically detect negative reviews**, so the community does not have to read and label thousands of texts by hand. A reliable model helps to:

* Classify new reviews as soon as they are published.
* Summarize the general opinion about a movie from many reviews.
* Free moderators from manual labeling.

The model must predict one target:

| Target | Description |
| --- | --- |
| `pos` | 1 if the review is positive, 0 if it is negative |

---

# NLP Concepts

| Term | Meaning |
| --- | --- |
| Corpus | The full collection of texts (here, the reviews) |
| Token | A unit of text, usually a word |
| Lemmatization | Reducing a word to its dictionary form (`movies` → `movie`, `were` → `be`) |
| TF-IDF | Term Frequency × Inverse Document Frequency: a word gets a high weight if it is frequent in a review but rare in the rest of the corpus |
| N-gram | A sequence of *n* consecutive words. Unigrams are single words; bigrams are pairs such as `not good` |
| Sparse matrix | A matrix where most values are zero. Each review uses only a few hundred of the hundreds of thousands of possible words |
| `sublinear_tf` | Uses `1 + log(tf)` instead of the raw count, so repeating a word many times does not dominate the review |
| `min_df` / `max_df` | Ignore words that appear in fewer than 2 reviews or in more than 95% of them |

## Text Representation

```text
"This movie wasn't very good!!!"
        │  basic cleaning
        ▼
"this movie wasn't very good"
        │  TF-IDF (unigrams + bigrams)
        ▼
[ this, movie, wasn't, very, good, this movie, movie wasn't, wasn't very, very good ]
        │
        ▼
Sparse vector with ~417,000 possible features → linear classifier
```

---

# Evaluation Metric

The main metric is the **F1-score**, the harmonic mean of Precision and Recall:

$$
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

F1 is only high when the model both **finds most negative/positive reviews** (Recall) and **is right when it flags them** (Precision). It also exposes trivial models: a `DummyClassifier` that always predicts the majority class gets about 50% Accuracy but **F1 = 0**.

**ROC-AUC** and **Accuracy** were recorded as secondary metrics. For LinearSVC, ROC-AUC is computed from `decision_function` scores, since the model does not output probabilities.

The project requirement was **F1 ≥ 0.85** on the test set.

---

# Dataset

| File | Rows | Columns | Description |
| --- | ---: | ---: | --- |
| `data/imdb_reviews.tsv` | 47,331 | 17 | Review text, sentiment label, movie metadata, and the original train/test partition |

Main columns used:

| Column | Description |
| --- | --- |
| `review` | Review text (the only feature used by the models) |
| `pos` | Target: 0 = negative, 1 = positive |
| `ds_part` | Original partition: `train` or `test` |

* The classes are balanced: **23,715 negative (50.1%)** and **23,616 positive (49.9%)** reviews.
* Only `average_rating` and `votes` have missing values (2 each); neither is used by the models.

**Data source.** The dataset was provided by **TripleTen** as part of its Data Science Bootcamp. It is based on the IMDB review dataset published by Maas et al. (2011), *Learning Word Vectors for Sentiment Analysis*, and is included in this repository for educational and reproducibility purposes.

---

# Project Workflow

```text
IMDB reviews (47,331)
            │
            ▼
1. Data loading and quality checks
            │
            ▼
2. Exploratory analysis: class balance and review length
            │
            ▼
3. Split: original test kept apart; original train → 80% train / 20% validation
            │
            ▼
4. Text preprocessing
   ├── Basic cleaning (lowercase, letters and apostrophes only)
   ├── + NLTK lemmatization
   └── + spaCy lemmatization
            │
            ▼
5. TF-IDF vectorization (unigrams, then unigrams + bigrams)
            │
            ▼
6. Modeling on validation
   ├── DummyClassifier (baseline)
   ├── Logistic Regression × 3 preprocessing strategies
   ├── LinearSVC
   ├── Logistic Regression and LinearSVC with bigrams
   └── LightGBM with bigrams
            │
            ▼
7. Retrain the best model on the full original train set and evaluate once on test
            │
            ▼
8. Classify 14 hand-written reviews
```

---

# 1. Data Preparation

* The TSV file was loaded with `votes` as a nullable integer (`Int64`).
* No rows were removed and no values were imputed: the models only use `review` → `pos`, and both columns are complete.
* The original `ds_part` partition was respected. The test set (**23,535 reviews**) was kept apart until the final evaluation.
* The original train set (**23,796 reviews**) was split into internal train and validation with `stratify`:

```text
Complete dataset (47,331)
│
├── ds_part = train (23,796)
│       ├── Internal train ──→ 19,036 (80%) → model training
│       └── Validation ──────→  4,760 (20%) → model and hyperparameter selection
│
└── ds_part = test (23,535) ───→ final evaluation (used only once)
```

All three sets keep the same class proportion (about 50.1% negative / 49.9% positive).

---

# 2. Exploratory Analysis

## Class Balance

| Partition | Negative | Positive |
| --- | ---: | ---: |
| Train | 11,912 (50.06%) | 11,884 (49.94%) |
| Test | 11,803 (50.15%) | 11,732 (49.85%) |

The difference between classes is below 1%, so no balancing technique (oversampling, undersampling, or class weights) was needed.

## Review Length

| Class | Mean words | Median words | Max words |
| --- | ---: | ---: | ---: |
| Negative | 227.5 | 173 | 1,491 |
| Positive | 231.3 | 171 | 2,459 |

* Reviews are long: a median of **172 words**, with the middle 50% between about 126 and 278 words.
* The length distributions of both classes almost completely overlap, so **length is not a useful signal**; the sentiment must be learned from the words themselves.

---

# 3. Text Preprocessing and Vectorization

## Preprocessing Strategies

| Strategy | Steps | TF-IDF features (unigrams) |
| --- | --- | ---: |
| Basic cleaning | Lowercase, keep only letters `a-z` and apostrophes, collapse spaces | 39,531 |
| NLTK lemmatization | Basic cleaning + `word_tokenize` + `WordNetLemmatizer` | 35,027 |
| spaCy lemmatization | Basic cleaning + `en_core_web_sm` lemmas (parser and NER disabled) | — |

Apostrophes were kept so that negations such as `wasn't` or `didn't` remain single tokens.

Example of lemmatization:

```text
Original: The movies were amazing and the actors were running beautifully.
NLTK:     The movie were amazing and the actor were running beautifully .
spaCy:    the movie be amazing and the actor be run beautifully .
```

NLTK's `WordNetLemmatizer` treats every word as a noun when no part of speech is given, so it only changed plurals. spaCy also lemmatized verbs, but it took about **3.5 minutes** to process the training set.

## TF-IDF Configuration

```python
TfidfVectorizer(ngram_range=(1, 1),   # or (1, 2) for bigrams
                min_df=2, max_df=0.95, sublinear_tf=True)
```

The vectorizer was always **fitted on the training texts only** and then applied to validation and test. Adding bigrams increased the feature space from 39,531 to **350,997** columns on the internal train set (416,551 on the full train set).

---

# 4. Modeling

Each model was tested with a small grid of hyperparameters (`C` for the linear models; trees, learning rate, and leaves for LightGBM), and the best configuration by validation F1 was kept.

## Experiment 1: Basic Cleaning vs Lemmatization (Logistic Regression, unigrams)

| Preprocessing | Best `C` | F1 | ROC-AUC |
| --- | ---: | ---: | ---: |
| **Basic cleaning** | 2 | **0.8989** | 0.9606 |
| NLTK lemmatization | 2 | 0.8973 | 0.9595 |
| spaCy lemmatization | 5 | 0.8932 | 0.9578 |

Lemmatization reduced the vocabulary but did not improve the results, and spaCy was much slower. **Basic cleaning was kept** for the rest of the project.

## Experiment 2: Logistic Regression vs LinearSVC (unigrams)

| Model | Best `C` | F1 | Accuracy | ROC-AUC | Training time |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 2 | **0.8989** | 0.8979 | 0.9606 | 1.78 s |
| LinearSVC | 0.5 | 0.8988 | **0.8983** | **0.9614** | **0.21 s** |

Both linear models performed almost the same, but LinearSVC trained about **8 times faster**.

## Experiment 3: Impact of Bigrams

| Model | Unigrams | Unigrams + Bigrams | Gain |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.8989 | 0.9076 | +0.0086 |
| LinearSVC | 0.8988 | **0.9139** | **+0.0150** |

Bigrams gave the **largest improvement in the project**, larger than any preprocessing change. They let the model capture short expressions such as `not good`, `not bad`, or `waste of`, where a single word gives the wrong signal.

## Experiment 4: LightGBM (unigrams + bigrams)

| Configuration | F1 | ROC-AUC | Training time |
| --- | ---: | ---: | ---: |
| 200 trees, `learning_rate=0.1`, 31 leaves | **0.8785** | 0.9506 | 93 s |
| 300 trees, `learning_rate=0.05`, 31 leaves | 0.8763 | 0.9473 | 141 s |
| 200 trees, `learning_rate=0.05`, 15 leaves | 0.8530 | 0.9306 | 58 s |

LightGBM met the requirement but was **less accurate and about 200 times slower** than LinearSVC. With hundreds of thousands of sparse features, a linear boundary fits the problem better than trees.

## Validation Summary

| Model | F1 | Accuracy | ROC-AUC | Training time |
| --- | ---: | ---: | ---: | ---: |
| **LinearSVC + Bigrams** (`C=0.5`) | **0.9139** | **0.9137** | **0.9677** | 0.47 s |
| Logistic Regression + Bigrams (`C=5`) | 0.9076 | 0.9074 | 0.9662 | 4.50 s |
| Logistic Regression (`C=2`) | 0.8989 | 0.8979 | 0.9606 | 1.78 s |
| LinearSVC (`C=0.5`) | 0.8988 | 0.8983 | 0.9614 | 0.21 s |
| Logistic Regression + NLTK (`C=2`) | 0.8973 | 0.8966 | 0.9595 | 1.51 s |
| Logistic Regression + spaCy (`C=5`) | 0.8932 | 0.8924 | 0.9578 | 2.48 s |
| LightGBM + Bigrams | 0.8785 | 0.8782 | 0.9506 | 92.95 s |
| DummyClassifier (most frequent) | 0.0000 | 0.5006 | 0.5000 | 0.001 s |

## Final Result

The TF-IDF vectorizer and the model were refitted on the **full original train set (23,796 reviews)** and evaluated **once** on the test set:

```text
Model:          LinearSVC (C=0.5, max_iter=5000) + TF-IDF unigrams + bigrams
Validation F1:  0.9139
Test F1:        0.9043
Test Accuracy:  0.9046
Test ROC-AUC:   0.9663
Requirement:    F1 ≥ 0.85   ✓
```

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Negative | 0.91 | 0.90 | 0.90 | 11,803 |
| Positive | 0.90 | 0.90 | 0.90 | 11,732 |

The F1 dropped by about 0.01 from validation to test, which is expected for data that took no part in model selection. Performance is **balanced between both classes**.

## Hand-Written Reviews

The best configuration of each model was used to classify 14 new reviews written by hand:

* All the linear models (Logistic Regression and LinearSVC, with or without lemmatization or bigrams) gave **the same label to all 14 reviews**.
* Clear reviews such as *"One of the worst movies I have ever seen…"* or *"I loved this film…"* were classified correctly by every model.
* *"I didn't expect the reboot to be so good!"* was classified as **negative** by the linear models, probably because words such as `didn't` are frequent in negative reviews. LightGBM got it right, but it also labeled two clearly negative reviews as positive (*"I did not simply like it…"* and *"Well, I was bored and felt asleep…"*).

---

# Conclusions

1. **The dataset was balanced**, so no balancing technique was needed and F1 was a fair metric for both classes.

2. **Basic cleaning was better than lemmatization.** Neither NLTK nor spaCy improved F1, and spaCy added several minutes of processing.

3. **Logistic Regression and LinearSVC performed almost identically** with unigrams, but LinearSVC was about 8 times faster.

4. **Bigrams were the most important improvement** (+0.015 F1 for LinearSVC). How the text is represented mattered more than how it was normalized.

5. **LightGBM was less accurate and much slower** on this high-dimensional sparse data.

6. **The final model exceeds the requirement**: 0.9043 F1 and 0.9663 ROC-AUC on 23,535 unseen reviews, with balanced results for both classes.

## Business Interpretation

The model classifies about 9 out of 10 reviews correctly and trains in under one second, so it can be retrained often and used to label new reviews automatically. Reviews with sarcasm, contrast (*"not bad, but…"*), or unexpected praise are the most likely errors, so borderline cases could be sent to a moderator.

## Next Steps

* Use the LinearSVC `decision_function` score to send low-confidence reviews to manual review.
* Inspect the largest positive and negative coefficients to explain which words and bigrams drive each prediction.
* Try trigrams or character n-grams to capture longer negations.
* Compare with a pre-trained transformer model (for example, BERT) to measure the gain against its higher computational cost.

---

# Project Structure

```text
movie-review-sentiment-classification/
│
├── data/
│   └── imdb_reviews.tsv
│
├── notebooks/
│   └── movie_reviews_classification.ipynb
│
├── src/
│   └── funciones_personales.py
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── LEEME.md
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

```bash
conda env create -f environment.yml
conda activate nlp_movie_reviews
```

## Option 2: pip

Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
```

## Language Resources

The lemmatization experiments need the spaCy English model and the NLTK data:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet')"
```

Then start Jupyter, open `notebooks/movie_reviews_classification.ipynb`, and run all cells:

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
* scikit-learn (TF-IDF, Logistic Regression, LinearSVC, DummyClassifier)
* LightGBM
* NLTK
* spaCy
* tqdm
* Jupyter Notebook

---

# Skills Demonstrated

* Natural Language Processing
* Text cleaning and normalization
* Lemmatization with NLTK and spaCy
* TF-IDF vectorization with unigrams and bigrams
* Sentiment classification
* Model comparison: Logistic Regression, LinearSVC, LightGBM, and a baseline
* Hyperparameter tuning
* Train / validation / test methodology without data leakage
* F1, ROC-AUC, and confusion matrix evaluation
* Performance vs computational cost analysis

---

## Author

**Pablo Andrés Vázquez Calva**

Data Scientist

GitHub: https://github.com/pablo-vc87

LinkedIn: https://www.linkedin.com/in/pablo-avc/
