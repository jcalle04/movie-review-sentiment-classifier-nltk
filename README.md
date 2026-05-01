# Movie Review Sentiment Classifier — NLP Final Project

Binary sentiment classifier (**positive / negative**) for movie reviews, built entirely with [NLTK](https://www.nltk.org/).  
Final project for the Natural Language Processing course — Academic Year 2025–2026.

**Group:** Juan Calle · Juan Francisco González · Rodrigo Pelayo · Mar Blanco

---

## Overview

Given a movie review, the system predicts whether it expresses a **positive** or **negative** opinion.  
The full pipeline — corpus access, text normalisation, feature engineering, supervised learning, and evaluation — is implemented exclusively with NLTK components covered in class.

| Property | Value |
|---|---|
| Dataset | `nltk.corpus.movie_reviews` (2 000 reviews, balanced) |
| Best model | Naive Bayes + BoW + POS features |
| Best accuracy | **76.50%** |
| Train / Test split | 80 % / 20 % (1 600 / 400 samples) |

---

## NLTK Components Used

| Component | Chapter | Purpose |
|---|---|---|
| `nltk.corpus.movie_reviews` | Ch02 | Labelled corpus access |
| `nltk.FreqDist` | Ch02 | Vocabulary selection (top-2 000 words) |
| `nltk.corpus.stopwords` | Ch02 | Noise removal |
| `nltk.word_tokenize` | Ch03 | Tokenisation & normalisation |
| `nltk.stem.PorterStemmer` | Ch03 | Stemming |
| `nltk.pos_tag` | Ch05 | POS-augmented features (adjectives) |
| `nltk.NaiveBayesClassifier` | Ch06 | Primary classifier |
| `nltk.DecisionTreeClassifier` | Ch06 | Secondary classifier (comparison) |
| `nltk.classify.accuracy` | Ch06 | Evaluation |
| `nltk.ConfusionMatrix` | Ch06 | Error analysis |

---

## Results

| Model | Features | Accuracy |
|---|---|---|
| Naive Bayes | BoW | 76.00 % |
| Decision Tree | BoW | 64.25 % |
| **Naive Bayes** | **BoW + POS** | **76.50 %** |
| Decision Tree | BoW + POS | 62.75 % |

**Per-class metrics — best model (Naive Bayes, BoW + POS):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Positive | 0.8214 | 0.6832 | 0.7459 |
| Negative | 0.7241 | 0.8485 | 0.7814 |

---

## Repository Structure

```
├── src/
│   └── main.py                      # Full runnable pipeline
├── report/
│   ├── report.pdf                   # Final report (10 pages)
│   ├── report.tex                   # LaTeX source
│   └── PROJECT_DOCUMENTATION.md    # Technical documentation
├── figures/
│   ├── frequency_distribution.png  # Top-30 word frequencies (Figure 1)
│   └── category_frequencies.png    # Top-20 words per category (Figure 2)
├── chapters/
│   ├── ch01.html                   # NLTK Book — Ch1: Language Processing & Python
│   ├── ch02.html                   # NLTK Book — Ch2: Text Corpora & Lexical Resources
│   ├── ch03.html                   # NLTK Book — Ch3: Processing Raw Text
│   ├── ch05.html                   # NLTK Book — Ch5: Categorizing & Tagging Words
│   ├── ch06.html                   # NLTK Book — Ch6: Learning to Classify Text
│   └── ch08.html                   # NLTK Book — Ch8: Analyzing Sentence Structure
└── README.md
```

---

## How to Run

```bash
pip install nltk matplotlib
python3 src/main.py
```

NLTK data (corpus, stopwords, POS tagger) is downloaded automatically on first run.

**Output:**
- Console: accuracy, confusion matrix, top-20 informative features, Precision/Recall/F1 for both experiments
- `frequency_distribution.png` — word frequency plot
- `category_frequencies.png` — per-category word frequency plot

---

## Report

The full technical report is available as [`report.pdf`](report.pdf) (10 pages).  
It covers methodology, NLTK component justifications, results, and discussion.
