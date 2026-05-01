"""
Movie Review Sentiment Classifier — NLP Final Project
Binary classification (positive / negative) using NLTK.

Components used:
  - nltk.corpus.movie_reviews      (Ch02) corpus access
  - nltk.FreqDist                  (Ch02) vocabulary selection
  - nltk.corpus.stopwords          (Ch02) noise removal
  - nltk.word_tokenize             (Ch03) tokenization
  - nltk.stem.PorterStemmer        (Ch03) stemming
  - nltk.pos_tag                   (Ch05) POS-filtered features
  - nltk.NaiveBayesClassifier      (Ch06) primary classifier
  - nltk.DecisionTreeClassifier    (Ch06) secondary classifier
  - nltk.classify.accuracy         (Ch06) evaluation
  - nltk.ConfusionMatrix           (Ch05/06) error analysis
"""

import random
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import PorterStemmer
from nltk.classify import accuracy, apply_features
from nltk import FreqDist, NaiveBayesClassifier, DecisionTreeClassifier, pos_tag

# ---------------------------------------------------------------------------
# 0. Download required NLTK data (safe to call repeatedly)
# ---------------------------------------------------------------------------

def download_nltk_data():
    resources = [
        "movie_reviews",
        "stopwords",
        "punkt",
        "punkt_tab",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
    ]
    for r in resources:
        nltk.download(r, quiet=True)


# ---------------------------------------------------------------------------
# 1. Data loading  (Ch02 — corpus access)
# ---------------------------------------------------------------------------

def load_documents():
    """Return a shuffled list of (word_list, category) tuples."""
    documents = [
        (list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)
    ]
    random.seed(42)
    random.shuffle(documents)
    return documents


# ---------------------------------------------------------------------------
# 2. Preprocessing  (Ch02 stopwords + Ch03 tokenization & stemming)
# ---------------------------------------------------------------------------

def preprocess(word_list):
    """Lowercase, remove stopwords and non-alphabetic tokens, then stem."""
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    return [
        stemmer.stem(w.lower())
        for w in word_list
        if w.isalpha() and w.lower() not in stop_words
    ]


# ---------------------------------------------------------------------------
# 3. Vocabulary selection  (Ch02 — FreqDist)
# ---------------------------------------------------------------------------

def build_vocabulary(documents, top_n=2000):
    """Use FreqDist to select the top-N most frequent words across all docs."""
    all_words = FreqDist(
        word
        for (word_list, _) in documents
        for word in preprocess(word_list)
    )
    return [w for (w, _) in all_words.most_common(top_n)]


# ---------------------------------------------------------------------------
# 4. Feature extraction  (Ch06 — bag-of-words features)
# ---------------------------------------------------------------------------

def bow_features(word_list, vocabulary):
    """Boolean bag-of-words: True if word appears in document."""
    doc_set = set(preprocess(word_list))
    return {f"contains({w})": (w in doc_set) for w in vocabulary}


def pos_features(word_list, vocabulary):
    """
    POS-augmented features (Ch05): adds a flag for adjective presence.
    Adjectives (JJ, JJR, JJS) carry strong sentiment signal.
    """
    base = bow_features(word_list, vocabulary)
    tagged = pos_tag(word_list[:200])  # tag first 200 tokens (speed)
    adjectives = {w.lower() for (w, tag) in tagged if tag.startswith("JJ")}
    base["has_adjectives"] = bool(adjectives)
    base["adj_count"] = len(adjectives)
    return base


# ---------------------------------------------------------------------------
# 5. Train / test split and classifier training  (Ch06)
# ---------------------------------------------------------------------------

def build_and_evaluate(documents, vocabulary, feature_fn, label):
    """Train NaiveBayes + DecisionTree and print evaluation results."""
    featuresets = [(feature_fn(words, vocabulary), cat) for (words, cat) in documents]

    split = int(len(featuresets) * 0.8)
    train_set = featuresets[:split]
    test_set  = featuresets[split:]

    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Training samples : {len(train_set)}")
    print(f"  Test samples     : {len(test_set)}")

    # --- Naive Bayes ---
    nb_classifier = NaiveBayesClassifier.train(train_set)
    nb_acc = accuracy(nb_classifier, test_set)
    print(f"\n  [Naive Bayes] Accuracy: {nb_acc:.4f} ({nb_acc*100:.2f}%)")

    print("\n  Top 20 most informative features (Naive Bayes):")
    nb_classifier.show_most_informative_features(20)

    # --- Decision Tree ---
    dt_classifier = DecisionTreeClassifier.train(
        train_set, entropy_cutoff=0.05, depth_cutoff=100, support_cutoff=10
    )
    dt_acc = accuracy(dt_classifier, test_set)
    print(f"\n  [Decision Tree] Accuracy: {dt_acc:.4f} ({dt_acc*100:.2f}%)")

    # --- Confusion Matrix (Ch05/06) ---
    gold      = [cat for (_, cat) in test_set]
    predicted = [nb_classifier.classify(feat) for (feat, _) in test_set]
    cm = nltk.ConfusionMatrix(gold, predicted)
    print("\n  Confusion Matrix (Naive Bayes):")
    print(cm.pretty_format(sort_by_count=True, show_percents=True))

    # --- Precision / Recall / F1 ---
    print_precision_recall_f1(gold, predicted)

    return nb_classifier, dt_classifier, nb_acc, dt_acc


def print_precision_recall_f1(gold, predicted):
    """Compute and display per-class Precision, Recall, and F1."""
    classes = ["pos", "neg"]
    print("\n  Per-class metrics (Naive Bayes):")
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for cls in classes:
        tp = sum(1 for g, p in zip(gold, predicted) if g == cls and p == cls)
        fp = sum(1 for g, p in zip(gold, predicted) if g != cls and p == cls)
        fn = sum(1 for g, p in zip(gold, predicted) if g == cls and p != cls)
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall    = tp / (tp + fn) if (tp + fn) else 0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) else 0)
        print(f"  {cls:<8} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")


# ---------------------------------------------------------------------------
# 6. Visualisations  (Ch02 — FreqDist.plot)
# ---------------------------------------------------------------------------

def plot_frequency_distribution(documents, top_n=30):
    """Plot the top-N most frequent words across all reviews."""
    all_words = FreqDist(
        word
        for (word_list, _) in documents
        for word in preprocess(word_list)
    )
    all_words.plot(top_n, title=f"Top {top_n} Word Frequencies (after preprocessing)")
    plt.tight_layout()
    plt.savefig("frequency_distribution.png", dpi=150)
    plt.close()
    print(f"\n  Frequency plot saved → frequency_distribution.png")


def plot_category_word_frequencies(documents, top_n=20):
    """Plot top words per sentiment category (positive vs negative)."""
    from nltk import ConditionalFreqDist
    cfd = ConditionalFreqDist(
        (cat, word)
        for (word_list, cat) in documents
        for word in preprocess(word_list)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cat in zip(axes, ["pos", "neg"]):
        words, counts = zip(*cfd[cat].most_common(top_n))
        ax.barh(words[::-1], counts[::-1])
        ax.set_title(f"Top {top_n} words — {cat.upper()}")
        ax.set_xlabel("Frequency")
    plt.tight_layout()
    plt.savefig("category_frequencies.png", dpi=150)
    plt.close()
    print("  Category frequency plot saved → category_frequencies.png")


# ---------------------------------------------------------------------------
# 7. Main entry point
# ---------------------------------------------------------------------------

def main():
    print("Downloading NLTK resources...")
    download_nltk_data()

    print("\nLoading and preprocessing movie reviews corpus...")
    documents = load_documents()
    print(f"  Total documents: {len(documents)}")
    print(f"  Positive: {sum(1 for _, c in documents if c == 'pos')}")
    print(f"  Negative: {sum(1 for _, c in documents if c == 'neg')}")

    print("\nBuilding vocabulary (top 2000 words via FreqDist)...")
    vocabulary = build_vocabulary(documents)
    print(f"  Vocabulary size: {len(vocabulary)} words")

    # --- Experiment 1: pure bag-of-words ---
    nb1, dt1, nb_acc1, dt_acc1 = build_and_evaluate(
        documents, vocabulary, bow_features,
        label="Experiment 1 — Bag-of-Words Features"
    )

    # --- Experiment 2: BoW + POS adjective features ---
    nb2, dt2, nb_acc2, dt_acc2 = build_and_evaluate(
        documents, vocabulary, pos_features,
        label="Experiment 2 — BoW + POS Adjective Features"
    )

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Model':<35} {'Accuracy':>10}")
    print(f"  {'-'*48}")
    print(f"  {'NaiveBayes (BoW)':<35} {nb_acc1:>10.4f}")
    print(f"  {'DecisionTree (BoW)':<35} {dt_acc1:>10.4f}")
    print(f"  {'NaiveBayes (BoW + POS)':<35} {nb_acc2:>10.4f}")
    print(f"  {'DecisionTree (BoW + POS)':<35} {dt_acc2:>10.4f}")

    # --- Visualisations ---
    print("\nGenerating visualisations...")
    plot_frequency_distribution(documents)
    plot_category_word_frequencies(documents)

    print("\nDone.")


if __name__ == "__main__":
    main()
