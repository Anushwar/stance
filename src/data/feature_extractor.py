"""
Feature extraction for stance detection
Modular, composable feature extractors
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from typing import Optional, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


# Download VADER if not present
try:
    nltk.data.find("vader_lexicon")
except LookupError:
    print("Downloading VADER lexicon...")
    nltk.download("vader_lexicon", quiet=True)


class FeatureExtractor:
    """
    Modular feature extractor that can combine multiple feature types

    Philosophy: Start simple, add complexity only if it helps
    """

    def __init__(
        self,
        use_tfidf: bool = True,
        use_sentiment: bool = False,
        use_target: bool = False,
        use_text_stats: bool = False,
        tfidf_config: Optional[dict] = None,
    ):
        """
        Args:
            use_tfidf: Use TF-IDF features
            use_sentiment: Use VADER sentiment scores
            use_target: Use target one-hot encoding
            use_text_stats: Use text statistics (length, punctuation, etc.)
            tfidf_config: TF-IDF configuration dict
        """
        self.use_tfidf = use_tfidf
        self.use_sentiment = use_sentiment
        self.use_target = use_target
        self.use_text_stats = use_text_stats

        # TF-IDF
        if use_tfidf:
            config = tfidf_config or {}
            self.tfidf_vectorizer = TfidfVectorizer(**config)
        else:
            self.tfidf_vectorizer = None

        # Sentiment analyzer
        if use_sentiment:
            self.sia = SentimentIntensityAnalyzer()
        else:
            self.sia = None

        # Track feature names and dimensions
        self.feature_names_ = []
        self.feature_dimensions_ = {}

    def fit(
        self, df: pd.DataFrame, text_column: str = "cleaned_text"
    ) -> "FeatureExtractor":
        """
        Fit the feature extractor on training data

        Args:
            df: DataFrame with text and other columns
            text_column: Name of text column

        Returns:
            self
        """
        if self.use_tfidf:
            self.tfidf_vectorizer.fit(df[text_column])
            self.feature_dimensions_["tfidf"] = len(self.tfidf_vectorizer.vocabulary_)
            print(f"TF-IDF vocabulary size: {self.feature_dimensions_['tfidf']}")

        if self.use_target and "Target" in df.columns:
            self.target_categories_ = sorted(df["Target"].unique())
            self.feature_dimensions_["target"] = len(self.target_categories_)
            print(f"Target categories: {self.target_categories_}")

        return self

    def transform(
        self, df: pd.DataFrame, text_column: str = "cleaned_text"
    ) -> csr_matrix:
        """
        Transform data into feature matrix

        Args:
            df: DataFrame with text and other columns
            text_column: Name of text column

        Returns:
            Sparse feature matrix
        """
        features = []

        # 1. TF-IDF features
        if self.use_tfidf:
            tfidf_features = self.tfidf_vectorizer.transform(df[text_column])
            features.append(tfidf_features)

        # 2. Sentiment features
        if self.use_sentiment:
            sentiment_features = self._extract_sentiment(df, text_column)
            features.append(csr_matrix(sentiment_features))
            if "sentiment" not in self.feature_dimensions_:
                self.feature_dimensions_["sentiment"] = sentiment_features.shape[1]

        # 3. Target features
        if self.use_target and "Target" in df.columns:
            target_features = self._extract_target(df)
            features.append(csr_matrix(target_features))
            if "target" not in self.feature_dimensions_:
                self.feature_dimensions_["target"] = target_features.shape[1]

        # 4. Text statistics
        if self.use_text_stats:
            text_stats = self._extract_text_stats(df, text_column)
            features.append(csr_matrix(text_stats))
            if "text_stats" not in self.feature_dimensions_:
                self.feature_dimensions_["text_stats"] = text_stats.shape[1]

        # Combine all features
        if len(features) == 1:
            return features[0]
        else:
            return hstack(features)

    def fit_transform(
        self, df: pd.DataFrame, text_column: str = "cleaned_text"
    ) -> csr_matrix:
        """Fit and transform in one step"""
        self.fit(df, text_column)
        return self.transform(df, text_column)

    def _extract_sentiment(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        """Extract VADER sentiment scores"""

        def get_sentiment_scores(text):
            scores = self.sia.polarity_scores(text)
            return [scores["neg"], scores["neu"], scores["pos"], scores["compound"]]

        sentiment_features = np.array(
            [get_sentiment_scores(text) for text in df[text_column]]
        )
        return sentiment_features

    def _extract_target(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target one-hot encoding"""
        target_dummies = pd.get_dummies(df["Target"], prefix="target")

        # Ensure all categories are present
        for cat in self.target_categories_:
            col_name = f"target_{cat}"
            if col_name not in target_dummies.columns:
                target_dummies[col_name] = 0

        # Sort columns for consistency
        target_dummies = target_dummies[sorted(target_dummies.columns)]
        return target_dummies.values

    def _extract_text_stats(self, df: pd.DataFrame, text_column: str) -> np.ndarray:
        """Extract text statistics"""
        # Use original Tweet column for punctuation if available
        original_col = "Tweet" if "Tweet" in df.columns else text_column

        stats = np.array(
            [
                df[text_column].str.len(),  # character length
                df[text_column].str.split().str.len(),  # word count
                df[original_col].str.count("!"),  # exclamation marks
                df[original_col].str.count("\\?"),  # question marks
                df[original_col].str.count("#"),  # hashtags
                df[original_col].str.count("@"),  # mentions
            ]
        ).T

        return stats

    def get_feature_summary(self) -> dict:
        """Get summary of features"""
        total = sum(self.feature_dimensions_.values())
        summary = {"total_features": total, "by_type": self.feature_dimensions_.copy()}
        return summary


def experiment_with_features(
    train_df: pd.DataFrame, test_df: pd.DataFrame, text_column: str = "cleaned_text"
) -> dict:
    """
    Experiment with different feature combinations

    Returns dict with feature matrices for different configurations
    """
    results = {}

    # Configuration 1: TF-IDF only (baseline)
    print("\n" + "=" * 60)
    print("Configuration 1: TF-IDF only (baseline)")
    print("=" * 60)
    extractor1 = FeatureExtractor(
        use_tfidf=True, use_sentiment=False, use_target=False, use_text_stats=False
    )
    X_train_1 = extractor1.fit_transform(train_df, text_column)
    X_test_1 = extractor1.transform(test_df, text_column)
    results["tfidf_only"] = {
        "X_train": X_train_1,
        "X_test": X_test_1,
        "extractor": extractor1,
    }
    print(f"Shape: {X_train_1.shape}")

    # Configuration 2: TF-IDF + Sentiment
    print("\n" + "=" * 60)
    print("Configuration 2: TF-IDF + Sentiment")
    print("=" * 60)
    extractor2 = FeatureExtractor(
        use_tfidf=True, use_sentiment=True, use_target=False, use_text_stats=False
    )
    X_train_2 = extractor2.fit_transform(train_df, text_column)
    X_test_2 = extractor2.transform(test_df, text_column)
    results["tfidf_sentiment"] = {
        "X_train": X_train_2,
        "X_test": X_test_2,
        "extractor": extractor2,
    }
    print(f"Shape: {X_train_2.shape}")

    # Configuration 3: TF-IDF + Target
    print("\n" + "=" * 60)
    print("Configuration 3: TF-IDF + Target")
    print("=" * 60)
    extractor3 = FeatureExtractor(
        use_tfidf=True, use_sentiment=False, use_target=True, use_text_stats=False
    )
    X_train_3 = extractor3.fit_transform(train_df, text_column)
    X_test_3 = extractor3.transform(test_df, text_column)
    results["tfidf_target"] = {
        "X_train": X_train_3,
        "X_test": X_test_3,
        "extractor": extractor3,
    }
    print(f"Shape: {X_train_3.shape}")

    # Configuration 4: All features
    print("\n" + "=" * 60)
    print("Configuration 4: All features")
    print("=" * 60)
    extractor4 = FeatureExtractor(
        use_tfidf=True, use_sentiment=True, use_target=True, use_text_stats=True
    )
    X_train_4 = extractor4.fit_transform(train_df, text_column)
    X_test_4 = extractor4.transform(test_df, text_column)
    results["all_features"] = {
        "X_train": X_train_4,
        "X_test": X_test_4,
        "extractor": extractor4,
    }
    print(f"Shape: {X_train_4.shape}")

    return results
