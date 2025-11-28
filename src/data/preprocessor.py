"""
Text preprocessing pipeline
Clean, normalize, and prepare text data for modeling
"""

import re
import pandas as pd
import numpy as np
from typing import List, Optional


class TextPreprocessor:
    """
    Handles all text preprocessing operations

    Philosophy: Keep it simple and interpretable.
    Only remove noise, preserve meaningful content.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtag_symbol: bool = True,
        remove_numbers: bool = False,
        min_length: int = 3,
    ):
        """
        Args:
            lowercase: Convert to lowercase
            remove_urls: Remove HTTP(S) URLs
            remove_mentions: Remove @mentions
            remove_hashtag_symbol: Remove # but keep the word
            remove_numbers: Remove all numbers
            min_length: Minimum tweet length after cleaning
        """
        self.lowercase = lowercase
        self.remove_urls = remove_urls
        self.remove_mentions = remove_mentions
        self.remove_hashtag_symbol = remove_hashtag_symbol
        self.remove_numbers = remove_numbers
        self.min_length = min_length

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string

        Args:
            text: Raw text string

        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Lowercase
        if self.lowercase:
            text = text.lower()

        # Remove URLs
        if self.remove_urls:
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove mentions
        if self.remove_mentions:
            text = re.sub(r"@\w+", "", text)

        # Remove hashtag symbol but keep word
        if self.remove_hashtag_symbol:
            text = re.sub(r"#", "", text)

        # Remove numbers (optional)
        if self.remove_numbers:
            text = re.sub(r"\d+", "", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text.strip()

    def clean_dataframe(
        self, df: pd.DataFrame, text_column: str = "Tweet"
    ) -> pd.DataFrame:
        """
        Clean text column in a dataframe

        Args:
            df: DataFrame with text column
            text_column: Name of column containing text

        Returns:
            DataFrame with additional 'cleaned_text' column
        """
        df = df.copy()
        df["cleaned_text"] = df[text_column].apply(self.clean_text)

        # Filter out very short texts
        if self.min_length > 0:
            initial_count = len(df)
            df = df[df["cleaned_text"].str.len() >= self.min_length].reset_index(
                drop=True
            )
            filtered_count = initial_count - len(df)
            if filtered_count > 0:
                print(
                    f"Filtered out {filtered_count} texts shorter than {self.min_length} characters"
                )

        return df

    def get_stats(self, df: pd.DataFrame, text_column: str = "cleaned_text") -> dict:
        """Get statistics about cleaned text"""
        stats = {
            "num_texts": len(df),
            "avg_length": df[text_column].str.len().mean(),
            "median_length": df[text_column].str.len().median(),
            "avg_words": df[text_column].str.split().str.len().mean(),
            "median_words": df[text_column].str.split().str.len().median(),
            "min_length": df[text_column].str.len().min(),
            "max_length": df[text_column].str.len().max(),
        }
        return stats


def load_and_preprocess(
    train_path: str, test_path: str, preprocessor: Optional[TextPreprocessor] = None
) -> tuple:
    """
    Load and preprocess train and test data

    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV
        preprocessor: TextPreprocessor instance (creates default if None)

    Returns:
        Tuple of (train_df, test_df) with cleaned text
    """
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Remove BOM if present
    train_df.columns = train_df.columns.str.replace("\ufeff", "")
    test_df.columns = test_df.columns.str.replace("\ufeff", "")

    # Create preprocessor if not provided
    if preprocessor is None:
        preprocessor = TextPreprocessor()

    # Clean
    train_df = preprocessor.clean_dataframe(train_df)
    test_df = preprocessor.clean_dataframe(test_df)

    print(f"\nLoaded and preprocessed:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Show stats
    print(f"\nText statistics (train):")
    stats = preprocessor.get_stats(train_df)
    for key, value in stats.items():
        print(
            f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    return train_df, test_df
