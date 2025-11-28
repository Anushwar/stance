"""
Evaluation metrics and error analysis
Focus on understanding WHY models fail, not just measuring performance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from typing import List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Comprehensive model evaluation with error analysis

    Philosophy: Metrics alone don't tell us how to improve.
    We need to understand WHAT the model gets wrong and WHY.
    """

    def __init__(self, label_names: List[str]):
        """
        Args:
            label_names: List of class names (e.g., ['AGAINST', 'FAVOR', 'NONE'])
        """
        self.label_names = label_names

    def evaluate(self, y_true, y_pred, dataset_name: str = "Test") -> dict:
        """
        Comprehensive evaluation

        Args:
            y_true: True labels
            y_pred: Predicted labels
            dataset_name: Name of dataset (for display)

        Returns:
            Dictionary with all metrics
        """
        results = {}

        # Overall metrics
        results["accuracy"] = accuracy_score(y_true, y_pred)
        results["macro_f1"] = f1_score(y_true, y_pred, average="macro")
        results["weighted_f1"] = f1_score(y_true, y_pred, average="weighted")

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(self.label_names)), average=None
        )

        results["per_class"] = {}
        for i, label in enumerate(self.label_names):
            results["per_class"][label] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": support[i],
            }

        # Confusion matrix
        results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # Print report
        print(f"\n{'=' * 70}")
        print(f"{dataset_name} Set Evaluation")
        print("=" * 70)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro F1: {results['macro_f1']:.4f}")
        print(f"Weighted F1: {results['weighted_f1']:.4f}")

        print(f"\nPer-Class Performance:")
        print("-" * 70)
        for label in self.label_names:
            metrics = results["per_class"][label]
            print(f"\n{label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  Support:   {metrics['support']}")

        print(f"\nConfusion Matrix:")
        print(results["confusion_matrix"])

        return results

    def analyze_errors(
        self, y_true, y_pred, texts: List[str], n_samples: int = 5
    ) -> pd.DataFrame:
        """
        Analyze misclassified examples

        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels (encoded)
            texts: Original text samples
            n_samples: Number of examples to show per error type

        Returns:
            DataFrame with error analysis
        """
        print(f"\n{'=' * 70}")
        print("ERROR ANALYSIS")
        print("=" * 70)

        errors = []
        misclassified_indices = np.where(y_true != y_pred)[0]

        print(
            f"\nTotal misclassified: {len(misclassified_indices)} / {len(y_true)} "
            f"({len(misclassified_indices) / len(y_true) * 100:.1f}%)"
        )

        # Analyze each type of error
        for true_label_idx in range(len(self.label_names)):
            for pred_label_idx in range(len(self.label_names)):
                if true_label_idx == pred_label_idx:
                    continue

                true_label = self.label_names[true_label_idx]
                pred_label = self.label_names[pred_label_idx]

                # Find examples of this error type
                error_mask = (y_true == true_label_idx) & (y_pred == pred_label_idx)
                error_indices = np.where(error_mask)[0]

                if len(error_indices) == 0:
                    continue

                error_rate = len(error_indices) / np.sum(y_true == true_label_idx) * 100

                print(
                    f"\n{true_label} → {pred_label}: {len(error_indices)} errors "
                    f"({error_rate:.1f}% of {true_label})"
                )

                # Show examples
                print(f"Examples:")
                for idx in error_indices[:n_samples]:
                    print(f"  • {texts[idx][:100]}...")
                    errors.append(
                        {
                            "true": true_label,
                            "predicted": pred_label,
                            "text": texts[idx],
                            "index": idx,
                        }
                    )

        return pd.DataFrame(errors)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            title: Plot title
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            cbar_kws={"label": "Count"},
            ax=ax,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✓ Saved confusion matrix: {save_path}")

        plt.close()

    def compare_models(
        self, results_list: List[dict], model_names: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple models

        Args:
            results_list: List of result dicts from evaluate()
            model_names: Names of models

        Returns:
            DataFrame with comparison
        """
        comparison = []

        for model_name, results in zip(model_names, results_list):
            row = {
                "Model": model_name,
                "Accuracy": results["accuracy"],
                "Macro_F1": results["macro_f1"],
            }

            # Add per-class F1
            for label in self.label_names:
                row[f"F1_{label}"] = results["per_class"][label]["f1"]

            comparison.append(row)

        df = pd.DataFrame(comparison)

        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        print(df.to_string(index=False))

        return df


def analyze_class_distribution(y_true, label_names: List[str]) -> None:
    """Analyze and visualize class distribution"""
    unique, counts = np.unique(y_true, return_counts=True)

    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION")
    print("=" * 60)

    total = len(y_true)
    for label_idx, count in zip(unique, counts):
        label = label_names[label_idx]
        pct = count / total * 100
        print(f"{label:10s}: {count:4d} ({pct:5.1f}%)")

    imbalance_ratio = counts.max() / counts.min()
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

    if imbalance_ratio > 2.0:
        print("⚠️  Significant class imbalance detected!")
        print("   Consider: SMOTE, class weights, or undersampling")
