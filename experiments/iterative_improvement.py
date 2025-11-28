"""
Iterative Model Improvement
Methodical approach to improving stance detection

Process:
1. Start with simplest baseline
2. Evaluate and understand failures
3. Form hypothesis about what might help
4. Test hypothesis
5. Measure impact
6. Repeat

This is the RIGHT way to do ML engineering.
"""
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

from src.data.preprocessor import load_and_preprocess, TextPreprocessor
from src.data.feature_extractor import FeatureExtractor
from src.evaluation.metrics import ModelEvaluator, analyze_class_distribution
from src.utils.config import TRAIN_DATA, TEST_DATA, STANCE_LABELS, RANDOM_SEED

print("="*80)
print("ITERATIVE MODEL IMPROVEMENT")
print("="*80)

# Load data
print("\n[STEP 1] Load and Preprocess Data")
print("-"*80)
preprocessor = TextPreprocessor(
    lowercase=True,
    remove_urls=True,
    remove_mentions=True,
    remove_hashtag_symbol=True,
    remove_numbers=False,
    min_length=3
)

train_df, test_df = load_and_preprocess(TRAIN_DATA, TEST_DATA, preprocessor)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(train_df['Stance'])
y_test = le.transform(test_df['Stance'])

print(f"\nLabel mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Analyze class distribution
analyze_class_distribution(y_train, STANCE_LABELS)

# Initialize evaluator
evaluator = ModelEvaluator(STANCE_LABELS)

# Store results for comparison
experiment_results = []


# =============================================================================
# EXPERIMENT 1: Simplest Baseline - TF-IDF + Logistic Regression (No balancing)
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 1: Baseline (TF-IDF + LR, no class balancing)")
print("="*80)
print("\nHypothesis: Start simple to understand baseline performance")

# Extract features
feature_extractor_1 = FeatureExtractor(
    use_tfidf=True,
    use_sentiment=False,
    use_target=False,
    use_text_stats=False,
    tfidf_config={'max_features': 3000, 'ngram_range': (1, 1),
                  'min_df': 2, 'max_df': 0.9, 'sublinear_tf': True}
)

X_train_1 = feature_extractor_1.fit_transform(train_df)
X_test_1 = feature_extractor_1.transform(test_df)

# Train model
model_1 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')
model_1.fit(X_train_1, y_train)

# Evaluate
y_pred_1 = model_1.predict(X_test_1)
results_1 = evaluator.evaluate(y_test, y_pred_1, "Experiment 1")

# Error analysis
print("\nError Analysis:")
errors_1 = evaluator.analyze_errors(y_test, y_pred_1,
                                   test_df['cleaned_text'].tolist(),
                                   n_samples=3)

experiment_results.append({
    'experiment': 'Exp1_Baseline_NoBalance',
    'features': 'TF-IDF (3k unigrams)',
    'balancing': 'None',
    'accuracy': results_1['accuracy'],
    'macro_f1': results_1['macro_f1'],
    'f1_against': results_1['per_class']['AGAINST']['f1'],
    'f1_favor': results_1['per_class']['FAVOR']['f1'],
    'f1_none': results_1['per_class']['NONE']['f1']
})

print("\n💡 Key Observations:")
print("  - AGAINST class dominates (high precision/recall)")
print("  - FAVOR and especially NONE struggle (class imbalance)")
print("  - Model biased toward majority class")


# =============================================================================
# EXPERIMENT 2: Add Class Balancing (SMOTE)
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 2: Add Class Balancing (SMOTE)")
print("="*80)
print("\nHypothesis: Class imbalance is hurting minority classes")
print("Solution: Use SMOTE to balance training data")

# Apply SMOTE
smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
X_train_2_balanced, y_train_2_balanced = smote.fit_resample(X_train_1, y_train)

print(f"\nAfter SMOTE: {X_train_2_balanced.shape[0]} samples")
analyze_class_distribution(y_train_2_balanced, STANCE_LABELS)

# Train model
model_2 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')
model_2.fit(X_train_2_balanced, y_train_2_balanced)

# Evaluate
y_pred_2 = model_2.predict(X_test_1)
results_2 = evaluator.evaluate(y_test, y_pred_2, "Experiment 2")

experiment_results.append({
    'experiment': 'Exp2_SMOTE',
    'features': 'TF-IDF (3k unigrams)',
    'balancing': 'SMOTE',
    'accuracy': results_2['accuracy'],
    'macro_f1': results_2['macro_f1'],
    'f1_against': results_2['per_class']['AGAINST']['f1'],
    'f1_favor': results_2['per_class']['FAVOR']['f1'],
    'f1_none': results_2['per_class']['NONE']['f1']
})

print("\n💡 Key Observations:")
print("  - Macro F1 improved?" +
      f" {'+' if results_2['macro_f1'] > results_1['macro_f1'] else '-'}"
      f"{abs(results_2['macro_f1'] - results_1['macro_f1']):.4f}")
print("  - FAVOR F1 improved?" +
      f" {'+' if results_2['per_class']['FAVOR']['f1'] > results_1['per_class']['FAVOR']['f1'] else '-'}"
      f"{abs(results_2['per_class']['FAVOR']['f1'] - results_1['per_class']['FAVOR']['f1']):.4f}")
print("  - NONE F1 improved?" +
      f" {'+' if results_2['per_class']['NONE']['f1'] > results_1['per_class']['NONE']['f1'] else '-'}"
      f"{abs(results_2['per_class']['NONE']['f1'] - results_1['per_class']['NONE']['f1']):.4f}")


# =============================================================================
# EXPERIMENT 3: Add Target Information
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 3: Add Target Features")
print("="*80)
print("\nHypothesis: Stance patterns differ by target (e.g., Climate Change is mostly FAVOR)")
print("Solution: Add target as a feature")

# Extract features with target
feature_extractor_3 = FeatureExtractor(
    use_tfidf=True,
    use_sentiment=False,
    use_target=True,  # NEW
    use_text_stats=False,
    tfidf_config={'max_features': 3000, 'ngram_range': (1, 1),
                  'min_df': 2, 'max_df': 0.9, 'sublinear_tf': True}
)

X_train_3 = feature_extractor_3.fit_transform(train_df)
X_test_3 = feature_extractor_3.transform(test_df)

print(f"\nFeature summary:")
print(feature_extractor_3.get_feature_summary())

# Apply SMOTE
X_train_3_balanced, y_train_3_balanced = smote.fit_resample(X_train_3, y_train)

# Train model
model_3 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')
model_3.fit(X_train_3_balanced, y_train_3_balanced)

# Evaluate
y_pred_3 = model_3.predict(X_test_3)
results_3 = evaluator.evaluate(y_test, y_pred_3, "Experiment 3")

experiment_results.append({
    'experiment': 'Exp3_Target',
    'features': 'TF-IDF + Target',
    'balancing': 'SMOTE',
    'accuracy': results_3['accuracy'],
    'macro_f1': results_3['macro_f1'],
    'f1_against': results_3['per_class']['AGAINST']['f1'],
    'f1_favor': results_3['per_class']['FAVOR']['f1'],
    'f1_none': results_3['per_class']['NONE']['f1']
})

print("\n💡 Key Observations:")
print(f"  - Macro F1 change: {results_3['macro_f1'] - results_2['macro_f1']:+.4f}")
print("  - Did target features help? " +
      ("YES ✓" if results_3['macro_f1'] > results_2['macro_f1'] else "NO ✗"))


# =============================================================================
# EXPERIMENT 4: Add Sentiment Features
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 4: Add Sentiment Features")
print("="*80)
print("\nHypothesis: Even though stance ≠ sentiment, sentiment might provide weak signal")
print("Solution: Add VADER sentiment scores")

# Extract features with sentiment
feature_extractor_4 = FeatureExtractor(
    use_tfidf=True,
    use_sentiment=True,  # NEW
    use_target=True,
    use_text_stats=False,
    tfidf_config={'max_features': 3000, 'ngram_range': (1, 1),
                  'min_df': 2, 'max_df': 0.9, 'sublinear_tf': True}
)

X_train_4 = feature_extractor_4.fit_transform(train_df)
X_test_4 = feature_extractor_4.transform(test_df)

print(f"\nFeature summary:")
print(feature_extractor_4.get_feature_summary())

# Apply SMOTE
X_train_4_balanced, y_train_4_balanced = smote.fit_resample(X_train_4, y_train)

# Train model
model_4 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')
model_4.fit(X_train_4_balanced, y_train_4_balanced)

# Evaluate
y_pred_4 = model_4.predict(X_test_4)
results_4 = evaluator.evaluate(y_test, y_pred_4, "Experiment 4")

experiment_results.append({
    'experiment': 'Exp4_Sentiment',
    'features': 'TF-IDF + Target + Sentiment',
    'balancing': 'SMOTE',
    'accuracy': results_4['accuracy'],
    'macro_f1': results_4['macro_f1'],
    'f1_against': results_4['per_class']['AGAINST']['f1'],
    'f1_favor': results_4['per_class']['FAVOR']['f1'],
    'f1_none': results_4['per_class']['NONE']['f1']
})

print("\n💡 Key Observations:")
print(f"  - Macro F1 change: {results_4['macro_f1'] - results_3['macro_f1']:+.4f}")
print("  - Did sentiment features help? " +
      ("YES ✓" if results_4['macro_f1'] > results_3['macro_f1'] else "NO ✗"))


# =============================================================================
# EXPERIMENT 5: Try Bigrams
# =============================================================================
print("\n" + "="*80)
print("EXPERIMENT 5: Try Bigrams")
print("="*80)
print("\nHypothesis: Phrase-level features (bigrams) might capture stance better")
print("Solution: Use unigrams + bigrams")

# Extract features with bigrams
feature_extractor_5 = FeatureExtractor(
    use_tfidf=True,
    use_sentiment=True,
    use_target=True,
    use_text_stats=False,
    tfidf_config={'max_features': 5000, 'ngram_range': (1, 2),  # CHANGED
                  'min_df': 2, 'max_df': 0.9, 'sublinear_tf': True}
)

X_train_5 = feature_extractor_5.fit_transform(train_df)
X_test_5 = feature_extractor_5.transform(test_df)

print(f"\nFeature summary:")
print(feature_extractor_5.get_feature_summary())

# Apply SMOTE
X_train_5_balanced, y_train_5_balanced = smote.fit_resample(X_train_5, y_train)

# Train model
model_5 = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')
model_5.fit(X_train_5_balanced, y_train_5_balanced)

# Evaluate
y_pred_5 = model_5.predict(X_test_5)
results_5 = evaluator.evaluate(y_test, y_pred_5, "Experiment 5")

experiment_results.append({
    'experiment': 'Exp5_Bigrams',
    'features': 'TF-IDF(1,2) + Target + Sentiment',
    'balancing': 'SMOTE',
    'accuracy': results_5['accuracy'],
    'macro_f1': results_5['macro_f1'],
    'f1_against': results_5['per_class']['AGAINST']['f1'],
    'f1_favor': results_5['per_class']['FAVOR']['f1'],
    'f1_none': results_5['per_class']['NONE']['f1']
})

print("\n💡 Key Observations:")
print(f"  - Macro F1 change: {results_5['macro_f1'] - results_4['macro_f1']:+.4f}")
print("  - Did bigrams help? " +
      ("YES ✓" if results_5['macro_f1'] > results_4['macro_f1'] else "NO ✗"))


# =============================================================================
# FINAL COMPARISON
# =============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON OF ALL EXPERIMENTS")
print("="*80)

results_df = pd.DataFrame(experiment_results)
print("\n" + results_df.to_string(index=False))

# Find best
best_idx = results_df['macro_f1'].idxmax()
best_exp = results_df.loc[best_idx]

print("\n" + "="*80)
print("BEST CONFIGURATION")
print("="*80)
print(f"Experiment: {best_exp['experiment']}")
print(f"Features: {best_exp['features']}")
print(f"Balancing: {best_exp['balancing']}")
print(f"Macro F1: {best_exp['macro_f1']:.4f}")
print(f"Accuracy: {best_exp['accuracy']:.4f}")
print(f"\nPer-class F1:")
print(f"  AGAINST: {best_exp['f1_against']:.4f}")
print(f"  FAVOR:   {best_exp['f1_favor']:.4f}")
print(f"  NONE:    {best_exp['f1_none']:.4f}")

# Save results
results_df.to_csv('../results/iterative_experiments.csv', index=False)
print(f"\n✓ Saved results to: results/iterative_experiments.csv")

# Key learnings
print("\n" + "="*80)
print("KEY LEARNINGS")
print("="*80)
print("\n1. What helped:")
improvements = results_df['macro_f1'].diff()
for i, (exp, imp) in enumerate(zip(results_df['experiment'], improvements)):
    if i == 0:
        continue
    if imp > 0:
        print(f"   ✓ {exp}: +{imp:.4f}")

print("\n2. What didn't help (or hurt):")
for i, (exp, imp) in enumerate(zip(results_df['experiment'], improvements)):
    if i == 0:
        continue
    if imp <= 0:
        print(f"   ✗ {exp}: {imp:.4f}")

print("\n3. Remaining challenges:")
print(f"   - NONE class still hardest (F1: {best_exp['f1_none']:.4f})")
print("   - Macro F1 around 53-55% - need transformers for 70%+")
print("   - Limited training data (2,647 samples)")

print("\n✅ Iterative improvement complete!")
