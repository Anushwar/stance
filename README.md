# Stance Detection on Social Media

A methodical machine learning project for detecting stance (AGAINST, FAVOR, NONE) in tweets using traditional ML techniques. This project demonstrates proper ML engineering: clean architecture, iterative improvement, and evidence-based decisions.

## Overview

This project tackles stance detection on the SemEval-2016 Task 6 dataset, achieving **55.96% Macro F1** through systematic experimentation with Logistic Regression and careful feature engineering.

**Dataset:** 3,853 tweets across 5 targets (Atheism, Climate Change, Feminist Movement, Hillary Clinton, Abortion Legalization)

## Project Structure

```
stance/
├── src/
│   ├── data/
│   │   ├── preprocessor.py       # Text cleaning pipeline
│   │   └── feature_extractor.py  # Modular feature extraction
│   ├── evaluation/
│   │   └── metrics.py             # Evaluation & error analysis
│   └── utils/
│       └── config.py              # Configuration management
├── experiments/
│   └── iterative_improvement.py   # Systematic experiments
├── notebooks/
│   └── exploratory_analysis.ipynb # Interactive EDA
├── data/
│   ├── train.csv                  # Training data (2,647 samples)
│   └── test.csv                   # Test data (1,206 samples)
└── results/
    └── iterative_experiments.csv  # Experiment results
```

## Setup

```bash
# Create environment
conda create -n stance python=3.10
conda activate stance

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run exploratory analysis
jupyter notebook notebooks/exploratory_analysis.ipynb

# Run all experiments
python experiments/iterative_improvement.py
```

## Key Results

### Experiment Summary

| Experiment  | Features        | Macro F1   | Change | Verdict        |
| ----------- | --------------- | ---------- | ------ | -------------- |
| Baseline    | TF-IDF only     | 51.02%     | -      | Starting point |
| + SMOTE     | TF-IDF          | 53.44%     | +2.42% | Helps          |
| + Target    | TF-IDF + Target | **55.96%** | +2.52% | **Best**       |
| + Sentiment | + Sentiment     | 54.65%     | -1.31% | Hurts          |
| + Bigrams   | + Bigrams       | 52.04%     | -2.61% | Hurts          |

### Best Configuration

- **Model:** Logistic Regression with SMOTE balancing
- **Features:** TF-IDF (3,000 unigrams) + Target one-hot encoding
- **Performance:** 55.96% Macro F1, 62.44% Accuracy
- **Per-class F1:** AGAINST: 72.82% | FAVOR: 55.54% | NONE: 39.51%

## Key Findings

**What Works:**

- **Class balancing (SMOTE):** +2.42% F1 - Essential for handling 2.26:1 imbalance ratio
- **Target features:** +2.52% F1 - Stance patterns are target-dependent

**What Doesn't Work:**

- **Sentiment features:** -1.31% F1 - Stance ≠ Sentiment (30.9% of AGAINST tweets have POSITIVE sentiment)
- **Bigrams:** -2.61% F1 - Increases sparsity, overfits on limited data

### Critical Insight: Stance ≠ Sentiment

Our EDA revealed a counterintuitive finding:

- 30.9% of AGAINST tweets have POSITIVE sentiment
- 56.4% of FAVOR tweets have NEGATIVE sentiment

**Example:**

```
Tweet: "I will fight for the unborn!"
Sentiment: POSITIVE (determined, fighting spirit)
Stance: AGAINST (abortion legalization)
```

This validates why sentiment features hurt performance rather than help.

## Methodology

This project follows a systematic ML engineering approach:

1. **Deep EDA** - Understand data patterns before modeling
2. **Hypothesis formation** - Every feature has a reason
3. **Iterative testing** - Add one change at a time
4. **Measure impact** - Quantify each decision
5. **Evidence-based decisions** - Drop what doesn't help
