"""
Configuration management for the project
"""
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Create directories if they don't exist
for dir_path in [RESULTS_DIR, LOGS_DIR, EXPERIMENTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data paths
TRAIN_DATA = DATA_DIR / "train.csv"
TEST_DATA = DATA_DIR / "test.csv"

# Model configurations
MODEL_CONFIGS = {
    'logistic_regression': {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        'random_state': 42
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 30,
        'min_samples_split': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    },
    'knn': {
        'n_neighbors': 5,
        'weights': 'distance',
        'metric': 'cosine',
        'n_jobs': -1
    }
}

# Feature extraction configs
TFIDF_CONFIG = {
    'max_features': 3000,
    'ngram_range': (1, 1),
    'min_df': 2,
    'max_df': 0.9,
    'sublinear_tf': True
}

# Class labels
STANCE_LABELS = ['AGAINST', 'FAVOR', 'NONE']

# Random seed
RANDOM_SEED = 42
