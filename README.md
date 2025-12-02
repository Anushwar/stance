# Stance Detection on Social Media

Machine learning project for detecting stance (AGAINST, FAVOR, NONE) in tweets using traditional ML and transformer models on the SemEval-2016 Task 6 dataset.

## Setup

```bash
conda create -n stance python=3.10
conda activate stance
pip install -r requirements.txt
```

## Usage

### Web Application

```bash
python app.py
```

Open browser to `http://localhost:7860`

### Train Models

```bash
# Traditional ML (Logistic Regression, Random Forest, KNN)
python src/models/train_all_traditional_ml.py

# Transformer models (BERT, BERTweet, TwHIN-BERT)
python src/models/train_bert.py
python src/models/train_bertweet.py
python src/models/train_twhin_bert.py

# Large Language Models (LLMs) - Run Jupyter notebooks
jupyter notebook notebooks/llm_experiments/llama3_1_8B_lora_semeval_finetuned.ipynb
jupyter notebook notebooks/llm_experiments/mistral7B_LoRA_semeval.ipynb
jupyter notebook notebooks/llm_experiments/phi3_LoRA_semeval.ipynb
```

## Results

### Traditional ML Models

| Model | Macro F1 | Accuracy |
|-------|----------|----------|
| Logistic Regression | 55.96% | 62.44% |
| Random Forest | 52.74% | 63.43% |
| K-Nearest Neighbors | 41.01% | 41.13% |

**Best configuration:** TF-IDF (3K unigrams) + Target encoding + SMOTE balancing

### Large Language Models (LLMs)

| Model | Parameters | Training Method | Macro F1 | Accuracy |
|-------|-----------|-----------------|----------|----------|
| Mistral-7B | 7B | LoRA Fine-tuning | 77.53% | 80.02% |
| Llama 3.1 | 8B | LoRA Fine-tuning | 75.71% | 78.52% |
| Phi-3 | 3.8B | LoRA Fine-tuning | 73.18% | 75.46% |

**Training:** All LLMs use LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning on the SemEval-2016 dataset. Results show LLMs significantly outperform traditional ML and transformer models.

### Key Findings

- **SMOTE:** +2.42% F1 (handles class imbalance)
- **Target features:** +2.52% F1 (stance is target-dependent)
- **Sentiment features:** -1.31% F1 (stance ≠ sentiment)
- **Bigrams:** -2.61% F1 (increases sparsity)

## Dataset

SemEval-2016 Task 6: 3,853 tweets across 5 targets
- Training: 2,647 samples
- Test: 1,206 samples
- Targets: Atheism, Climate Change, Feminist Movement, Hillary Clinton, Abortion Legalization
