# Mental Health Risk Detection from Social Media

Detecting mental health conditions from Reddit posts using sequential deep learning models. This project explores whether temporal patterns in user posting behavior can enable earlier detection compared to simple aggregation methods.

## Motivation

Mental health conditions often go undiagnosed for years. Social media platforms contain linguistic signals that could enable earlier intervention — but most existing tools just analyze single posts in isolation. I wanted to test whether modeling the *sequence* of someone's posts over time could detect risk earlier, with less data.

The core question: **Can sequential models (BiLSTM, Transformer) outperform simple baselines for early detection when we only have a few posts?**

Spoiler: The answer surprised me.

## Dataset

Reddit Mental Health Posts from HuggingFace ([solomonk/reddit_mental_health_posts](https://huggingface.co/datasets/solomonk/reddit_mental_health_posts))

| Metric | Value |
|--------|-------|
| Total Users | 2,558 |
| Total Posts | 78,530 |
| Avg Posts/User | 9.4 |
| Categories | 5 |

**Class Distribution:**

| Condition | Users | Posts |
|-----------|-------|-------|
| OCD | 960 | 27,285 |
| Aspergers | 567 | 14,148 |
| ADHD | 456 | 13,771 |
| PTSD | 359 | 13,227 |
| Depression | 216 | 10,099 |

Labels come from the subreddit where users posted (r/OCD, r/depression, etc). Not clinical diagnoses, but a reasonable proxy for research purposes.

## Models

I trained three models to compare:

### 1. Baseline: Mean Pooling + Logistic Regression
- Averages all post embeddings into a single vector
- Simple sklearn LogisticRegression on top
- No sequential information — just "what topics does this person talk about?"

### 2. BiLSTM + Attention
- Bidirectional LSTM processes posts in chronological order
- Attention mechanism learns which posts matter most
- Captures how language patterns evolve over time

### 3. Transformer Encoder
- Self-attention across all posts simultaneously
- Positional encoding preserves temporal order
- Can directly attend to distant posts without sequential bottleneck

All models use Sentence-BERT (all-MiniLM-L6-v2) to encode each post into a 384-dimensional vector.


## Why These Models?

The hypothesis was that sequential models would shine for *early detection* — when you only have 2-3 posts from a user, understanding the progression might matter more than just averaging.

I based this on recent papers showing BiLSTM + attention outperforming simpler methods:

- Bin Saeed & Cha (2025) — "Multi-modal deep-attention-BiLSTM based early detection of mental health issues" (Scientific Reports)
- Ji et al. (2021) — "Suicidal ideation detection with attentive relation networks" (Neural Computing and Applications)
- Al-Mosaiwi & Johnstone (2018) — Absolutist thinking patterns in mental health conditions

The eRisk workshop (CLEF) has been running early detection challenges since 2017, establishing that temporal modeling matters — at least in theory.

## Results

### Full History (All Posts)

| Model | Accuracy | F1 (Macro) | Std Dev |
|-------|----------|------------|---------|
| Baseline | 93.0% | 0.92 | ±0.3% |
| BiLSTM + Attention | **93.8%** | **0.94** | ±0.4% |
| Transformer | 91.1% | 0.90 | ±0.5% |

BiLSTM wins, but the margin over baseline (~0.8%) is within noise range. Would need more runs or larger test set to claim statistical significance.

### Early Detection (Partial History)

This is the interesting part. I evaluated each model using only the first 25%, 50%, 75% of each user's posts:

| History | Baseline | BiLSTM | Transformer |
|---------|----------|--------|-------------|
| 25% (~2 posts) | **84.1%** | 81.2% | 74.7% |
| 50% (~4 posts) | **89.2%** | 87.8% | 83.5% |
| 75% (~7 posts) | 91.5% | **91.8%** | 88.9% |
| 100% (~9 posts) | 93.0% | **93.8%** | 91.1% |

**The baseline wins at early detection.** With only 2 posts, mean pooling + logistic regression beats both deep learning models by ~3%.

### Per-Class Performance (BiLSTM, Full History)

| Condition | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| OCD | 0.98 | 0.98 | 0.98 |
| PTSD | 0.96 | 0.95 | 0.96 |
| Aspergers | 0.94 | 0.93 | 0.93 |
| Depression | 0.93 | 0.94 | 0.94 |
| ADHD | 0.88 | 0.89 | 0.88 |

ADHD is hardest to classify — probably because ADHD-related language (focus issues, medication, frustration) overlaps with other conditions.

## What I Learned

**1. Simple baselines matter.** I expected BiLSTM/Transformer to dominate early detection. They didn't. The baseline's robustness with sparse data was the biggest surprise.

**2. Sequential models need sufficient history.** BiLSTM only beats baseline at 75%+ history. With 2-3 posts, there's not enough sequence to learn from.

**3. Transformers need more data.** The Transformer consistently underperformed BiLSTM, likely because the dataset (2.5k users) is too small for self-attention to learn meaningful patterns. I tried:
- Smaller model (1 layer, 64 hidden) — still underperformed
- Different positional encodings (learned vs sinusoidal) — no significant difference
- More aggressive dropout (0.5) — helped slightly but still behind BiLSTM

**4. Class imbalance hurts.** Depression has the fewest users (216) and shows more variance across runs than other classes.

**5. Statistical significance matters.** The 0.8% gap between BiLSTM and baseline isn't convincing. In a real setting, I'd run more seeds and compute confidence intervals before claiming one model is better.

If I were building an actual early warning system, I'd probably use the simple baseline for new users and switch to BiLSTM once they have 7+ posts.

## Linguistic Risk Scoring

Beyond classification, I implemented a rule-based risk scorer based on research-backed linguistic markers:

- **Absolutist language** (always, never, nothing, completely) — linked to depression/anxiety per Al-Mosaiwi & Johnstone (2018)
- **Negative emotion words** (hopeless, worthless, exhausted)
- **Self-focused language** (I, me, my) — high self-reference correlates with depression
- **Cognitive distortions** (should, must, can't, ruined)
- **Crisis language** (explicit mentions of self-harm, suicide)

Each category contributes to a 0-100 risk score. This runs alongside the neural model in the Streamlit app.

## Project Structure

```
├── app.py                          # Streamlit web interface
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA
│   ├── 02_build_timelines.ipynb    # User timeline construction
│   ├── 03_create_embeddings.ipynb  # SBERT encoding
│   ├── 04_baseline_model.ipynb     # Mean pooling + LogReg
│   ├── 05_bilstm_model.ipynb       # BiLSTM + Attention
│   ├── 06_transformer_model.ipynb  # Transformer encoder
│   └── 07_final_summary.ipynb      # Results compilation
├── results/
│   ├── bilstm_model.pth            # Trained BiLSTM weights
│   ├── baseline_results.pkl
│   ├── bilstm_results.pkl
│   └── transformer_results.pkl
├── data/                           # Not included (download from HuggingFace)
└── requirements.txt
```

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

Make sure `results/bilstm_model.pth` exists. If not, run the training notebooks first.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- sentence-transformers
- streamlit
- scikit-learn
- pandas, numpy, matplotlib


## References

**Papers:**
- Bin Saeed, Q., Cha, Y. (2025). Multi-modal deep-attention-BiLSTM based early detection of mental health issues using social media posts. *Scientific Reports*.
- Al-Mosaiwi, M., Johnstone, T. (2018). In an absolute state: Elevated use of absolutist words is a marker specific to anxiety, depression, and suicidal ideation. *Clinical Psychological Science*.
- Ji, S., et al. (2021). Suicidal ideation and mental disorder detection with attentive relation networks. *Neural Computing and Applications*.
- Pennebaker, J.W. (2011). The secret life of pronouns. *Bloomsbury Press*.

**Dataset:**
- solomonk/reddit_mental_health_posts on HuggingFace

## Limitations

This is a research project, not a clinical tool.

- Labels are subreddit-based, not clinical diagnoses
- Single platform (Reddit) with specific demographics
- Can't capture context, sarcasm, or intent
- Conditions have overlapping symptoms
- Small sample for some classes (Depression: 216 users)
- Results may not generalize to other platforms or languages

---

Built for my ML portfolio. The goal was demonstrating proper experimental design — baselines, ablations, honest evaluation — not chasing SOTA numbers.
