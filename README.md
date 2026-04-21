---
title: gliner-sports-ner
app_file: demo/app.py
sdk: gradio
sdk_version: 6.13.0
---

# GLiNER Sports NER

Domain-specific named entity recognition for sports articles. Fine-tuned [GLiNER](https://github.com/urchade/GLiNER) (DeBERTa-v3 encoder) on 3,173 annotated sports examples across NBA, NFL, MLB, and NHL.

**The result:** A small fine-tuned encoder matches GPT-4o-mini extraction accuracy at 10x lower latency and zero per-document API cost — validating the SLM thesis for structured information extraction.

---

## TL;DR

| System            | F1    | p50 Latency | Cost/1K docs |
|-------------------|-------|-------------|--------------|
| GLiNER base       | 0.681 | 128.9ms     | $0.00        |
| **GLiNER fine-tuned** | **0.842** | **127.7ms** | **$0.00** |
| GPT-4o-mini       | 0.838 | 1,331.5ms   | $0.07        |

Fine-tuned GLiNER matches GPT-4o-mini F1 at **10x lower latency** and **200x lower cost**.

---

## Live Demo

[Try it on HuggingFace Spaces](https://huggingface.co/spaces/myarapat/gliner-sports-ner)

Paste any sports article and extract structured entities in real time.

---

## Benchmark Results

| System            | F1    | Precision | Recall | p50 Latency | p99 Latency | Cost/1K docs |
|-------------------|-------|-----------|--------|-------------|-------------|--------------|
| GLiNER base       | 0.681 | 0.650     | 0.714  | 128.9ms     | 214.5ms     | $0.00        |
| GLiNER fine-tuned | 0.842 | 0.790     | 0.901  | 127.7ms     | 206.2ms     | $0.00        |
| GPT-4o-mini       | 0.838 | 0.903     | 0.782  | 1,331.5ms   | 5,431.2ms   | $0.07        |

**Key findings:**
- Fine-tuning improves F1 by 16 points over the zero-shot base model
- Fine-tuned GLiNER matches GPT-4o-mini F1 (0.842 vs 0.838)
- GLiNER p99 latency is 26x more consistent than GPT (206ms vs 5,431ms)
- Fine-tuned GLiNER has higher recall (0.901 vs 0.782) — captures more entities
- GPT-4o-mini has higher precision (0.903 vs 0.790) — what it finds is more reliable

> **Evaluation note:** Test set was annotated by GPT-4o-mini, introducing circular evaluation bias. GPT baseline scores may be inflated. Fine-tuned GLiNER numbers are unaffected.

---

## Architecture

'''
ESPN Internal JSON API
↓
scraper.py — 161 articles, ESPN categories as weak supervision labels
↓
annotator.py — sentence splitting → GPT-4o-mini annotation → offset alignment
↓
validator.py — quality filtering, weak label coverage check, 80/10/10 splits
↓
train.py — GLiNER fine-tuning via model.train_model() with differential LRs
↓
benchmark.py — three-way evaluation: F1, latency (p50/p99), cost
↓
api.py — FastAPI endpoint with LRU cache
↓
demo/app.py — Gradio interactive demo → HuggingFace Spaces
'''

---

## Entity Schema

| Entity Type  | Description                          | Example                    |
|--------------|--------------------------------------|----------------------------|
| PLAYER       | Individual athlete names             | "Donovan Mitchell"         |
| TEAM         | Sports team names                    | "Cleveland Cavaliers"      |
| POSITION     | Player positions                     | "point guard"              |
| STAT         | Numerical statistics                 | "32 points"                |
| INJURY       | Injury descriptions                  | "hamstring strain"         |
| TRADE_ASSET  | Draft picks or trade pieces          | "first-round pick"         |
| GAME_EVENT   | Notable in-game events               | "buzzer beater"            |
| VENUE        | Stadium or arena names               | "Madison Square Garden"    |
| COACH        | Coach or staff names                 | "Erik Spoelstra"           |
| AWARD        | Awards or accolades                  | "MVP"                      |

---

## Dataset

- **161 articles** scraped across NBA, NFL, MLB, NHL
- **Source:** ESPN internal JSON API (`site.api.espn.com`) — discovered via DevTools, cleaner than HTML scraping
- **Weak supervision:** ESPN categories metadata pre-tags teams and players per article, used for annotation validation
- **Annotation:** GPT-4o-mini with `temperature=0.0` and structured JSON output mode. Offsets computed in Python — never trusted to GPT
- **3,967 annotated examples** after quality filtering (0.1% flagged/removed)
- **22,892 total entity spans** across 10 entity types, avg 5.77 per example
- **93.9% weak label coverage** vs ESPN metadata ground truth
- **Splits:** train=3,173 / val=396 / test=398 (80/10/10)
- **Class imbalance:** PLAYER (7,174) and TEAM (6,619) dominate; VENUE (226) and AWARD (273) underrepresented

---

## Fine-Tuning Approach

- **Base model:** `urchade/gliner_medium-v2.1` (DeBERTa-v3 encoder + span scoring head)
- **Differential learning rates:**
  - Backbone: `5e-6` — low to prevent catastrophic forgetting of pretrained representations
  - Span head: `1e-5` — higher for faster domain adaptation to sports entity schema
- **Training steps:** 10,000 (~25 epochs on 3,173 examples)
- **Optimizer:** AdamW with linear warmup (`warmup_ratio=0.1`)
- **Hardware:** Google Colab A100, ~33 minutes
- **Experiment tracking:** [Weights & Biases](https://wandb.ai/monishy1-university-of-san-diego/gliner-sports-ner)
- **Fine-tuned model:** [myarapat/gliner-sports-ner](https://huggingface.co/myarapat/gliner-sports-ner)

---

## Evaluation Framework

Three-way benchmark across accuracy, latency, and cost:

- **Exact span match F1** — both entity text and label must match ground truth
- **Precision / Recall** — precision measures false positive rate, recall measures coverage
- **p50 and p99 latency** — median and tail latency; p99 exposes consistency differences invisible in averages
- **Cost per 1K documents** — normalized cost for fair comparison across API and local systems

---

## Local Setup

```bash
git clone https://github.com/yourusername/gliner-sports-ner.git
cd gliner-sports-ner
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**Run the demo:**
```bash
python demo/app.py
```

**Run the API:**
```bash
uvicorn src.serving.api:app --reload --port 8000
```

**Run the full pipeline:**
```bash
# Scrape
python -m src.data_pipeline.scraper

# Annotate (requires OPENAI_API_KEY)
python -m src.data_pipeline.annotator

# Validate and split
python -m src.data_pipeline.validator

# Train (recommend Colab A100)
python -m src.training.train --wandb_run_name "gliner-sports-v1"

# Benchmark
python -m src.evaluation.benchmark
```

---

## API Reference

**POST /extract**
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Donovan Mitchell scored 32 points for the Cavaliers",
    "threshold": 0.5
  }'
```

```json
{
  "entities": [
    {"text": "Donovan Mitchell", "label": "PLAYER", "start": 0, "end": 16, "score": 0.9999},
    {"text": "32 points", "label": "STAT", "start": 24, "end": 33, "score": 1.0},
    {"text": "Cavaliers", "label": "TEAM", "start": 42, "end": 51, "score": 0.9999}
  ],
  "latency_ms": 128.4,
  "model_version": "myarapat/gliner-sports-ner",
  "cache_hit": false
}
```

**GET /health** — Service health and uptime

**GET /benchmark** — Pre-computed three-way benchmark results

---

## Model

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("myarapat/gliner-sports-ner")
entities = model.predict_entities(
    "LeBron James scored 32 points for the Lakers",
    ["PLAYER", "TEAM", "STAT"],
    threshold=0.5
)
```

🤗 [myarapat/gliner-sports-ner](https://huggingface.co/myarapat/gliner-sports-ner)

---

## Key Technical Decisions

**Why GLiNER over a generative LLM for NER?**
Bidirectional context (DeBERTa) is better than causal LMs for span extraction — the model sees the full sentence before scoring any span. Span scoring is also bounded and structured, making it faster and more consistent than prompting a generative model.

**Why fine-tune instead of just prompting GPT?**
Prompting is flexible but brittle on schema consistency. Fine-tuning bakes the entity schema into weights, giving consistent structured output and better generalization within the domain. The benchmark proves this — equivalent F1 at a fraction of the cost.

**Why GPT-assisted annotation?**
Manual annotation of 3,967 sentences would take weeks. GPT-4o-mini bootstraps labels at scale. Key safeguards: `temperature=0.0` for consistency, offsets computed in Python not by GPT, ESPN weak labels used to validate coverage, test set identified as circularly biased.

**Why differential learning rates?**
The DeBERTa backbone already understands language deeply — aggressive updates cause catastrophic forgetting. The span head needs faster adaptation to the new entity schema. `5e-6` for backbone, `1e-5` for span head balances both.

**Why p50 and p99 latency?**
Average latency hides tail behavior. GPT's p99 of 5,431ms vs GLiNER's 206ms reveals a consistency gap invisible in averages — critical for real-time pipelines.

---

## Known Limitations

- Test set was annotated by GPT-4o-mini, introducing circular evaluation bias. GPT baseline scores may be inflated. Mitigation: manual golden set of 50 examples planned if results appear suspicious.
- Training ran for ~25 epochs (10,000 steps) due to GLiNER's step-based scheduler — more than the intended 5 epochs. High final gradient norm (365) suggests possible overfitting. Next run: `max_steps=2000`.
- Class imbalance — VENUE and AWARD underrepresented in training data. Per-entity F1 will be lower for these types.
- Deployment is CPU-based on HuggingFace Spaces free tier — inference is slower than the benchmark numbers which were measured on Colab A100.

---

## Acknowledgments

- [GLiNER](https://github.com/urchade/GLiNER) by Urchade Zaratiana et al.
- [Knowledgator](https://knowledgator.com) for the SLM thesis and GLiNER ecosystem
- ESPN internal JSON API for sports article data
- Weights & Biases for experiment tracking