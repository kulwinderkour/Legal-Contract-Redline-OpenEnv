---
title: Legal Contract Redline OpenEnv
emoji: 📋
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - legal
  - nlp
---

# Legal Contract Redline OpenEnv

## Overview

Legal contract review is one of the most time-consuming tasks in law — junior lawyers spend thousands of hours reading clauses, flagging risks, and suggesting rewrites. This RL environment simulates that workflow so AI agents can learn to perform **contract redlining**: identifying risky language in draft contracts, classifying the type of risk, and proposing safer alternatives.

**Who it's for:** RL researchers, legal-AI builders, and hackathon participants looking to benchmark language agents on structured legal reasoning tasks.

The environment exposes a simple REST API. An agent receives a contract clause, takes an action (classification + optional rewrite), and receives a graded reward based on a deterministic rubric — no LLM in the grading loop.

## Tasks

| Task | Description | Difficulty | Reward Logic |
|------|-------------|------------|--------------|
| **easy** | Binary classification — is the clause risky or safe? | 1 | 1.0 if correct, 0.0 if wrong |
| **medium** | Classify risk type and extract the specific risky phrase | 2 | Weighted: is_risky (0.3) + category (0.4) + phrase (0.3) |
| **hard** | Full redline: classify risk, extract phrase, and rewrite the clause | 3 | Weighted: is_risky (0.15) + category (0.25) + phrase (0.20) + rewrite (0.40) |

## Action Space

| Field | Type | Description |
|-------|------|-------------|
| `is_risky` | `boolean` | Whether the clause contains risky language |
| `risk_category` | `string` | One of: `liability`, `termination`, `ip_ownership`, `indemnification`, `governing_law`, `safe` |
| `risky_phrase` | `string` | The exact risky phrase extracted from the clause text, or `""` if safe |
| `rewrite` | `string` | A safe rewritten version of the clause, or `""` if not needed |

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `clause_id` | `integer` | Unique identifier for the clause |
| `clause_text` | `string` | The full text of the contract clause |
| `task` | `string` | Current task level: `easy`, `medium`, or `hard` |
| `step_number` | `integer` | Current step in the episode (0-indexed at reset) |
| `instructions` | `string` | Task-specific instructions for the agent |

## Reward Function

All rewards are in the range **[0.0, 1.0]** with partial credit.

### Easy Task
- **is_risky correct:** 1.0 — **is_risky wrong:** 0.0

### Medium Task (weighted sum = 1.0)
- **is_risky correct:** +0.3
- **risk_category correct:** +0.4
- **risky_phrase match:** +0.3 (full substring match = 1.0×, partial 3+ word overlap = 0.5×)

### Hard Task (weighted sum = 1.0)
- **is_risky correct:** +0.15
- **risk_category correct:** +0.25
- **risky_phrase match:** +0.20 (same matching logic as medium)
- **rewrite quality:** +0.40
  - 0.40 if rewrite >20 words AND risky phrase absent AND differs from original
  - 0.20 if rewrite >10 words AND differs from original
  - 0.10 if any non-empty rewrite provided
  - 0.00 if empty

Episodes end when the agent scores ≥0.8 or after 3 steps.

## API Endpoints

| Method | Path | Description | Example Body |
|--------|------|-------------|--------------|
| `GET` | `/` | Service info and available endpoints | — |
| `GET` | `/health` | Health check | — |
| `GET` | `/tasks` | List available tasks with descriptions | — |
| `POST` | `/reset` | Start a new episode | `{"task": "easy"}` |
| `POST` | `/step` | Submit an action and receive reward | `{"is_risky": true, "risk_category": "termination", "risky_phrase": "without notice", "rewrite": ""}` |
| `GET` | `/state` | Get current episode state | — |

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python server.py

# Test health endpoint
curl http://localhost:7860/health

# Reset with easy task
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task":"easy"}'

# Submit a step
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"is_risky":true,"risk_category":"termination","risky_phrase":"without notice","rewrite":""}'
```

## Docker

```bash
# Build the image
docker build -t legal-redline .

# Run the container
docker run -p 7860:7860 legal-redline
```

## Inference

The `inference.py` script runs an LLM agent against the environment for all 3 task levels.

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Run inference (server must be running)
python inference.py
```

The script outputs structured logs in the format:
```
[START] task=easy env=legal-redline model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=is_risky=True,category=termination reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```

## Baseline Scores

| Task | Avg Score | Success Rate |
|------|-----------|--------------|
| easy | 0.85 | 80% |
| medium | 0.62 | 55% |
| hard | 0.48 | 35% |
