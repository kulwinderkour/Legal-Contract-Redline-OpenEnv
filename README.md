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

## 🌐 Live Demo

Try the environment here:

👉 https://kulludon123-legal-contract-redline.hf.space

Test it using the interactive API:

👉 https://kulludon123-legal-contract-redline.hf.space/docs

No setup needed — just open /docs and start testing.

Lawyers spend a ridiculous amount of time reading contract clauses and flagging risky language. This project turns that into an RL environment — give it a clause, have an agent analyze it, and get a graded score back. No API keys, no LLM, no setup headaches. Just run it.

It's a REST API that works with the [OpenEnv spec](https://github.com/open-env). Grading is all rule-based, rewards are always 0–1, and the same input always gives the same output.

---

## What It Does

Your agent gets a contract clause and has to:

1. **Easy** — is the clause risky or safe?
2. **Medium** — what kind of risk? Which phrase is the problem?
3. **Hard** — all of the above, plus rewrite the clause to make it safe

You get up to 3 attempts per episode. Each attempt is graded and scored.

---

## Project Files

```
server.py          — FastAPI server, all API endpoints
env.py             — Environment logic: clauses, grading, rewards
models.py          — Pydantic models for requests/responses
inference.py       — Baseline agent (rule-based, no API key needed)
test_env.py        — Self-test script (44 checks)
openenv.yaml       — OpenEnv manifest
Dockerfile         — Docker image for HF Spaces
requirements.txt   — Python dependencies
```

---

## API Endpoints

| Method | Path | What it does |
|--------|------|-------------|
| `GET` | `/health` | Returns `{"status": "ok"}` |
| `GET` | `/tasks` | Lists the 3 task levels |
| `POST` | `/reset` | Starts a new episode, returns a clause |
| `POST` | `/step` | Submits an action, returns reward + done |
| `GET` | `/state` | Current episode state |
| `GET` | `/metrics` | Aggregated scores across all episodes |

### POST /reset

Use the built-in dataset:
```json
{"task": "easy"}
```

Or paste your own clause (great for testing with real contracts):
```json
{
  "task": "easy",
  "clause_text": "Vendor may terminate without notice."
}
```

Custom clauses get `clause_id: -1`. The environment auto-labels them using keyword detection so it can still grade your response.

### POST /step

```json
{
  "is_risky": true,
  "risk_category": "termination",
  "risky_phrase": "without notice",
  "rewrite": ""
}
```

Returns:
```json
{
  "observation": { "clause_id": -1, "clause_text": "...", "task": "easy", "step_number": 1, "instructions": "..." },
  "reward": 1.0,
  "done": true,
  "info": { "breakdown": {"is_risky": 1.0} }
}
```

### Error Handling

The API won't crash on bad input. You'll get a clean error instead:

- `/step` before `/reset` → `400`
- `/step` after episode is done → `400`
- Invalid `risk_category` → `422`
- Invalid task name → `422`

---

## Reward Scoring

Rewards are always between 0.0 and 1.0. No randomness involved.

**Easy** — 1.0 if `is_risky` is correct, 0.0 if wrong.

**Medium** — weighted sum:
- `is_risky` correct: +0.3
- `risk_category` match: +0.4
- `risky_phrase` overlap: +0.3

**Hard** — weighted sum:
- `is_risky` correct: +0.15
- `risk_category` match: +0.25
- `risky_phrase` overlap: +0.20
- Rewrite quality: +0.40

Phrase scoring works on token overlap — exact match gets 1.0, partial overlap gets partial credit. Rewrite scoring checks that the risky phrase is actually gone and the text isn't just copied.

Episodes end when reward ≥ 0.8 or after 3 steps, whichever comes first.

---

## How to Run Locally

```bash
pip install -r requirements.txt
python server.py
```

That's it. Server starts at `http://localhost:7860`. Head to `/docs` to try the API in your browser.

### Run the baseline agent

```bash
python inference.py
```

No API key needed. It uses keyword matching to classify risk. Output looks like:

```
[START] task=easy env=legal-redline model=baseline
[STEP] step=1 action=is_risky=True,category=termination reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```

If you set `HF_TOKEN`, it'll use an LLM instead. If the LLM fails for any reason, it falls back to rule-based automatically — so it never crashes.

### Run the self-test

```bash
python test_env.py
```

Runs 44 checks — endpoints, error codes, reward ranges, custom clauses. If something's wrong, you'll know immediately.

---

## Custom Clause Support

This is probably the most fun part to play with. You can paste any contract clause into `/reset` and the environment will grade it:

```json
{
  "task": "medium",
  "clause_text": "The vendor shall not be liable for any damages under any circumstances."
}
```

Then call `/step` with your analysis. The environment figures out the "right answer" using keyword matching.

Here are some good ones to try:

| Clause | Expected |
|--------|----------|
| `"Vendor may terminate without notice."` | risky / termination |
| `"The vendor shall not be liable under any circumstances."` | risky / liability |
| `"The contractor shall indemnify the client against all claims."` | risky / indemnification |
| `"Payment is due within 30 days."` | safe |

---

## Docker / Hugging Face Spaces

```bash
docker build -t legal-redline .
docker run -p 7860:7860 legal-redline
```

Uses `python:3.11-slim`, non-root user, port 7860 — standard HF Spaces setup.

To deploy: push to a Hugging Face Space with Docker SDK selected. The frontmatter at the top of this file configures it automatically.

---

## Environment Variables

All optional. The server runs fine with zero config.

| Variable | Default | What it does |
|----------|---------|-------------|
| `HF_TOKEN` | — | Enables LLM mode in `inference.py` |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Which model to use |
| `ENV_URL` | `http://localhost:7860` | Where the server is running |

---

## What Makes This Different

- **No API keys required.** Everything works offline, right out of the box.
- **Fully local + deterministic.** Works even without internet — everything runs locally and gives the same result every time.
- **Paste any clause.** Judges can try their own contract text and get graded instantly.
- **Fully reproducible.** Same input, same output, every time.
- **Partial credit.** You don't just get right/wrong — close answers get partial scores.
- **Won't crash.** Bad input gives you a proper error code, not a stack trace.

---

## Quick Judge Testing Guide

The fastest way to see what this does:

1. `python server.py`
2. Open `http://localhost:7860/docs`
3. Hit `POST /reset` with `{"task": "easy"}`
4. Or try your own clause: `{"task": "easy", "clause_text": "Vendor may terminate without notice."}`
5. 👉 You can also paste your own contract clause using the `clause_text` field in `/reset`.
6. Hit `POST /step` with `{"is_risky": true, "risk_category": "termination"}`
7. Run `python inference.py` to watch the baseline agent go
8. Run `python test_env.py` to make sure everything passes

---

## License

Built for the OpenEnv hackathon. Free to use for research and experimentation.

> If something breaks, just run `test_env.py` — it will quickly show what's wrong.
