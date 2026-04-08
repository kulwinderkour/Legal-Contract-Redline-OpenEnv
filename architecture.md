# Architecture — Legal Contract Redline OpenEnv

This document provides an in-depth technical architecture reference for the Legal Contract Redline OpenEnv project. It covers system design, component responsibilities, data flow, type system, reward mechanics, and deployment topology.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Diagram](#component-diagram)
3. [Module Dependency Graph](#module-dependency-graph)
4. [Data Flow — Full Episode Lifecycle](#data-flow--full-episode-lifecycle)
5. [Type System — Pydantic Models](#type-system--pydantic-models)
6. [Environment Engine — `env.py`](#environment-engine--envpy)
7. [API Layer — `server.py`](#api-layer--serverpy)
8. [Inference Agent — `inference.py`](#inference-agent--inferencepy)
9. [Reward System — Detailed Mechanics](#reward-system--detailed-mechanics)
10. [Clause Dataset — Schema and Distribution](#clause-dataset--schema-and-distribution)
11. [OpenEnv Manifest — `openenv.yaml`](#openenv-manifest--openenvyaml)
12. [Deployment Architecture](#deployment-architecture)
13. [Error Handling Matrix](#error-handling-matrix)
14. [Sequence Diagrams](#sequence-diagrams)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL AGENT                               │
│                      (inference.py / any client)                    │
│                                                                     │
│  ┌──────────┐   HTTP POST /reset   ┌──────────────────────────┐    │
│  │          │ ───────────────────>  │                          │    │
│  │   LLM    │   HTTP POST /step    │     FastAPI Server       │    │
│  │  Agent   │ ───────────────────>  │      (server.py)         │    │
│  │          │   HTTP GET /state    │                          │    │
│  │          │ ───────────────────>  │  ┌──────────────────┐   │    │
│  │          │ <───────────────────  │  │  ContractRedline  │   │    │
│  └──────────┘   JSON responses     │  │      Env          │   │    │
│                                     │  │   (env.py)        │   │    │
│                                     │  └──────────────────┘   │    │
│                                     │           │              │    │
│                                     │  ┌────────▼─────────┐   │    │
│                                     │  │   Pydantic Models │   │    │
│                                     │  │   (models.py)     │   │    │
│                                     │  └──────────────────┘   │    │
│                                     └──────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

The system follows a **3-layer architecture**:

1. **Agent Layer** (`inference.py`) — Dual-mode client (rule-based default, optional LLM) that observes clauses and submits actions
2. **API Layer** (`server.py`) — FastAPI HTTP server that validates requests, routes to the environment, and tracks metrics
3. **Environment Layer** (`env.py`) — Core RL environment logic: clause selection, action grading, reward computation

All layers share a common **Type Layer** (`models.py`) that enforces data contracts via Pydantic.

---

## Component Diagram

```
┌──────────────────────────────────────────────────┐
│                   models.py                       │
│                                                   │
│  ┌──────────────┐  ┌────────────┐  ┌───────────┐│
│  │ RiskCategory │  │   Action   │  │Observation ││
│  │   (Enum)     │  │(BaseModel) │  │(BaseModel) ││
│  └──────────────┘  └────────────┘  └───────────┘│
│  ┌──────────────┐  ┌────────────┐  ┌───────────┐│
│  │ ResetRequest │  │ StepResult │  │ Metrics   ││
│  │(BaseModel)   │  │(BaseModel) │  │ Response  ││
│  └──────────────┘  └────────────┘  └───────────┘│
└────────────┬─────────────┬───────────────────────┘
             │             │
     ┌───────▼──────┐  ┌──▼────────────────┐
     │   env.py     │  │    server.py       │
     │              │  │                    │
     │ CLAUSES[]    │  │ GET  /             │
     │ TASK_INSTR{} │  │ GET  /health       │
     │ auto_label() │  │ GET  /tasks        │
     │              │  │ POST /reset        │
     │ Redline      │  │ POST /step         │
     │   Env        │──│ GET  /state        │
     │  .reset()    │  │ GET  /metrics      │
     │  .step()     │  │                    │
     │  .state()    │  │ _env (global)      │
     │  ._grade()   │  │ _history (global)  │
     │  ._score_*() │  │                    │
     └──────────────┘  └────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   inference.py       │
                    │                     │
                    │ call_env()          │
                    │ get_action()        │
                    │  ├─ _rule_based()   │
                    │  └─ _ask_llm()      │
                    │ run_task()          │
                    │                     │
                    │ OpenAI client ──────│──> HF Inference API (optional)
                    │ requests client ────│──> server.py
                    └─────────────────────┘
```

---

## Module Dependency Graph

```
models.py          (no dependencies — pure Pydantic definitions)
    │
    ├──> env.py    (imports Action, Observation, RiskCategory from models)
    │       │
    │       └──> server.py  (imports ContractRedlineEnv from env)
    │               │
    └───────────────┘  (imports Action, ResetRequest, StepResult, MetricsResponse from models)

inference.py       (independent — communicates with server.py over HTTP only)
    ├── openai     (LLM API client — imported only if HF_TOKEN is set)
    ├── requests   (HTTP client for env API)
    └── os/json    (environment variables, JSON parsing)
```

**Key design principle:** `inference.py` has **zero imports** from `env.py`, `server.py`, or `models.py`. It communicates exclusively via HTTP, making it a true external agent that can run against any compatible OpenEnv server.

---

## Data Flow — Full Episode Lifecycle

### Phase 1: Reset

```
Agent                          Server                         Environment
  │                              │                                │
  │  POST /reset                 │                                │
  │  {"task":"med"}              │                                │
  │  or                          │                                │
  │  {"task":"med",              │                                │
  │   "clause_text":"..."}       │                                │
  │ ─────────────────────────>   │                                │
  │                              │  ResetRequest validated        │
  │                              │  ContractRedlineEnv(task)      │
  │                              │ ────────────────────────────>  │
  │                              │                                │  if custom clause:
  │                              │                                │    auto_label_clause(text)
  │                              │                                │    clause_id = -1
  │                              │                                │  else:
  │                              │                                │    _clause_index++ (deterministic)
  │                              │                                │  step_count = 0
  │                              │                                │  done = False
  │                              │  <── Observation ──────────    │
  │  <── Observation JSON ────   │                                │
  │                              │                                │
```

### Phase 2: Step (repeated up to MAX_STEPS=3)

```
Agent                          Server                         Environment
  │                              │                                │
  │  clause_text ──> Agent ──>   │                                │
  │  (rule-based or LLM)         │                                │
  │                              │                                │
  │  POST /step {Action}         │                                │
  │ ─────────────────────────>   │                                │
  │                              │  Action validated (Pydantic)   │
  │                              │  _env.step(action)             │
  │                              │ ────────────────────────────>  │
  │                              │                                │  step_count++
  │                              │                                │  reward, breakdown = _grade(action)
  │                              │                                │  done = reward>=0.8 or steps>=3
  │                              │  <── (obs, reward, done, info) │
  │                              │                                │
  │                              │  _history.append(record)       │
  │                              │  StepResult constructed        │
  │  <── StepResult JSON ────   │                                │
  │                              │                                │
  │  if done: break              │                                │
  │  else: next step             │                                │
```

### Phase 3: Logging

```
Agent
  │
  │  [START] task=... env=... model=...     (after reset)
  │  [STEP]  step=N action=... reward=...   (after each step)
  │  [END]   success=... steps=... score=... rewards=...  (after loop)
  │
```

---

## Type System — Pydantic Models

### `RiskCategory` (str Enum)

```
RiskCategory
├── LIABILITY        = "liability"
├── TERMINATION      = "termination"
├── IP_OWNERSHIP     = "ip_ownership"
├── INDEMNIFICATION  = "indemnification"
├── GOVERNING_LAW    = "governing_law"
└── SAFE             = "safe"
```

Used by `Action.risk_category`. Any value not in this enum is rejected with HTTP 422.

### `Action` (BaseModel)

```
Action
├── is_risky: bool                          [REQUIRED]
├── risk_category: RiskCategory = SAFE      [enum-validated]
├── risky_phrase: str = ""                  [max_length=1000, auto-stripped]
└── rewrite: str = ""                       [max_length=5000, auto-stripped]
```

**Validators:**
- `strip_strings` — `field_validator` on `risky_phrase` and `rewrite` that calls `.strip()` before storage

### `Observation` (BaseModel)

```
Observation
├── clause_id: int
├── clause_text: str
├── task: str
├── step_number: int
└── instructions: str
```

Returned by `env.reset()` and embedded in `StepResult`.

### `StepResult` (BaseModel)

```
StepResult
├── observation: Observation
├── reward: float                           [ge=0.0, le=1.0]
├── done: bool
└── info: dict                              (contains "breakdown" sub-dict)
```

Returned by `POST /step`.

### `ResetRequest` (BaseModel)

```
ResetRequest
├── task: str = "easy"                      [validated: must be easy|medium|hard]
└── clause_text: str | None = None          [optional, max_length=5000, for custom input]
```

### `MetricsResponse` (BaseModel)

```
MetricsResponse
├── avg_reward: float
├── success_rate: float
└── task_breakdown: dict                    (task_name -> {avg_reward, success_rate, episodes})
```

---

## Environment Engine — `env.py`

### Class: `ContractRedlineEnv`

**Class-level state:**
- `MAX_STEPS = 3` — maximum steps per episode
- `_clause_index: int = 0` — shared across all instances, provides deterministic round-robin clause cycling

**Instance state:**

| Attribute | Type | Initial Value | Description |
|-----------|------|---------------|-------------|
| `task` | `str` | Constructor arg | Task difficulty level |
| `current_clause` | `dict \| None` | `None` | Active clause from `CLAUSES` |
| `step_count` | `int` | `0` | Steps taken this episode |
| `done` | `bool` | `False` | Whether episode has ended |
| `last_reward` | `float` | `0.0` | Most recent step reward |
| `episode_rewards` | `list[float]` | `[]` | All rewards this episode |

### Method: `reset(custom_clause_text=None) -> Observation`

1. If `custom_clause_text` is provided and non-empty:
   - Call `auto_label_clause(text)` to generate ground-truth labels via keyword detection
   - Set `clause_id = -1` (custom input marker)
2. Otherwise:
   - Select `CLAUSES[_clause_index % 10]`
   - Increment `_clause_index` (class-level)
3. Reset `step_count=0`, `done=False`, `last_reward=0.0`, `episode_rewards=[]`
4. Return `Observation` with clause data and task instructions

### Function: `auto_label_clause(text) -> dict`

Generates ground-truth labels for custom clause input using keyword matching:
- Scans for 17 risk phrases across 5 categories (termination, liability, indemnification, ip_ownership, governing_law)
- Returns first match with `is_risky=True`, detected category, and extracted phrase
- If no match found, returns `is_risky=False, risk_category="safe"`
- Always sets `clause_id = -1` and `safe_rewrite = ""`

### Method: `step(action: Action) -> tuple`

1. Guard: raise `RuntimeError` if `done=True` or `current_clause is None`
2. Increment `step_count`
3. Compute `reward, breakdown = _grade(action)`
4. Set `done = reward >= 0.8 or step_count >= MAX_STEPS`
5. Append reward to `episode_rewards`
6. Return `(Observation, reward, done, {"breakdown": breakdown})`

### Method: `_grade(action: Action) -> tuple[float, dict]`

Dispatches to task-specific scoring. See [Reward System](#reward-system--detailed-mechanics) for full details.

### Method: `_score_phrase(predicted, expected) -> float`

Token-overlap phrase matching. Returns score in `[0.0, 1.0]`.

### Method: `_score_rewrite(rewrite, clause) -> float`

Multi-criteria rewrite quality assessment. Returns score in `[0.0, 0.40]`.

### Method: `state() -> dict`

Returns snapshot: `{task, step_count, done, current_clause_id, episode_rewards, last_reward}`.

---

## API Layer — `server.py`

### Global State

| Variable | Type | Description |
|----------|------|-------------|
| `_env` | `ContractRedlineEnv \| None` | Current environment instance. `None` before first `/reset`. Replaced on each `/reset`. |
| `_history` | `list[dict]` | Append-only log of `{task, reward, done, success}` for every `/step` call. Used by `/metrics`. |

### Endpoint Details

#### `GET /` — Root

Returns static service metadata. No state interaction.

```python
{
    "name": "legal-contract-redline",
    "version": "1.0.0",
    "tasks": ["easy", "medium", "hard"],
    "endpoints": ["/reset", "/step", "/state", "/health", "/tasks", "/metrics"]
}
```

#### `GET /health` — Health Check

Returns `{"status": "ok"}`. Used by orchestrators, load balancers, and HF Spaces to verify liveness.

#### `GET /tasks` — Task Catalog

Returns array of 3 task objects with `name`, `description`, `difficulty`. Static data.

#### `POST /reset` — Episode Initialization

1. Validate `ResetRequest` (Pydantic rejects invalid `task`; optional `clause_text` accepted)
2. Instantiate new `ContractRedlineEnv(task=req.task)`
3. Call `env.reset(custom_clause_text=req.clause_text)` → `Observation`
4. Return observation as JSON dict

**Custom clause support:** If `clause_text` is provided, the environment auto-labels it using keyword detection and sets `clause_id = -1`. This allows judges to paste any contract text and get graded feedback.

**Important:** Each `/reset` creates a **new** environment instance, discarding any previous episode.

#### `POST /step` — Action Submission

1. Guard: 400 if `_env is None`
2. Guard: 400 if `_env.done is True`
3. Validate `Action` (Pydantic rejects invalid `risk_category` enum)
4. Call `env.step(action)` → `(obs, reward, done, info)`
5. Wrap in `StepResult` (enforces `reward` in `[0,1]`)
6. Append to `_history`
7. Return `StepResult` as JSON dict

#### `GET /state` — Episode Snapshot

Returns `env.state()` dict if `_env` exists, else `{"status": "no active episode"}`.

#### `GET /metrics` — Aggregated Performance

Computes from `_history`:
- **`avg_reward`** — mean of all step rewards
- **`success_rate`** — fraction of completed episodes with reward >= 0.8
- **`task_breakdown`** — per-task `{avg_reward, success_rate, episodes}` for each of easy/medium/hard

---

## Inference Agent — `inference.py`

### Architecture

```
inference.py
├── Configuration (env vars)
│   ├── HF_TOKEN (optional — enables LLM mode)
│   ├── API_BASE_URL
│   ├── MODEL_NAME
│   └── ENV_URL
│
├── Mode Detection
│   ├── USE_LLM = True if HF_TOKEN is set
│   └── AGENT_LABEL = MODEL_NAME if LLM, else "baseline"
│
├── Rule-Based Agent (default)
│   └── _rule_based_agent(clause_text)
│       ├── Scan 14 risk phrases across 5 categories
│       ├── First match → is_risky=True + category + extracted phrase
│       ├── No match → is_risky=False, category="safe"
│       └── Generate rewrite by removing risky phrase
│
├── LLM Agent (optional, only if HF_TOKEN set)
│   └── _ask_llm(clause_text, instructions)
│       ├── OpenAI client (temperature=0.0, max_tokens=512)
│       ├── Strip markdown fences, parse JSON
│       └── Raise on invalid response
│
├── get_action(clause_text, instructions)
│   ├── If USE_LLM: try _ask_llm, fallback to _rule_based_agent on error
│   └── Else: _rule_based_agent directly
│
├── call_env(method, path, body)
│   └── requests.get/post → JSON
│
└── run_task(task_name)
    ├── POST /reset → obs
    ├── [START] log
    ├── Loop (max 3 steps):
    │   ├── get_action(clause_text, instructions)
    │   ├── POST /step → result
    │   ├── [STEP] log
    │   └── break if done
    └── [END] log
```

### Dual-Mode Design

- **Mode 1 (default): Rule-based agent** — works with zero dependencies, no API key needed
- **Mode 2 (optional): LLM agent** — activated only when `HF_TOKEN` is set; falls back to rule-based on any error

The rule-based agent matches 14 risk phrases across 5 categories (termination, liability, indemnification, ip_ownership, governing_law). It extracts the matched phrase and generates a rewrite by removing the risky language.

### LLM System Prompt Strategy

The system prompt (used only in LLM mode) enforces:
1. **JSON-only output** — no explanations, no markdown
2. **Exact field names** — `is_risky`, `risk_category`, `risky_phrase`, `rewrite`
3. **Enum values** — lists the 6 valid `risk_category` values
4. **Example response** — shows expected JSON structure

### Error Recovery

| Failure Mode | Recovery |
|-------------|----------|
| `HF_TOKEN` not set | Use rule-based agent (no crash, no error) |
| LLM returns non-JSON | Fallback to rule-based agent, log error |
| LLM returns markdown-wrapped JSON | Strip ` ```json ` and ` ``` ` before parsing |
| LLM returns invalid `is_risky` | Fallback to rule-based agent, log error |
| LLM API timeout/error | Fallback to rule-based agent, log error string |
| `openai` package not installed | `USE_LLM` set to False, use rule-based |
| Environment API error | Catch exception, emit `[STEP]` with `reward=0.00`, proceed to `[END]` |

**The script never crashes.** Every code path produces valid structured log output.

---

## Reward System — Detailed Mechanics

### Reward Computation Pipeline

```
Action ──> _grade(action)
               │
               ├── Extract risk_category value (handle Enum or str)
               │
               ├── [EASY]
               │   └── is_risky match? → 1.0 or 0.0
               │
               ├── [MEDIUM]
               │   ├── is_risky match?       → +0.3
               │   ├── category match?        → +0.4
               │   └── phrase_score × 0.3     → +0.0 to +0.3
               │
               └── [HARD]
                   ├── is_risky match?       → +0.15
                   ├── category match?        → +0.25
                   ├── phrase_score × 0.20   → +0.0 to +0.20
                   └── rewrite_score          → +0.0 to +0.40
               │
               └── min(round(sum, 4), 1.0)
```

### Phrase Scoring — `_score_phrase(predicted, expected)`

**Input:** Two strings (predicted phrase from agent, expected phrase from dataset).

**Algorithm:**

```
if expected is empty:
    return 1.0 if predicted is empty else 0.5

pred = predicted.strip().lower()
exp = expected.strip().lower()

if pred is empty:           return 0.0
if pred == exp:             return 1.0    # exact match
if exp in pred or pred in exp: return 0.9 # substring containment

pred_tokens = set(pred.split())
exp_tokens = set(exp.split())
overlap = pred_tokens & exp_tokens
ratio = len(overlap) / len(exp_tokens)

if ratio >= 0.75:          return 0.7
if ratio >= 0.50:          return 0.5
if len(overlap) >= 2:      return 0.3
else:                       return 0.0
```

**Design rationale:** Token overlap ratio rewards partial matches proportionally. The threshold tiers (0.75, 0.50, 2+ tokens) provide clear grading levels without being overly strict.

### Rewrite Scoring — `_score_rewrite(rewrite, clause)`

**Input:** Rewrite string and full clause dict (with `text` and `risky_phrase`).

**Algorithm:**

```
if empty:                                               return 0.00
word_count = len(rewrite.split())
differs = (rewrite.lower() != original.lower())
risky_absent = (risky_phrase not in rewrite) if risky_phrase else True

if word_count < 5:                                      return 0.05
if word_count > 20 AND risky_absent AND differs:        return 0.40
if word_count > 10 AND risky_absent AND differs:        return 0.30
if word_count > 10 AND differs:                         return 0.20
if differs:                                             return 0.10
else:                                                   return 0.05
```

**Design rationale:** The 5-tier system rewards:
- **Thoroughness** (>20 words) — indicates complete rewrite
- **Risky phrase removal** — primary goal of redlining
- **Semantic difference** — ensures agent didn't just copy the original
- **Minimum effort** — even a bad attempt gets some credit

### Reward Properties

- **Deterministic:** Same input always produces same reward
- **Bounded:** All rewards clamped to `[0.0, 1.0]`
- **Decomposable:** Every reward includes a `breakdown` dict showing per-component scores
- **Partial credit:** Agents earn proportional credit for partially correct answers

---

## Clause Dataset — Schema and Distribution

### Schema

Each clause in `CLAUSES` is a dict with:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` | Unique identifier (1-10) |
| `text` | `str` | Full contract clause text |
| `is_risky` | `bool` | Ground truth: risky or safe |
| `risk_category` | `str` | One of the 6 RiskCategory values |
| `risky_phrase` | `str` | Exact risky substring, or `""` |
| `safe_rewrite` | `str` | Reference rewrite, or `""` |

### Distribution

| Category | Count | Clause IDs |
|----------|-------|------------|
| `termination` | 2 | 1, 8 |
| `liability` | 1 | 2 |
| `ip_ownership` | 2 | 4, 10 |
| `indemnification` | 1 | 6 |
| `governing_law` | 1 | 7 |
| `safe` | 3 | 3, 5, 9 |
| **Total** | **10** | |

**Risk ratio:** 70% risky, 30% safe.

### Deterministic Selection

Clauses are served via round-robin using `ContractRedlineEnv._clause_index`:
- Episode 1 → Clause 1
- Episode 2 → Clause 2
- ...
- Episode 10 → Clause 10
- Episode 11 → Clause 1 (wraps around)

The `_clause_index` is a **class-level** variable, shared across all instances. It increments on every `reset()` call regardless of which task is being used.

---

## OpenEnv Manifest — `openenv.yaml`

### Structure

```yaml
name: legal-contract-redline
version: "1.0.0"
description: "..."

tasks:                          # 3 tasks with difficulty, scoring formulas
action_schema:                  # JSON Schema-style with types, enums, defaults
observation_schema:             # JSON Schema-style with types
reward:                         # range, partial_credit, deterministic flag
endpoints:                      # 5 API routes
environment:
  variables:                    # HF_TOKEN, API_BASE_URL, MODEL_NAME
tags:                           # [legal, nlp, contract-review, openenv]
```

### Compliance Checklist

| Requirement | Status |
|-------------|--------|
| `name` field | Present |
| `version` field | Present ("1.0.0") |
| `description` field | Present (detailed) |
| `tasks` with difficulty and reward_range | Present (3 tasks) |
| `action_schema` with typed properties | Present (4 fields, enum, defaults) |
| `observation_schema` with typed properties | Present (5 fields) |
| `reward` with range and description | Present ([0,1], partial credit) |
| `endpoints` for reset/step/state/health | Present (5 endpoints) |
| `environment.variables` | Present (3 variables) |
| `tags` | Present (4 tags) |

---

## Deployment Architecture

### Local Development

```
┌─────────────┐    HTTP     ┌────────────────┐
│ inference.py │ ─────────> │   server.py    │
│ (agent)      │ <───────── │   port 7860    │
└─────────────┘             └────────────────┘
       │
       │ HTTPS
       ▼
┌─────────────────────┐
│  HF Inference API   │
│  (LLM endpoint)     │
└─────────────────────┘
```

### Docker Deployment

```
┌──────────────────────────────────────┐
│  Docker Container                     │
│  python:3.11-slim                     │
│  User: appuser (UID 1000)            │
│                                       │
│  ┌────────────────────────────────┐  │
│  │  uvicorn → FastAPI (server.py) │  │
│  │  Port 7860                     │  │
│  └────────────────────────────────┘  │
│                                       │
│  EXPOSE 7860                         │
└──────────────────────────────────────┘
```

### Hugging Face Spaces

```
┌─────────────────────────────────────────────────┐
│  Hugging Face Space (Docker SDK)                 │
│                                                   │
│  README.md frontmatter:                          │
│    sdk: docker                                    │
│    title: Legal Contract Redline OpenEnv          │
│                                                   │
│  Secrets:                                         │
│    HF_TOKEN = <user token>                       │
│                                                   │
│  Resources:                                       │
│    2 vCPU, 8GB RAM                               │
│                                                   │
│  ┌──────────────────────────────────────────┐    │
│  │  Docker Container (see above)             │    │
│  │  Port 7860 → public URL                   │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

---

## Error Handling Matrix

| Layer | Error Condition | HTTP Code | Response |
|-------|----------------|-----------|----------|
| **Server** | Invalid `task` in `/reset` | 422 | Pydantic validation error array |
| **Server** | Invalid `risk_category` in `/step` | 422 | Pydantic enum validation error |
| **Server** | `/step` before `/reset` | 400 | `"Call /reset first"` |
| **Server** | `/step` after episode done | 400 | `"Episode is done. Call /reset to start a new episode."` |
| **Env** | `step()` called when `done=True` | RuntimeError | Caught by server → 400 |
| **Env** | `step()` called when `current_clause=None` | RuntimeError | Caught by server → 400 |
| **Env** | Invalid task in constructor | ValueError | Caught by server → 422 |
| **Inference** | Missing `HF_TOKEN` | — | Uses rule-based agent (no error) |
| **Inference** | LLM returns invalid JSON | — | Fallback to rule-based agent + error in `[STEP]` log |
| **Inference** | LLM API error | — | Fallback to rule-based agent + error in `[STEP]` log |
| **Inference** | Environment API error | — | Emit `[STEP] reward=0.00` + `[END]` |

---

## Sequence Diagrams

### Successful Easy Task Episode

```
Agent              Server             Env
  │                  │                  │
  │ POST /reset      │                  │
  │ {"task":"easy"}  │                  │
  │ or {"task":"easy",│                  │
  │  "clause_text":..}│                  │
  │ ───────────────> │                  │
  │                  │ new Env("easy")  │
  │                  │ env.reset(text)  │
  │                  │ ───────────────> │
  │                  │                  │ if custom: auto_label(text)
  │                  │                  │ else: clause = CLAUSES[idx++]
  │                  │ <─ Observation ─ │
  │ <─ 200 JSON ──── │                  │
  │                  │                  │
  │ [START] log      │                  │
  │                  │                  │
  │ get_action() ──> (rule-based       │
  │                   or LLM → HF API) │
  │ <── action dict  │                  │
  │                  │                  │
  │ POST /step       │                  │
  │ {"is_risky":T..} │                  │
  │ ───────────────> │                  │
  │                  │ validate Action  │
  │                  │ env.step(action) │
  │                  │ ───────────────> │
  │                  │                  │ _grade() → 1.0
  │                  │                  │ done = True
  │                  │ <─ StepResult ── │
  │                  │ _history.append  │
  │ <─ 200 JSON ──── │                  │
  │                  │                  │
  │ [STEP] log       │                  │
  │ [END] log        │                  │
```

### Failed Validation

```
Agent              Server
  │                  │
  │ POST /step       │
  │ {"is_risky":true,│
  │  "risk_category":│
  │  "invalid_cat"}  │
  │ ───────────────> │
  │                  │ Pydantic rejects "invalid_cat"
  │ <─ 422 JSON ──── │ (not in RiskCategory enum)
  │                  │
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Server startup time | <2 seconds |
| `/reset` latency | <1ms (no I/O, no DB) |
| `/step` latency | <1ms (pure computation) |
| Memory footprint | <50MB (10 clauses in-memory) |
| Inference per task (rule-based) | <1s |
| Inference per task (LLM) | ~5-15s (API-dependent) |
| Full benchmark (3 tasks, rule-based) | <3s |
| Max concurrent episodes | 1 (single global `_env`) |
| Docker image size | ~150MB (python:3.11-slim + deps) |
