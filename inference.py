import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Optional — if you use from_docker_image():
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

BENCHMARK = "legal-redline"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 3

# ---------------------------------------------------------------------------
# Mode detection: LLM if HF_TOKEN is set, otherwise deterministic rule-based
# ---------------------------------------------------------------------------
USE_LLM = bool(HF_TOKEN)
_llm_client = None

if USE_LLM:
    try:
        _llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception:
        USE_LLM = False
        _llm_client = None

AGENT_LABEL = MODEL_NAME if USE_LLM else "baseline"

SYSTEM_PROMPT = """You are a legal contract reviewer. You will be given a contract clause and a task.
You must respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.

The JSON must have these fields:
  "is_risky": boolean (true if the clause contains risk, false if safe)
  "risk_category": one of: "liability", "termination", "ip_ownership", "indemnification", "governing_law", "safe"
  "risky_phrase": the exact risky phrase from the clause text, or "" if safe
  "rewrite": a safe rewritten version of the clause, or "" if not needed

Example response:
{"is_risky": true, "risk_category": "liability", "risky_phrase": "under any circumstances", "rewrite": "The vendor liability shall not exceed total fees paid in the prior 12 months."}"""

# ---------------------------------------------------------------------------
# Rule-based agent (deterministic, zero-dependency)
# ---------------------------------------------------------------------------
_RISK_PHRASES = [
    ("without notice", "termination"),
    ("without cause", "termination"),
    ("at no additional cost", "termination"),
    ("under any circumstances", "liability"),
    ("not be liable for any damages", "liability"),
    ("including those arising from the client's own negligence", "indemnification"),
    ("client's own negligence", "indemnification"),
    ("indemnify", "indemnification"),
    ("whether related to the project or not", "ip_ownership"),
    ("sole property of the client", "ip_ownership"),
    ("without the contractor's consent", "ip_ownership"),
    ("assign this agreement to any third party", "ip_ownership"),
    ("binding arbitration in wilmington, delaware", "governing_law"),
    ("binding arbitration", "governing_law"),
]


def _rule_based_agent(clause_text: str) -> dict:
    text_lower = clause_text.lower()
    matched_category = "safe"
    risky_phrase = ""

    for phrase, category in _RISK_PHRASES:
        if phrase in text_lower:
            matched_category = category
            start = text_lower.find(phrase)
            risky_phrase = clause_text[start:start + len(phrase)]
            break

    is_risky = matched_category != "safe"

    rewrite = ""
    if is_risky and risky_phrase:
        rewrite = clause_text.replace(risky_phrase, "").strip()
        if rewrite.endswith(","):
            rewrite = rewrite[:-1].strip()
        while rewrite.endswith(",") or rewrite.endswith(" ,"):
            rewrite = rewrite.rstrip(", ").strip()
        if len(rewrite.split()) < 6:
            rewrite = f"This clause should be revised to remove the risky language regarding {matched_category}."

    return {
        "is_risky": is_risky,
        "risk_category": matched_category,
        "risky_phrase": risky_phrase if is_risky else "",
        "rewrite": rewrite if is_risky else "",
    }


# ---------------------------------------------------------------------------
# LLM agent (optional, only when HF_TOKEN is available)
# ---------------------------------------------------------------------------
def _ask_llm(clause_text: str, instructions: str) -> dict:
    if _llm_client is None:
        raise RuntimeError("LLM client not available")
    user_msg = f"Instructions: {instructions}\n\nClause to review:\n{clause_text}"
    response = _llm_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=512,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Unified agent dispatcher with automatic fallback
# ---------------------------------------------------------------------------
def get_action(clause_text: str, instructions: str) -> tuple:
    """Returns (action_dict, error_string). error_string is 'null' on success."""
    if USE_LLM:
        try:
            action = _ask_llm(clause_text, instructions)
            if not isinstance(action.get("is_risky"), bool):
                raise ValueError("LLM returned invalid is_risky")
            return action, "null"
        except Exception as e:
            err = str(e).replace("\n", " ")[:120]
            return _rule_based_agent(clause_text), err
    return _rule_based_agent(clause_text), "null"


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------
def call_env(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{ENV_URL}{path}"
    if method == "GET":
        r = requests.get(url, timeout=30)
    else:
        r = requests.post(url, json=body or {}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Task runner with strict log format
# ---------------------------------------------------------------------------
def run_task(task_name: str) -> None:
    rewards: list[float] = []
    final_score = 0.0
    success = False
    total_steps = 0
    error_str = "null"
    started = False

    try:
        obs = call_env("POST", "/reset", {"task": task_name})
        print(f"[START] task={task_name} env={BENCHMARK} model={AGENT_LABEL}", flush=True)
        started = True

        for step_n in range(1, MAX_STEPS + 1):
            clause_text = obs.get("clause_text", "")
            instructions = obs.get("instructions", "")

            action, error_str = get_action(clause_text, instructions)

            action_payload = {
                "is_risky": bool(action.get("is_risky", False)),
                "risk_category": str(action.get("risk_category", "safe")),
                "risky_phrase": str(action.get("risky_phrase", "")),
                "rewrite": str(action.get("rewrite", "")),
            }

            action_str = f"is_risky={action_payload['is_risky']},category={action_payload['risk_category']}"

            result = call_env("POST", "/step", action_payload)

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            rewards.append(reward)
            total_steps = step_n

            done_str = "true" if done else "false"
            print(f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={done_str} error={error_str}", flush=True)

            if done:
                break

            obs = result.get("observation", obs)

    except Exception as e:
        error_str = str(e).replace("\n", " ")[:120]
        if not started:
            print(f"[START] task={task_name} env={BENCHMARK} model={AGENT_LABEL}", flush=True)
        if not rewards:
            rewards = [0.0]
            total_steps = 1
            print(f"[STEP] step=1 action=none reward=0.00 done=true error={error_str}", flush=True)

    if rewards:
        final_score = max(rewards)
    success = final_score >= 0.8

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(f"[END] success={success_str} steps={total_steps} score={final_score:.2f} rewards={rewards_str}", flush=True)


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
