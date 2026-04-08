import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are a legal contract reviewer. You will be given a contract clause and a task.
You must respond ONLY with a valid JSON object. No explanation, no markdown, no extra text.

The JSON must have these fields:
  "is_risky": boolean (true if the clause contains risk, false if safe)
  "risk_category": one of: "liability", "termination", "ip_ownership", "indemnification", "governing_law", "safe"
  "risky_phrase": the exact risky phrase from the clause text, or "" if safe
  "rewrite": a safe rewritten version of the clause, or "" if not needed

Example response:
{"is_risky": true, "risk_category": "liability", "risky_phrase": "under any circumstances", "rewrite": "The vendor liability shall not exceed total fees paid in the prior 12 months."}"""

BENCHMARK = "legal-redline"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 3


def call_env(method, path, body=None):
    url = f"{ENV_URL}{path}"
    if method == "GET":
        r = requests.get(url, timeout=30)
    else:
        r = requests.post(url, json=body or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def ask_llm(clause_text, instructions):
    user_msg = f"Instructions: {instructions}\n\nClause to review:\n{clause_text}"
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"is_risky": False, "risk_category": "safe", "risky_phrase": "", "rewrite": ""}


def run_task(task_name):
    rewards = []
    final_score = 0.0
    success = False
    total_steps = 0
    error_str = "null"

    try:
        obs = call_env("POST", "/reset", {"task": task_name})
        print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

        for step_n in range(1, MAX_STEPS + 1):
            clause_text = obs.get("clause_text", "")
            instructions = obs.get("instructions", "")

            try:
                action = ask_llm(clause_text, instructions)
                error_str = "null"
            except Exception as e:
                action = {"is_risky": False, "risk_category": "safe", "risky_phrase": "", "rewrite": ""}
                error_str = str(e).replace("\n", " ")[:120]

            action_str = f"is_risky={action.get('is_risky')},category={action.get('risk_category')}"

            result = call_env("POST", "/step", {
                "is_risky": action.get("is_risky", False),
                "risk_category": action.get("risk_category", "safe"),
                "risky_phrase": action.get("risky_phrase", ""),
                "rewrite": action.get("rewrite", ""),
            })

            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            rewards.append(reward)
            total_steps = step_n

            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_n} action={action_str} reward={reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

            if done:
                final_score = reward
                success = reward >= 0.8
                break

            obs = result.get("observation", obs)

        if rewards:
            final_score = max(rewards)
            success = final_score >= 0.8

    except Exception as e:
        error_str = str(e).replace("\n", " ")[:120]
        if not rewards:
            rewards = [0.0]
            total_steps = 1
            print(
                f"[STEP] step=1 action=none reward=0.00 done=true error={error_str}",
                flush=True,
            )

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} steps={total_steps} score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )


if __name__ == "__main__":
    for task in TASKS:
        run_task(task)
