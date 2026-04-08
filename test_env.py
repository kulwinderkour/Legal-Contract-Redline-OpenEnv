"""Self-test script for Legal Contract Redline OpenEnv.

Run while the server is up:
    python test_env.py

Validates /health, /reset, /step, /metrics, and error handling.
Prints ALL TESTS PASSED on success, exits with code 1 on failure.
"""
import os
import sys
import requests

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def get(path: str) -> requests.Response:
    return requests.get(f"{ENV_URL}{path}", timeout=10)


def post(path: str, body: dict | None = None) -> requests.Response:
    return requests.post(f"{ENV_URL}{path}", json=body or {}, timeout=10)


# ── 1. Health ─────────────────────────────────────────────────────────────
print("\n=== /health ===")
r = get("/health")
check("status 200", r.status_code == 200)
check("body has status=ok", r.json().get("status") == "ok")

# ── 2. Step before reset → 400 ───────────────────────────────────────────
print("\n=== /step before /reset ===")
r = post("/step", {"is_risky": False})
check("status 400", r.status_code == 400, f"got {r.status_code}")

# ── 3. Reset easy ─────────────────────────────────────────────────────────
print("\n=== /reset (easy) ===")
r = post("/reset", {"task": "easy"})
check("status 200", r.status_code == 200)
obs = r.json()
check("has clause_text", "clause_text" in obs)
check("has clause_id", "clause_id" in obs)
check("has instructions", "instructions" in obs)
check("task is easy", obs.get("task") == "easy")
check("step_number is 0", obs.get("step_number") == 0)

# ── 4. Step easy ──────────────────────────────────────────────────────────
print("\n=== /step (easy) ===")
r = post("/step", {
    "is_risky": True,
    "risk_category": "termination",
    "risky_phrase": "without notice",
    "rewrite": "",
})
check("status 200", r.status_code == 200)
result = r.json()
reward = result.get("reward", -1)
check("reward is float", isinstance(reward, (int, float)))
check("reward in [0,1]", 0.0 <= reward <= 1.0, f"got {reward}")
check("done is bool", isinstance(result.get("done"), bool))
check("has observation", "observation" in result)
check("has info", "info" in result)

# ── 5. Step after done → 400 ─────────────────────────────────────────────
print("\n=== /step after done ===")
if result.get("done"):
    r2 = post("/step", {"is_risky": False})
    check("status 400", r2.status_code == 400, f"got {r2.status_code}")
else:
    print("  [SKIP] episode not done yet")

# ── 6. Reset invalid task → 422 ──────────────────────────────────────────
print("\n=== /reset invalid task ===")
r = post("/reset", {"task": "impossible"})
check("status 422", r.status_code == 422, f"got {r.status_code}")

# ── 7. Step with invalid category → 422 ──────────────────────────────────
print("\n=== /step invalid category ===")
post("/reset", {"task": "easy"})
r = post("/step", {"is_risky": True, "risk_category": "INVALID_CAT"})
check("status 422", r.status_code == 422, f"got {r.status_code}")

# ── 8. Medium task ────────────────────────────────────────────────────────
print("\n=== /reset + /step (medium) ===")
r = post("/reset", {"task": "medium"})
check("reset medium 200", r.status_code == 200)
obs = r.json()
r = post("/step", {
    "is_risky": True,
    "risk_category": "liability",
    "risky_phrase": "under any circumstances",
    "rewrite": "",
})
check("step medium 200", r.status_code == 200)
med_reward = r.json().get("reward", -1)
check("medium reward in [0,1]", 0.0 <= med_reward <= 1.0, f"got {med_reward}")

# ── 9. Hard task ──────────────────────────────────────────────────────────
print("\n=== /reset + /step (hard) ===")
r = post("/reset", {"task": "hard"})
check("reset hard 200", r.status_code == 200)
obs = r.json()
r = post("/step", {
    "is_risky": False,
    "risk_category": "safe",
    "risky_phrase": "",
    "rewrite": "",
})
check("step hard 200", r.status_code == 200)
hard_reward = r.json().get("reward", -1)
check("hard reward in [0,1]", 0.0 <= hard_reward <= 1.0, f"got {hard_reward}")

# ── 10. Custom clause (risky) ─────────────────────────────────────────────
print("\n=== /reset + /step (custom risky clause) ===")
r = post("/reset", {
    "task": "easy",
    "clause_text": "Vendor may terminate without notice.",
})
check("custom reset 200", r.status_code == 200)
obs = r.json()
check("custom clause_id is -1", obs.get("clause_id") == -1)
check("custom clause_text returned", obs.get("clause_text") == "Vendor may terminate without notice.")
check("custom task is easy", obs.get("task") == "easy")

r = post("/step", {"is_risky": True, "risk_category": "termination"})
check("custom step 200", r.status_code == 200)
custom_reward = r.json().get("reward", -1)
check("custom reward in [0,1]", 0.0 <= custom_reward <= 1.0, f"got {custom_reward}")
check("custom reward is 1.0 (correct)", custom_reward == 1.0, f"got {custom_reward}")

# ── 11. Custom clause (safe) ─────────────────────────────────────────────
print("\n=== /reset + /step (custom safe clause) ===")
r = post("/reset", {
    "task": "easy",
    "clause_text": "Payment is due within 30 days of receipt of invoice.",
})
check("safe custom reset 200", r.status_code == 200)
obs = r.json()
check("safe custom clause_id is -1", obs.get("clause_id") == -1)

r = post("/step", {"is_risky": False, "risk_category": "safe"})
check("safe custom step 200", r.status_code == 200)
safe_reward = r.json().get("reward", -1)
check("safe custom reward is 1.0", safe_reward == 1.0, f"got {safe_reward}")

# ── 12. Custom clause (medium) ───────────────────────────────────────────
print("\n=== /reset + /step (custom medium clause) ===")
r = post("/reset", {
    "task": "medium",
    "clause_text": "The vendor shall not be liable for any damages under any circumstances.",
})
check("custom medium reset 200", r.status_code == 200)
r = post("/step", {
    "is_risky": True,
    "risk_category": "liability",
    "risky_phrase": "under any circumstances",
})
check("custom medium step 200", r.status_code == 200)
cm_reward = r.json().get("reward", -1)
check("custom medium reward in [0,1]", 0.0 <= cm_reward <= 1.0, f"got {cm_reward}")
check("custom medium reward >= 0.8", cm_reward >= 0.8, f"got {cm_reward}")

# ── 13. Missing fields → 422 ─────────────────────────────────────────────
print("\n=== /step missing is_risky → 422 ===")
post("/reset", {"task": "easy"})
r = post("/step", {"risk_category": "safe"})
check("missing is_risky → 422", r.status_code == 422, f"got {r.status_code}")

# ── 14. Metrics ───────────────────────────────────────────────────────────
print("\n=== /metrics ===")
r = get("/metrics")
check("metrics 200", r.status_code == 200)
m = r.json()
check("has avg_reward", "avg_reward" in m)
check("has success_rate", "success_rate" in m)
check("has task_breakdown", "task_breakdown" in m)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("ALL TESTS PASSED")
    sys.exit(0)
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
