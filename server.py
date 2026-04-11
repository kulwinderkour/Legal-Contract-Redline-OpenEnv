from fastapi import FastAPI, HTTPException
import uvicorn

from env import ContractRedlineEnv
from models import Action, ResetRequest, StepResult, MetricsResponse

app = FastAPI(title="Legal Contract Redline OpenEnv", version="1.0.0")

_env: ContractRedlineEnv | None = None
_history: list[dict] = []


@app.get("/")
def root():
    return {
        "name": "legal-contract-redline",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks", "/metrics"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Binary classification: is clause risky or safe?",
                "difficulty": 1,
            },
            {
                "name": "medium",
                "description": "Classify risk + identify specific risky phrase",
                "difficulty": 2,
            },
            {
                "name": "hard",
                "description": "Full redline: classify + extract phrase + rewrite clause",
                "difficulty": 3,
            },
        ]
    }


@app.post("/reset")
def reset(req: ResetRequest | None = None):
    global _env
    if req is None:
        req = ResetRequest()
    try:
        _env = ContractRedlineEnv(task=req.task)
        obs = _env.reset(custom_clause_text=req.clause_text)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
def step(req: Action):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    if _env.done:
        raise HTTPException(status_code=400, detail="Episode is done. Call /reset to start a new episode.")

    try:
        obs, reward, done, info = _env.step(req)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

    reward = max(0.0, min(round(reward, 4), 1.0))

    result = StepResult(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )

    _history.append({
        "task": _env.task,
        "reward": result.reward,
        "done": done,
        "success": result.reward >= 0.8,
    })

    return result.model_dump()


@app.get("/state")
def state():
    if _env is None:
        return {"status": "no active episode"}
    return _env.state()


@app.get("/metrics")
def metrics():
    try:
        if not _history:
            return MetricsResponse(
                avg_reward=0.0,
                success_rate=0.0,
                task_breakdown={},
            ).model_dump()

        completed = [h for h in _history if h["done"]]
        all_rewards = [h["reward"] for h in _history]
        avg_reward = round(sum(all_rewards) / len(all_rewards), 4) if all_rewards else 0.0
        success_count = sum(1 for h in completed if h["success"])
        success_rate = round(success_count / len(completed), 4) if completed else 0.0

        task_breakdown: dict[str, dict] = {}
        for task_name in ("easy", "medium", "hard"):
            task_entries = [h for h in _history if h["task"] == task_name]
            task_done = [h for h in task_entries if h["done"]]
            if task_entries:
                t_rewards = [h["reward"] for h in task_entries]
                t_success = sum(1 for h in task_done if h["success"])
                task_breakdown[task_name] = {
                    "avg_reward": round(sum(t_rewards) / len(t_rewards), 4),
                    "success_rate": round(t_success / len(task_done), 4) if task_done else 0.0,
                    "episodes": len(task_done),
                }

        return MetricsResponse(
            avg_reward=avg_reward,
            success_rate=success_rate,
            task_breakdown=task_breakdown,
        ).model_dump()
    except Exception:
        return MetricsResponse(
            avg_reward=0.0,
            success_rate=0.0,
            task_breakdown={},
        ).model_dump()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
