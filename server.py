from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from env import ContractRedlineEnv, ContractAction

app = FastAPI(title="Legal Contract Redline OpenEnv", version="1.0.0")

_env: ContractRedlineEnv = None


class ResetRequest(BaseModel):
    task: str = "easy"


class StepRequest(BaseModel):
    is_risky: bool
    risk_category: str = "safe"
    risky_phrase: str = ""
    rewrite: str = ""


@app.get("/")
def root():
    return {
        "name": "legal-contract-redline",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/health", "/tasks"],
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
def reset(req: ResetRequest):
    global _env
    if req.task not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=400, detail="task must be easy, medium, or hard")
    _env = ContractRedlineEnv(task=req.task)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(req: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    action = ContractAction(
        is_risky=req.is_risky,
        risk_category=req.risk_category,
        risky_phrase=req.risky_phrase,
        rewrite=req.rewrite,
    )
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": round(reward, 4),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    global _env
    if _env is None:
        return {"status": "no active episode"}
    return _env.state()


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
