from enum import Enum
from pydantic import BaseModel, Field, field_validator


class RiskCategory(str, Enum):
    LIABILITY = "liability"
    TERMINATION = "termination"
    IP_OWNERSHIP = "ip_ownership"
    INDEMNIFICATION = "indemnification"
    GOVERNING_LAW = "governing_law"
    SAFE = "safe"


class Action(BaseModel):
    is_risky: bool
    risk_category: RiskCategory = RiskCategory.SAFE
    risky_phrase: str = Field(default="", max_length=1000)
    rewrite: str = Field(default="", max_length=5000)

    @field_validator("risky_phrase", "rewrite", mode="before")
    @classmethod
    def strip_strings(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v


class Observation(BaseModel):
    clause_id: int
    clause_text: str
    task: str
    step_number: int
    instructions: str


class StepResult(BaseModel):
    observation: Observation
    reward: float = Field(ge=0.0, le=1.0)
    done: bool
    info: dict


class ResetRequest(BaseModel):
    task: str = "easy"
    clause_text: str | None = Field(default=None, max_length=5000)

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        if v not in ("easy", "medium", "hard"):
            raise ValueError("task must be one of: easy, medium, hard")
        return v


class MetricsResponse(BaseModel):
    avg_reward: float
    success_rate: float
    task_breakdown: dict
