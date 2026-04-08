import random
from pydantic import BaseModel


CLAUSES = [
    {
        "id": 1,
        "text": "Either party may terminate this agreement at any time without cause and without notice.",
        "is_risky": True,
        "risk_category": "termination",
        "risky_phrase": "without notice",
        "safe_rewrite": "Either party may terminate this agreement with 30 days written notice to the other party.",
    },
    {
        "id": 2,
        "text": "The vendor shall not be liable for any damages, whether direct, indirect, incidental, or consequential, under any circumstances.",
        "is_risky": True,
        "risk_category": "liability",
        "risky_phrase": "under any circumstances",
        "safe_rewrite": "The vendor's liability shall not exceed the total fees paid in the 12 months preceding the claim.",
    },
    {
        "id": 3,
        "text": "Payment is due within 30 days of invoice. Late payments incur 1.5% monthly interest.",
        "is_risky": False,
        "risk_category": "safe",
        "risky_phrase": "",
        "safe_rewrite": "",
    },
    {
        "id": 4,
        "text": "All intellectual property created by the contractor during this engagement, whether related to the project or not, shall become the sole property of the client.",
        "is_risky": True,
        "risk_category": "ip_ownership",
        "risky_phrase": "whether related to the project or not",
        "safe_rewrite": "All intellectual property created by the contractor specifically for deliverables under this agreement shall become the property of the client.",
    },
    {
        "id": 5,
        "text": "Confidentiality obligations under this agreement shall survive termination for a period of 2 years.",
        "is_risky": False,
        "risk_category": "safe",
        "risky_phrase": "",
        "safe_rewrite": "",
    },
    {
        "id": 6,
        "text": "The contractor shall indemnify the client against any and all claims, losses, and expenses of any nature whatsoever, including those arising from the client's own negligence.",
        "is_risky": True,
        "risk_category": "indemnification",
        "risky_phrase": "including those arising from the client's own negligence",
        "safe_rewrite": "The contractor shall indemnify the client against claims arising solely from the contractor's own acts or omissions.",
    },
    {
        "id": 7,
        "text": "This agreement shall be governed by the laws of the State of Delaware and disputes shall be resolved by binding arbitration in Wilmington, Delaware.",
        "is_risky": True,
        "risk_category": "governing_law",
        "risky_phrase": "binding arbitration in Wilmington, Delaware",
        "safe_rewrite": "This agreement shall be governed by the laws of the State of Delaware. Disputes shall first be subject to good-faith mediation before any legal proceedings.",
    },
    {
        "id": 8,
        "text": "The client may modify the scope of work at any time, and the contractor must comply within 5 business days at no additional cost.",
        "is_risky": True,
        "risk_category": "termination",
        "risky_phrase": "at no additional cost",
        "safe_rewrite": "The client may request scope modifications in writing. Changes affecting timeline or cost require a signed change order before implementation.",
    },
    {
        "id": 9,
        "text": "Each party shall maintain appropriate insurance coverage for the duration of this agreement.",
        "is_risky": False,
        "risk_category": "safe",
        "risky_phrase": "",
        "safe_rewrite": "",
    },
    {
        "id": 10,
        "text": "The client reserves the right to assign this agreement to any third party without the contractor's consent.",
        "is_risky": True,
        "risk_category": "ip_ownership",
        "risky_phrase": "without the contractor's consent",
        "safe_rewrite": "This agreement may not be assigned by either party without the prior written consent of the other party, which shall not be unreasonably withheld.",
    },
]


TASK_INSTRUCTIONS = {
    "easy": "Read the clause. Respond with is_risky (true/false) only.",
    "medium": "Read the clause. Identify is_risky, risk_category, and risky_phrase.",
    "hard": "Read the clause. Identify is_risky, risk_category, risky_phrase, and provide a safe rewrite.",
}


class ContractAction(BaseModel):
    is_risky: bool
    risk_category: str = "safe"
    risky_phrase: str = ""
    rewrite: str = ""


class ContractObservation(BaseModel):
    clause_id: int
    clause_text: str
    task: str
    step_number: int
    instructions: str


class ContractReward(BaseModel):
    score: float
    breakdown: dict


class ContractRedlineEnv:
    MAX_STEPS = 3

    def __init__(self, task: str = "easy"):
        assert task in ("easy", "medium", "hard")
        self.task = task
        self.current_clause = None
        self.step_count = 0
        self.done = False
        self.last_reward = 0.0
        self.episode_rewards: list[float] = []

    def reset(self) -> ContractObservation:
        self.current_clause = random.choice(CLAUSES)
        self.step_count = 0
        self.done = False
        self.last_reward = 0.0
        self.episode_rewards = []
        return ContractObservation(
            clause_id=self.current_clause["id"],
            clause_text=self.current_clause["text"],
            task=self.task,
            step_number=self.step_count,
            instructions=TASK_INSTRUCTIONS[self.task],
        )

    def step(self, action: ContractAction) -> tuple:
        self.step_count += 1
        reward = self._grade(action)
        self.last_reward = reward.score
        self.done = reward.score >= 0.8 or self.step_count >= self.MAX_STEPS
        self.episode_rewards.append(reward.score)

        obs = ContractObservation(
            clause_id=self.current_clause["id"],
            clause_text=self.current_clause["text"],
            task=self.task,
            step_number=self.step_count,
            instructions=TASK_INSTRUCTIONS[self.task],
        )
        return (obs, reward.score, self.done, {"breakdown": reward.breakdown})

    def _grade(self, action: ContractAction) -> ContractReward:
        clause = self.current_clause
        breakdown = {}
        score = 0.0

        if self.task == "easy":
            if action.is_risky == clause["is_risky"]:
                score = 1.0
                breakdown["is_risky"] = 1.0
            else:
                score = 0.0
                breakdown["is_risky"] = 0.0

        elif self.task == "medium":
            # is_risky: +0.3
            if action.is_risky == clause["is_risky"]:
                breakdown["is_risky"] = 0.3
                score += 0.3
            else:
                breakdown["is_risky"] = 0.0

            # risk_category: +0.4
            if action.risk_category.strip().lower() == clause["risk_category"].lower():
                breakdown["risk_category"] = 0.4
                score += 0.4
            else:
                breakdown["risk_category"] = 0.0

            # risky_phrase: +0.3
            phrase_score = self._score_phrase(action.risky_phrase, clause["risky_phrase"])
            phrase_weighted = round(phrase_score * 0.3, 4)
            breakdown["risky_phrase"] = phrase_weighted
            score += phrase_weighted

        elif self.task == "hard":
            # is_risky: +0.15
            if action.is_risky == clause["is_risky"]:
                breakdown["is_risky"] = 0.15
                score += 0.15
            else:
                breakdown["is_risky"] = 0.0

            # risk_category: +0.25
            if action.risk_category.strip().lower() == clause["risk_category"].lower():
                breakdown["risk_category"] = 0.25
                score += 0.25
            else:
                breakdown["risk_category"] = 0.0

            # risky_phrase: +0.20
            phrase_score = self._score_phrase(action.risky_phrase, clause["risky_phrase"])
            phrase_weighted = round(phrase_score * 0.20, 4)
            breakdown["risky_phrase"] = phrase_weighted
            score += phrase_weighted

            # rewrite quality: +0.40
            rewrite_score = self._score_rewrite(action.rewrite, clause)
            breakdown["rewrite"] = round(rewrite_score, 4)
            score += rewrite_score

        return ContractReward(score=min(round(score, 4), 1.0), breakdown=breakdown)

    def _score_phrase(self, predicted: str, expected: str) -> float:
        if not expected:
            return 1.0 if not predicted else 0.5

        pred_lower = predicted.strip().lower()
        exp_lower = expected.strip().lower()

        if exp_lower in pred_lower or pred_lower in exp_lower:
            return 1.0

        pred_words = set(pred_lower.split())
        exp_words = set(exp_lower.split())
        overlap = pred_words & exp_words
        if len(overlap) >= 3:
            return 0.5

        return 0.0

    def _score_rewrite(self, rewrite: str, clause: dict) -> float:
        if not rewrite or not rewrite.strip():
            return 0.0

        rewrite_stripped = rewrite.strip()
        word_count = len(rewrite_stripped.split())
        original_text = clause["text"]
        risky_phrase = clause["risky_phrase"]
        differs = rewrite_stripped.lower() != original_text.lower()
        risky_absent = (
            risky_phrase.lower() not in rewrite_stripped.lower() if risky_phrase else True
        )

        if word_count > 20 and risky_absent and differs:
            return 0.40
        elif word_count > 10 and differs:
            return 0.20
        else:
            return 0.10

    def state(self) -> dict:
        return {
            "task": self.task,
            "step_count": self.step_count,
            "done": self.done,
            "current_clause_id": self.current_clause["id"] if self.current_clause else None,
            "episode_rewards": self.episode_rewards,
            "last_reward": self.last_reward,
        }

    def close(self):
        pass
