from models import Action, Observation, RiskCategory


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


_AUTO_LABEL_PHRASES = [
    ("without notice", "termination"),
    ("without cause", "termination"),
    ("at no additional cost", "termination"),
    ("under any circumstances", "liability"),
    ("not be liable for any damages", "liability"),
    ("not be liable", "liability"),
    ("including those arising from the client's own negligence", "indemnification"),
    ("client's own negligence", "indemnification"),
    ("indemnify", "indemnification"),
    ("whether related to the project or not", "ip_ownership"),
    ("sole property of the client", "ip_ownership"),
    ("without the contractor's consent", "ip_ownership"),
    ("assign this agreement to any third party", "ip_ownership"),
    ("intellectual property", "ip_ownership"),
    ("binding arbitration", "governing_law"),
    ("jurisdiction", "governing_law"),
    ("governing law", "governing_law"),
]


def auto_label_clause(text: str) -> dict:
    """Generate ground-truth labels for a custom clause using rule-based detection."""
    text_lower = text.lower()
    for phrase, category in _AUTO_LABEL_PHRASES:
        if phrase in text_lower:
            start = text_lower.find(phrase)
            risky_phrase = text[start:start + len(phrase)]
            return {
                "id": -1,
                "text": text,
                "is_risky": True,
                "risk_category": category,
                "risky_phrase": risky_phrase,
                "safe_rewrite": "",
            }
    return {
        "id": -1,
        "text": text,
        "is_risky": False,
        "risk_category": "safe",
        "risky_phrase": "",
        "safe_rewrite": "",
    }


class ContractRedlineEnv:
    MAX_STEPS = 3

    _clause_index: int = 0

    def __init__(self, task: str = "easy"):
        if task not in ("easy", "medium", "hard"):
            raise ValueError("task must be one of: easy, medium, hard")
        self.task = task
        self.current_clause: dict | None = None
        self.step_count = 0
        self.done = False
        self.last_reward = 0.0
        self.episode_rewards: list[float] = []

    def reset(self, custom_clause_text: str | None = None) -> Observation:
        if custom_clause_text and custom_clause_text.strip():
            self.current_clause = auto_label_clause(custom_clause_text.strip())
        else:
            self.current_clause = CLAUSES[ContractRedlineEnv._clause_index % len(CLAUSES)]
            ContractRedlineEnv._clause_index += 1
        self.step_count = 0
        self.done = False
        self.last_reward = 0.0
        self.episode_rewards = []
        return Observation(
            clause_id=self.current_clause["id"],
            clause_text=self.current_clause["text"],
            task=self.task,
            step_number=self.step_count,
            instructions=TASK_INSTRUCTIONS[self.task],
        )

    def step(self, action: Action) -> tuple:
        if self.done:
            raise RuntimeError("Episode is done. Call /reset to start a new episode.")
        if self.current_clause is None:
            raise RuntimeError("No active episode. Call /reset first.")

        self.step_count += 1
        reward, breakdown = self._grade(action)
        self.last_reward = reward
        self.done = reward >= 0.8 or self.step_count >= self.MAX_STEPS
        self.episode_rewards.append(reward)

        obs = Observation(
            clause_id=self.current_clause["id"],
            clause_text=self.current_clause["text"],
            task=self.task,
            step_number=self.step_count,
            instructions=TASK_INSTRUCTIONS[self.task],
        )
        return (obs, reward, self.done, {"breakdown": breakdown})

    def _grade(self, action: Action) -> tuple[float, dict]:
        clause = self.current_clause
        breakdown: dict[str, float] = {}
        score = 0.0

        cat_value = action.risk_category.value if isinstance(action.risk_category, RiskCategory) else str(action.risk_category).strip().lower()

        if self.task == "easy":
            if action.is_risky == clause["is_risky"]:
                score = 1.0
                breakdown["is_risky"] = 1.0
            else:
                score = 0.0
                breakdown["is_risky"] = 0.0

        elif self.task == "medium":
            if action.is_risky == clause["is_risky"]:
                breakdown["is_risky"] = 0.3
                score += 0.3
            else:
                breakdown["is_risky"] = 0.0

            if cat_value == clause["risk_category"].lower():
                breakdown["risk_category"] = 0.4
                score += 0.4
            else:
                breakdown["risk_category"] = 0.0

            phrase_score = self._score_phrase(action.risky_phrase, clause["risky_phrase"])
            phrase_weighted = round(phrase_score * 0.3, 4)
            breakdown["risky_phrase"] = phrase_weighted
            score += phrase_weighted

        elif self.task == "hard":
            if action.is_risky == clause["is_risky"]:
                breakdown["is_risky"] = 0.15
                score += 0.15
            else:
                breakdown["is_risky"] = 0.0

            if cat_value == clause["risk_category"].lower():
                breakdown["risk_category"] = 0.25
                score += 0.25
            else:
                breakdown["risk_category"] = 0.0

            phrase_score = self._score_phrase(action.risky_phrase, clause["risky_phrase"])
            phrase_weighted = round(phrase_score * 0.20, 4)
            breakdown["risky_phrase"] = phrase_weighted
            score += phrase_weighted

            rewrite_score = self._score_rewrite(action.rewrite, clause)
            breakdown["rewrite"] = round(rewrite_score, 4)
            score += rewrite_score

        final_score = min(round(score, 4), 1.0)
        return final_score, breakdown

    def _score_phrase(self, predicted: str, expected: str) -> float:
        if not expected:
            return 1.0 if not predicted else 0.5

        pred_lower = predicted.strip().lower()
        exp_lower = expected.strip().lower()

        if not pred_lower:
            return 0.0

        if exp_lower == pred_lower:
            return 1.0

        if exp_lower in pred_lower or pred_lower in exp_lower:
            return 0.9

        pred_tokens = set(pred_lower.split())
        exp_tokens = set(exp_lower.split())
        if not exp_tokens:
            return 0.0
        overlap = pred_tokens & exp_tokens
        overlap_ratio = len(overlap) / len(exp_tokens)
        if overlap_ratio >= 0.75:
            return 0.7
        if overlap_ratio >= 0.5:
            return 0.5
        if len(overlap) >= 2:
            return 0.3

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
        min_length = word_count >= 5

        if not min_length:
            return 0.05

        if word_count > 20 and risky_absent and differs:
            return 0.40
        elif word_count > 10 and risky_absent and differs:
            return 0.30
        elif word_count > 10 and differs:
            return 0.20
        elif differs:
            return 0.10
        else:
            return 0.05

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
