"""
evaluator.py - Self-Evaluator (LLM as Judge)
A stronger LLM scores each generated sample on 4 dimensions.
Samples below the threshold are rejected before training.
"""
import json
import re
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from generator import Sample
from dotenv import load_dotenv

load_dotenv()

# How much each dimension contributes to the final score (must sum to 1.0)
WEIGHTS = {
    "accuracy":    0.40,   # Is the response factually correct?
    "relevance":   0.30,   # Does it directly answer the prompt?
    "coherence":   0.20,   # Is it clearly written and logical?
    "helpfulness": 0.10    # Would a real user find this useful?
}


class SelfEvaluator:
    def __init__(self, threshold: float = 0.75):
        # Use a stronger model as judge than the generator
        # gpt-4o is more critical and accurate than gpt-4o-mini
        self.judge = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.threshold = threshold

    def evaluate(self, sample: Sample) -> Sample:
        """Score a single sample. Updates sample in place and returns it."""

        user = f"""You are a rigorous quality evaluator for AI training data.
Score this prompt-response pair on each dimension from 0.0 to 1.0.
Be critical. Reserve scores above 0.9 for truly excellent responses.

PROMPT:
{sample.prompt}

RESPONSE:
{sample.response}

Score each dimension:
- accuracy: Is the response factually correct and technically sound?
- relevance: Does it directly and completely address the prompt?
- coherence: Is it clearly written, well-structured, and logical?
- helpfulness: Would a real person asking this find it genuinely useful?

Respond ONLY with JSON (no markdown):
{{
  "accuracy": 0.0,
  "relevance": 0.0,
  "coherence": 0.0,
  "helpfulness": 0.0,
  "reasoning": "One sentence explanation"
}}"""

        try:
            response = self.judge.invoke([
                SystemMessage(content="You are a rigorous AI training data quality evaluator."),
                HumanMessage(content=user)
            ])
            scores = self._parse(response.content)

            # Compute weighted score
            sample.quality_score = round(
                sum(scores.get(dim, 0.0) * w for dim, w in WEIGHTS.items()), 4
            )
            sample.passed = sample.quality_score >= self.threshold

        except Exception as e:
            print(f"  Evaluation error for {sample.id}: {e}")
            sample.quality_score = 0.0
            sample.passed = False

        return sample

    def evaluate_batch(
        self, samples: List[Sample]
    ) -> Tuple[List[Sample], List[Sample]]:
        """
        Evaluate all samples. Returns (passed_list, failed_list).
        Prints a status line for each sample.
        """
        print(f"\nEvaluating {len(samples)} samples (threshold={self.threshold})...")
        print("-" * 55)

        for i, sample in enumerate(samples, 1):
            self.evaluate(sample)
            icon = "✅" if sample.passed else "❌"
            print(
                f"  {icon} [{sample.id}] score={sample.quality_score:.3f} | "
                f"{sample.prompt[:55]}..."
            )

        passed = [s for s in samples if s.passed]
        failed = [s for s in samples if not s.passed]

        rate = len(passed) / len(samples) * 100 if samples else 0
        avg  = sum(s.quality_score for s in samples) / len(samples) if samples else 0

        print("-" * 55)
        print(
            f"  Result: {len(passed)} passed / {len(failed)} rejected | "
            f"acceptance={rate:.0f}% | avg_score={avg:.3f}"
        )
        return passed, failed

    def _parse(self, text: str) -> dict:
        """Extract JSON scores from judge response."""
        text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text)


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from generator import SyntheticGenerator

    gen = SyntheticGenerator()
    evaluator = SelfEvaluator(threshold=0.75)

    context = """
    Gradient boosting builds an ensemble of decision trees sequentially.
    Each tree corrects the errors of the previous one. XGBoost and LightGBM
    are popular implementations that are often the top performers on tabular data.
    """

    print("=== Generating ===")
    samples = gen.generate(context, n=5)
    samples = gen.deduplicate(samples)

    print("\n=== Evaluating ===")
    passed, failed = evaluator.evaluate_batch(samples)

    print(f"\n{len(passed)} samples ready for fine-tuning")
