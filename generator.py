"""
generator.py - Synthetic Data Generator
Asks the LLM to create diverse prompt/response training pairs
from retrieved context. Returns structured Sample objects.
"""
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Sample:
    """One prompt/response training pair."""
    prompt: str
    response: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    quality_score: Optional[float] = None
    passed: bool = False

    def to_jsonl(self) -> dict:
        """Format ready for fine-tuning."""
        return {
            "messages": [
                {"role": "user", "content": self.prompt},
                {"role": "assistant", "content": self.response}
            ],
            "quality_score": self.quality_score
        }


class SyntheticGenerator:
    def __init__(self):
        # gpt-4o-mini: cheap and fast for generating lots of pairs
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.85)

    def generate(self, context: str, n: int = 5) -> List[Sample]:
        """Generate n diverse prompt/response pairs from a context string."""

        system = (
            "You are an expert at creating high-quality, diverse training data "
            "for AI assistants specializing in data science and machine learning."
        )

        user = f"""Given this context, generate {n} realistic and diverse prompt-response pairs.
Cover different types: direct questions, how-to requests, explain-this, compare-that,
give-an-example, code help, and business reasoning.

Context:
{context}

Rules:
- Make prompts sound like a real person asking a real question
- Make responses accurate, detailed, and genuinely helpful
- Vary difficulty: some simple, some advanced
- Do NOT copy the context word for word in the response

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "pairs": [
    {{"prompt": "...", "response": "..."}},
    {{"prompt": "...", "response": "..."}}
  ]
}}"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content=user)
            ])
            return self._parse(response.content)
        except Exception as e:
            print(f"  Generation error: {e}")
            return []

    def _parse(self, text: str) -> List[Sample]:
        """Safely extract JSON from LLM response."""
        # Remove markdown fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip("` \n")

        # Find the JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            print("  Could not find JSON in response")
            return []

        try:
            data = json.loads(match.group())
            samples = []
            for p in data.get("pairs", []):
                if p.get("prompt") and p.get("response"):
                    samples.append(Sample(
                        prompt=p["prompt"].strip(),
                        response=p["response"].strip()
                    ))
            return samples
        except json.JSONDecodeError as e:
            print(f"  JSON parse error: {e}")
            return []

    @staticmethod
    def deduplicate(samples: List[Sample]) -> List[Sample]:
        """Remove near-duplicate prompts based on first 100 characters."""
        seen, unique = set(), []
        for s in samples:
            key = s.prompt.lower().strip()[:100]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        removed = len(samples) - len(unique)
        if removed > 0:
            print(f"  Removed {removed} near-duplicate prompts")
        return unique


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = SyntheticGenerator()

    context = """
    Random Forest is an ensemble learning method that builds many decision trees
    on random subsets of the training data and averages their predictions.
    It handles missing values, requires little preprocessing, and provides
    feature importance scores. It is robust to overfitting compared to a
    single decision tree.
    """

    print("=== Generating samples ===")
    samples = gen.generate(context, n=4)
    samples = gen.deduplicate(samples)

    for s in samples:
        print(f"\n[{s.id}] PROMPT:   {s.prompt}")
        print(f"       RESPONSE: {s.response[:120]}...")
