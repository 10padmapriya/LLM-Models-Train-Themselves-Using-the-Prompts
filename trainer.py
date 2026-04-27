"""
trainer.py - Self-Training Orchestration Loop
Ties RAG + Generator + Evaluator into a repeating cycle.
Each iteration: retrieve → generate → evaluate → save to JSONL.
"""
import json
from pathlib import Path
from typing import List, Optional

from rag import RAGPipeline
from generator import SyntheticGenerator, Sample
from evaluator import SelfEvaluator


# Diverse queries covering all three document domains
DEFAULT_QUERIES = [
    # Domain QA - ML for business
    "What machine learning techniques are used for business forecasting?",
    "How do I measure the ROI of a machine learning project?",
    "What is the difference between precision and recall in business terms?",
    "When should a company NOT use machine learning?",
    "How do I explain model performance to non-technical stakeholders?",

    # Code assistance
    "How do I handle missing values in a pandas DataFrame?",
    "How do I build a machine learning pipeline in scikit-learn?",
    "What is the correct way to do train test split to avoid data leakage?",
    "How do I tune hyperparameters with GridSearchCV?",
    "How do I calculate and visualize feature importance?",

    # General reasoning
    "How do I structure a data science problem from scratch?",
    "What is the 5 whys technique and how is it used in data analysis?",
    "How do I avoid overfitting my conclusions to the data?",
    "What are the most common reasons ML projects fail in production?",
    "How do I build a business case for a machine learning project?",
]


class SelfTrainingLoop:
    def __init__(self, threshold: float = 0.75, dry_run: bool = True):
        """
        Args:
            threshold: Minimum quality score to keep a sample (0.0 to 1.0)
            dry_run:   If True, skips actual model fine-tuning (no GPU needed)
        """
        print("Initializing pipeline components...")
        self.rag = RAGPipeline()
        self.generator = SyntheticGenerator()
        self.evaluator = SelfEvaluator(threshold=threshold)
        self.dry_run = dry_run
        self.all_passed: List[Sample] = []

        Path("output").mkdir(exist_ok=True)
        print("Ready.\n")

    def run(
        self,
        n_iterations: int = 3,
        samples_per_iter: int = 6,
        queries: Optional[List[str]] = None
    ):
        """
        Run the full self-training loop.

        Args:
            n_iterations:    How many generate → evaluate → train cycles to run
            samples_per_iter: How many samples to generate per context
            queries:         Custom retrieval queries (uses defaults if None)
        """
        queries = queries or DEFAULT_QUERIES
        total_generated = 0
        total_passed = 0

        print("=" * 60)
        print(f"  SELF-TRAINING LOOP")
        print(f"  Iterations:    {n_iterations}")
        print(f"  Samples/iter:  {samples_per_iter}")
        print(f"  Threshold:     {self.evaluator.threshold}")
        print(f"  Dry run:       {self.dry_run}")
        print("=" * 60)

        for i in range(1, n_iterations + 1):
            print(f"\n{'─' * 60}")
            print(f"  ITERATION {i} of {n_iterations}")
            print(f"{'─' * 60}")

            # ── Step 1: Retrieve context ──────────────────────────────
            # Rotate through queries so each iteration uses a different one
            query = queries[(i - 1) % len(queries)]
            print(f"\n[1] Retrieving context for: '{query}'")
            context = self.rag.retrieve(query, k=4)
            print(f"    Context preview: {context[:120].strip()}...")

            # ── Step 2: Generate synthetic pairs ──────────────────────
            print(f"\n[2] Generating {samples_per_iter} samples...")
            samples = self.generator.generate(context, n=samples_per_iter)
            samples = SyntheticGenerator.deduplicate(samples)
            total_generated += len(samples)

            # ── Step 3: Evaluate and filter ───────────────────────────
            print(f"\n[3] Evaluating {len(samples)} samples...")
            passed, failed = self.evaluator.evaluate_batch(samples)
            self.all_passed.extend(passed)
            total_passed += len(passed)

            # ── Step 4: Save the accepted samples ─────────────────────
            if passed:
                path = self._save(passed, iteration=i)
                print(f"\n[4] Saved {len(passed)} samples → {path}")
            else:
                print(f"\n[4] No samples passed the quality threshold this iteration")

            # ── Step 5: Fine-tune (optional, requires GPU) ────────────
            print(f"\n[5] Fine-tuning:")
            if self.dry_run:
                print(f"    [dry_run=True] Skipping — would fine-tune on "
                      f"{len(self.all_passed)} total accepted samples")
            elif len(self.all_passed) < 10:
                print(f"    Waiting for more data "
                      f"({len(self.all_passed)}/10 minimum samples collected)")
            else:
                print(f"    Starting LoRA fine-tune on {len(self.all_passed)} samples...")
                self._finetune()

        # ── Final summary ─────────────────────────────────────────────
        acceptance_rate = total_passed / total_generated * 100 if total_generated else 0
        print(f"\n{'=' * 60}")
        print(f"  COMPLETE")
        print(f"  Total generated:  {total_generated}")
        print(f"  Total accepted:   {total_passed} ({acceptance_rate:.0f}%)")
        print(f"  Total rejected:   {total_generated - total_passed}")
        print(f"  Dataset saved to: output/")
        print(f"{'=' * 60}")

        return self.all_passed

    def _save(self, samples: List[Sample], iteration: int) -> Path:
        """Save accepted samples to a JSONL file."""
        path = Path(f"output/iter_{iteration:03d}.jsonl")
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s.to_jsonl()) + "\n")
        return path

    def _finetune(self):
        """
        Plug your LoRA fine-tuner here.
        Requires: GPU + transformers + peft + trl

        Example:
            from trl import SFTTrainer
            from peft import LoraConfig, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
            from datasets import Dataset

            dataset = Dataset.from_list([s.to_jsonl() for s in self.all_passed])
            model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
            model = get_peft_model(model, lora_config)
            trainer = SFTTrainer(model=model, train_dataset=dataset, ...)
            trainer.train()
        """
        print("    [finetune stub] Would run SFTTrainer + LoRA here")


# ── Run it ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    loop = SelfTrainingLoop(
        threshold=0.75,
        dry_run=True       # Set to False + add GPU to run actual fine-tuning
    )
    loop.run(
        n_iterations=3,
        samples_per_iter=6
    )
