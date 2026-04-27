"""
mlops.py - MLOps: Monitoring and Drift Detection
Tracks quality scores across iterations, logs to MLflow,
and detects when the score distribution starts to shift.
"""
import numpy as np
from scipy.stats import entropy
from typing import List
import mlflow

from generator import Sample


class MLOpsTracker:
    def __init__(
        self,
        experiment_name: str = "llm-self-training",
        drift_threshold: float = 0.15
    ):
        """
        Args:
            experiment_name: MLflow experiment to log into
            drift_threshold: KL divergence above this = drift alert
        """
        mlflow.set_experiment(experiment_name)
        self.score_history: List[float] = []
        self.drift_threshold = drift_threshold
        print(f"MLOpsTracker ready | experiment='{experiment_name}'")

    def log_iteration(
        self,
        iteration: int,
        passed: List[Sample],
        total_generated: int,
        run_id: str = None
    ):
        """Log metrics for one training iteration to MLflow."""
        scores = [s.quality_score for s in passed if s.quality_score is not None]
        self.score_history.extend(scores)

        acceptance_rate = len(passed) / total_generated if total_generated else 0
        avg_score = np.mean(scores) if scores else 0
        drift = self.check_drift()

        metrics = {
            "acceptance_rate":  round(acceptance_rate, 4),
            "avg_quality_score": round(float(avg_score), 4),
            "samples_passed":   len(passed),
            "total_generated":  total_generated,
            "drift_detected":   int(drift),
            "cumulative_samples": len(self.score_history),
        }

        with mlflow.start_run(run_name=f"iteration_{iteration}", nested=True):
            mlflow.log_metrics(metrics, step=iteration)
            mlflow.log_param("iteration", iteration)
            mlflow.log_param("drift_threshold", self.drift_threshold)

        status = "⚠️  DRIFT" if drift else "✅ stable"
        print(
            f"\n  📊 Iteration {iteration} | "
            f"acceptance={acceptance_rate:.0%} | "
            f"avg_score={avg_score:.3f} | "
            f"distribution={status}"
        )
        return drift

    def check_drift(
        self,
        ref_window: int = 30,
        cur_window: int = 15
    ) -> bool:
        """
        Compare the current score distribution to the reference (first N samples).
        Uses KL divergence. Returns True if drift exceeds threshold.

        KL divergence = 0 means identical distributions.
        KL divergence > threshold means the distribution has shifted.
        """
        if len(self.score_history) < ref_window + cur_window:
            return False  # Not enough data yet

        ref = np.array(self.score_history[:ref_window])
        cur = np.array(self.score_history[-cur_window:])

        # Convert continuous scores to probability histograms
        bins = np.linspace(0, 1, 15)
        p, _ = np.histogram(ref, bins=bins, density=True)
        q, _ = np.histogram(cur, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10
        p /= p.sum()
        q /= q.sum()

        kl_divergence = float(entropy(q, p))
        is_drift = kl_divergence > self.drift_threshold

        if is_drift:
            print(
                f"\n  ⚠️  DRIFT DETECTED: KL={kl_divergence:.4f} "
                f"> threshold={self.drift_threshold}"
            )
        return is_drift

    def summary(self) -> dict:
        """Return descriptive stats on all quality scores recorded so far."""
        if not self.score_history:
            return {}
        arr = np.array(self.score_history)
        return {
            "total_samples": len(arr),
            "mean":  round(float(np.mean(arr)), 4),
            "std":   round(float(np.std(arr)), 4),
            "min":   round(float(np.min(arr)), 4),
            "max":   round(float(np.max(arr)), 4),
            "p25":   round(float(np.percentile(arr, 25)), 4),
            "p75":   round(float(np.percentile(arr, 75)), 4),
        }

    def print_summary(self):
        stats = self.summary()
        if not stats:
            print("No data recorded yet.")
            return
        print("\n📊 Overall Quality Score Summary")
        print(f"   Samples tracked : {stats['total_samples']}")
        print(f"   Mean score       : {stats['mean']:.4f}")
        print(f"   Std deviation    : {stats['std']:.4f}")
        print(f"   Range            : [{stats['min']:.4f} – {stats['max']:.4f}]")
        print(f"   25th / 75th pct  : {stats['p25']:.4f} / {stats['p75']:.4f}")


# ── Integrated run with full loop ────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from rag import RAGPipeline
    from generator import SyntheticGenerator
    from evaluator import SelfEvaluator

    rag = RAGPipeline()
    gen = SyntheticGenerator()
    evaluator = SelfEvaluator(threshold=0.75)
    tracker = MLOpsTracker(drift_threshold=0.15)

    QUERIES = [
        "How do I prevent overfitting in machine learning?",
        "What are the best practices for feature engineering?",
        "How do I communicate model results to business stakeholders?",
    ]

    print("Starting MLOps-instrumented training loop...\n")
    with mlflow.start_run(run_name="self_training_full_run"):
        mlflow.log_params({
            "quality_threshold": 0.75,
            "drift_threshold": 0.15,
            "n_iterations": len(QUERIES),
        })

        for i, query in enumerate(QUERIES, 1):
            print(f"\n{'─'*50}")
            print(f"Iteration {i}: {query}")

            context = rag.retrieve(query)
            samples = gen.generate(context, n=5)
            samples = SyntheticGenerator.deduplicate(samples)
            passed, _ = evaluator.evaluate_batch(samples)

            tracker.log_iteration(
                iteration=i,
                passed=passed,
                total_generated=len(samples)
            )

    tracker.print_summary()
    print("\nOpen MLflow dashboard: mlflow ui --port 5000")
    print("Then visit: http://localhost:5000")
