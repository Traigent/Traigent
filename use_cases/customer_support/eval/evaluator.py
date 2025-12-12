# Re-export from use-cases/customer-support/eval/evaluator.py
import importlib.util
from pathlib import Path

# Load the original module with a unique name
_original_path = Path(__file__).parent.parent.parent.parent / "use-cases" / "customer-support" / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("_customer_support_evaluator", _original_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
SupportEvaluator = _module.SupportEvaluator
SupportEvaluationResult = _module.SupportEvaluationResult
evaluate_sample = _module.evaluate_sample

__all__ = ["SupportEvaluator", "SupportEvaluationResult", "evaluate_sample"]
