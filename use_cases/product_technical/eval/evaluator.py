# Re-export from use-cases/product-technical/eval/evaluator.py
import importlib.util
from pathlib import Path

# Load the original module with a unique name
_original_path = Path(__file__).parent.parent.parent.parent / "use-cases" / "product-technical" / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("_product_technical_evaluator", _original_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
CodeEvaluator = _module.CodeEvaluator
CodeEvaluationResult = _module.CodeEvaluationResult
evaluate_sample = _module.evaluate_sample

__all__ = ["CodeEvaluator", "CodeEvaluationResult", "evaluate_sample"]
