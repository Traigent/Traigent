# Re-export from use-cases/gtm-acquisition/eval/evaluator.py
import importlib.util
from pathlib import Path

# Load the original module with a unique name
_original_path = Path(__file__).parent.parent.parent.parent / "use-cases" / "gtm-acquisition" / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("_gtm_evaluator", _original_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
MessageQualityEvaluator = _module.MessageQualityEvaluator
MessageQualityResult = _module.MessageQualityResult
evaluate_dataset_sample = _module.evaluate_dataset_sample

__all__ = ["MessageQualityEvaluator", "MessageQualityResult", "evaluate_dataset_sample"]
