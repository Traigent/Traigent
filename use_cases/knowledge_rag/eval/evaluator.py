# Re-export from use-cases/knowledge-rag/eval/evaluator.py
import importlib.util
from pathlib import Path

# Load the original module with a unique name
_original_path = Path(__file__).parent.parent.parent.parent / "use-cases" / "knowledge-rag" / "eval" / "evaluator.py"
_spec = importlib.util.spec_from_file_location("_knowledge_rag_evaluator", _original_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
RAGEvaluator = _module.RAGEvaluator
RAGEvaluationResult = _module.RAGEvaluationResult
evaluate_sample = _module.evaluate_sample

__all__ = ["RAGEvaluator", "RAGEvaluationResult", "evaluate_sample"]
