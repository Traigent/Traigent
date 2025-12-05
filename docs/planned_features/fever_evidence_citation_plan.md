# FEVER Evidence Citation Implementation Plan

## Problem Statement

The FEVER case study currently shows 0% evidence recall because:
- The pipeline passes raw retrieval results as "evidence" but the LLM never explicitly cites which evidence supports its verdict
- The official FEVER scorer expects exact `[page, line]` pairs that match gold annotations
- Since predicted_evidence is always empty, recall is always 0%

## Root Cause Analysis

### Current Flow (Broken)
```
1. collect_candidate_evidence() → retrieves top-k sentences
2. LLM sees evidence in prompt → makes verdict decision
3. Pipeline returns: {verdict, justification, evidence: [raw_retrieval]}
4. Metrics extracts: predicted_evidence = [] (wrong format)
5. Scorer compares: [] vs gold_evidence → 0% recall
```

### Expected Flow
```
1. collect_candidate_evidence() → retrieves top-k sentences
2. LLM sees numbered evidence → cites specific items
3. Pipeline parses citations → extracts [page, line] pairs
4. Metrics formats: predicted_evidence = [[page1, line1], ...]
5. Scorer compares: cited evidence vs gold → proper recall %
```

## Implementation Phases

### Phase 1: Modify LLM Prompt for Structured Citations

**File**: `paper_experiments/case_study_fever/pipeline.py`

**Changes**: Update the prompt construction (lines 244-257) to:

```python
def _build_real_pipeline():
    def _run(claim: str) -> dict[str, Any]:
        # ... existing code until line 243 ...

        # NEW: Build numbered evidence list for citations
        evidence_text = ""
        for i, item in enumerate(evidence_items, 1):
            page = item.get('page', 'Unknown')
            line = item.get('line', 0)
            text = item.get('text', '(text not available)')
            evidence_text += f"{i}. [{page}, line {line}]: {text}\n"

        # NEW: Structured prompt requesting citations
        prompt = (
            f"Claim: {claim}\n\n"
            "Numbered evidence sentences from Wikipedia:\n"
            f"{evidence_text}\n"
            "Task: Analyze the claim using the evidence above.\n\n"
            "Instructions:\n"
            "1. Identify which evidence sentences (by number) are relevant\n"
            "2. Explain your reasoning\n"
            "3. Provide the final verdict\n\n"
            "Required format:\n"
            "CITED_EVIDENCE: [comma-separated numbers, e.g., 1,3,5]\n"
            "REASONING: [Your analysis here]\n"
            "VERDICT: [SUPPORTS/REFUTES/NOT ENOUGH INFO]\n"
        )

        # Continue with existing LLM invocation...
```

### Phase 2: Add Evidence Citation Parser

**File**: `paper_experiments/case_study_fever/pipeline.py`

**New functions** to add after the imports section:

```python
import re
from typing import Any, Final

def _extract_cited_evidence(output_text: str, evidence_items: list[dict]) -> list[dict]:
    """Extract cited evidence indices from LLM response and map to [page, line] format."""
    cited = []

    # Look for CITED_EVIDENCE: pattern
    pattern = r'CITED_EVIDENCE:\s*\[?([\d,\s]+)\]?'
    match = re.search(pattern, output_text, re.IGNORECASE)

    if match:
        numbers_str = match.group(1)
        try:
            # Parse comma-separated numbers
            cited_indices = [
                int(n.strip())
                for n in numbers_str.split(',')
                if n.strip().isdigit()
            ]

            # Map 1-indexed citations back to evidence items
            for idx in cited_indices:
                if 1 <= idx <= len(evidence_items):
                    item = evidence_items[idx - 1]
                    cited.append({
                        'page': str(item.get('page', '')).strip(),
                        'line': int(item.get('line', 0))
                    })

        except (ValueError, IndexError) as e:
            logger.debug("Failed to parse citations from '%s': %s", numbers_str, e)

    return cited


def _extract_structured_response(output_text: str) -> tuple[str, str]:
    """Extract verdict and reasoning from structured LLM response."""
    verdict = ""
    reasoning = output_text  # Default to full text if parsing fails

    # Extract VERDICT
    verdict_match = re.search(r'VERDICT:\s*([^\n]+)', output_text, re.IGNORECASE)
    if verdict_match:
        verdict = verdict_match.group(1).strip()

    # Extract REASONING
    reasoning_match = re.search(
        r'REASONING:\s*([^\n]+(?:\n(?!VERDICT:)[^\n]+)*)',
        output_text,
        re.IGNORECASE
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Fallback to existing verdict extraction if structured parsing fails
    if not verdict:
        verdict = _extract_verdict_text(output_text)

    return verdict, reasoning
```

### Phase 3: Update Pipeline Return Value

**File**: `paper_experiments/case_study_fever/pipeline.py`

**Modify** the `_run` function return (around line 266-270):

```python
def _build_real_pipeline():
    def _run(claim: str) -> dict[str, Any]:
        # ... existing code through LLM invocation ...

        # Get LLM response (line 261-263 stay the same)
        if "gpt" in lowered:
            output_text = _invoke_openai(model, prompt, temperature)
        else:
            output_text = _invoke_anthropic(model, prompt, temperature)

        # NEW: Parse structured response and extract citations
        verdict_text, reasoning = _extract_structured_response(output_text)
        cited_evidence = _extract_cited_evidence(output_text, evidence_items)

        # Return with cited evidence instead of raw retrieval
        return _normalize_pipeline_output(
            verdict=verdict_text,
            justification=reasoning,
            evidence=cited_evidence  # Now contains only LLM-cited evidence
        )
```

### Phase 4: Enhanced Function Calling Approach (Optional - Better for GPT-4)

**File**: `paper_experiments/case_study_fever/pipeline.py`

**Alternative implementation** using OpenAI function calling for more reliable structured output:

```python
def _invoke_openai_structured(
    model: str,
    claim: str,
    evidence_items: list[dict],
    temperature: float
) -> dict[str, Any]:
    """Use OpenAI function calling for structured evidence citations."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai package required") from exc

    client = OpenAI()

    # Prepare numbered evidence for prompt
    evidence_text = ""
    for i, item in enumerate(evidence_items, 1):
        evidence_text += f"{i}. [{item.get('page')}, line {item.get('line')}]: {item.get('text', '')}\n"

    # Define function schema
    functions = [{
        "name": "submit_fever_verdict",
        "description": "Submit verdict with cited evidence for FEVER claim",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"],
                    "description": "The FEVER verdict"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of how evidence supports the verdict"
                },
                "cited_evidence_numbers": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                    "description": "Numbers (1-indexed) of evidence sentences used"
                }
            },
            "required": ["verdict", "reasoning", "cited_evidence_numbers"]
        }
    }]

    messages = [{
        "role": "user",
        "content": (
            f"Claim: {claim}\n\n"
            f"Evidence:\n{evidence_text}\n"
            "Analyze the claim using the numbered evidence and provide your verdict."
        )
    }]

    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call={"name": "submit_fever_verdict"},
        temperature=temperature,
        max_tokens=250
    )
    latency_ms = (time.perf_counter() - started) * 1000.0

    # Extract and parse function call
    function_call = response.choices[0].message.function_call
    if function_call:
        import json
        args = json.loads(function_call.arguments)

        # Map cited numbers to evidence items
        cited_evidence = []
        for num in args.get('cited_evidence_numbers', []):
            if 1 <= num <= len(evidence_items):
                item = evidence_items[num - 1]
                cited_evidence.append({
                    'page': item.get('page', ''),
                    'line': item.get('line', 0)
                })

        # Capture metrics
        capture_langchain_response({
            "raw_response": response,
            "response_time_ms": latency_ms,
            "usage": response.usage.model_dump() if response.usage else {}
        })

        return {
            'verdict': args.get('verdict', ''),
            'justification': args.get('reasoning', ''),
            'evidence': cited_evidence
        }

    # Fallback if function calling fails
    return {
        'verdict': '',
        'justification': str(response.choices[0].message.content),
        'evidence': []
    }
```

### Phase 5: Add Validation to Pipeline Output

**File**: `paper_experiments/case_study_fever/pipeline.py`

**Update** `_normalize_pipeline_output` (around line 74-83):

```python
def _normalize_pipeline_output(
    *,
    verdict: str,
    justification: str,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize and validate pipeline output."""

    # Ensure evidence is properly formatted for scorer
    validated_evidence = []
    if evidence:
        for item in evidence:
            if isinstance(item, dict) and 'page' in item and 'line' in item:
                try:
                    validated_evidence.append({
                        'page': str(item['page']).strip(),
                        'line': int(item['line'])
                    })
                except (ValueError, TypeError) as e:
                    logger.debug("Invalid evidence item %s: %s", item, e)

    return {
        "verdict": verdict.strip() if verdict else "",
        "justification": justification.strip() if justification else "",
        "evidence": validated_evidence,
    }
```

### Phase 6: Testing

**New file**: `tests/unit/case_study_fever/test_evidence_citation.py`

```python
"""Tests for FEVER evidence citation extraction."""

import pytest
from paper_experiments.case_study_fever.pipeline import (
    _extract_cited_evidence,
    _extract_structured_response,
    _normalize_pipeline_output
)


def test_extract_cited_evidence():
    """Test extraction of cited evidence indices."""
    evidence_items = [
        {'page': 'Barack_Obama', 'line': 1, 'text': 'Born in Hawaii'},
        {'page': 'Barack_Obama', 'line': 5, 'text': '44th president'},
        {'page': 'Hawaii', 'line': 10, 'text': 'US state'}
    ]

    # Test various formats
    test_cases = [
        ("CITED_EVIDENCE: [1, 3]", [0, 2]),  # Normal format
        ("CITED_EVIDENCE: 1,2,3", [0, 1, 2]),  # No brackets
        ("cited_evidence: [2]", [1]),  # Case insensitive
        ("No citations here", []),  # No match
    ]

    for output_text, expected_indices in test_cases:
        cited = _extract_cited_evidence(output_text, evidence_items)
        assert len(cited) == len(expected_indices)
        for i, exp_idx in enumerate(expected_indices):
            assert cited[i]['page'] == evidence_items[exp_idx]['page']
            assert cited[i]['line'] == evidence_items[exp_idx]['line']


def test_extract_structured_response():
    """Test extraction of verdict and reasoning."""
    test_output = """
    CITED_EVIDENCE: [1, 2]
    REASONING: The evidence clearly shows Obama was born in Hawaii.
    VERDICT: SUPPORTS
    """

    verdict, reasoning = _extract_structured_response(test_output)
    assert verdict == "SUPPORTS"
    assert "clearly shows Obama" in reasoning

    # Test fallback for unstructured response
    unstructured = "I think this claim is supported. Verdict: REFUTES"
    verdict, reasoning = _extract_structured_response(unstructured)
    assert verdict == "REFUTES" or "REFUTES" in verdict


def test_normalize_pipeline_output():
    """Test output normalization and validation."""
    # Valid evidence
    result = _normalize_pipeline_output(
        verdict="SUPPORTS",
        justification="Test reasoning",
        evidence=[
            {'page': 'Test_Page', 'line': 1},
            {'page': 'Another_Page', 'line': 5}
        ]
    )
    assert len(result['evidence']) == 2
    assert result['evidence'][0]['page'] == 'Test_Page'
    assert result['evidence'][0]['line'] == 1

    # Invalid evidence items should be filtered
    result = _normalize_pipeline_output(
        verdict="REFUTES",
        justification="Test",
        evidence=[
            {'page': 'Valid', 'line': 1},
            {'invalid': 'item'},  # Missing required fields
            {'page': 'Valid2', 'line': 'not_a_number'},  # Invalid line
        ]
    )
    assert len(result['evidence']) == 1  # Only first valid item
```

### Phase 7: Integration Test

**New test** in existing test file or new integration test:

```python
def test_fever_scorer_with_proper_citations():
    """Integration test: ensure recall > 0 when citations match gold."""
    from paper_experiments.case_study_fever.metrics import _call_official_fever_scorer

    # Simulate matched citation
    gold_evidence = [
        [None, None, 'Barack_Obama', 1],
        [None, None, 'Hawaii', 5]
    ]
    predicted_evidence = [
        ['Barack_Obama', 1],  # Exact match
        ['Hawaii', 5]  # Exact match
    ]

    scores = _call_official_fever_scorer(
        predicted_label='SUPPORTS',
        predicted_evidence=predicted_evidence,
        expected_label='SUPPORTS',
        gold_evidence=gold_evidence,
        fail_on_error=False
    )

    if scores is not None:  # Only if fever-scorer package is installed
        assert scores['recall'] == 1.0, "Should have perfect recall with exact matches"
        assert scores['precision'] == 1.0, "Should have perfect precision with exact matches"
        assert scores['fever_score'] == 1.0, "Should have perfect FEVER score"
```

## Rollout Plan

### Step 1: Basic Implementation (2-3 hours)
1. Add the citation extraction functions (Phase 2)
2. Modify the prompt (Phase 1)
3. Update pipeline return (Phase 3)
4. Test manually with a few examples

### Step 2: Validation (1 hour)
1. Add output normalization (Phase 5)
2. Run tests to ensure parsing works
3. Check that evidence format matches scorer expectations

### Step 3: Enhanced Reliability (2 hours, if needed)
1. If basic parsing is unreliable, implement Phase 4 (function calling)
2. This provides more structured output from GPT-4/GPT-3.5

### Step 4: Testing & Verification (1 hour)
1. Add unit tests (Phase 6)
2. Add integration test (Phase 7)
3. Run full FEVER evaluation to verify recall > 0

## Success Metrics

- [ ] Evidence recall > 0% on examples with gold evidence
- [ ] Evidence precision reasonable (>50% when evidence available)
- [ ] LLM consistently provides CITED_EVIDENCE in response
- [ ] Parser correctly extracts [page, line] pairs
- [ ] No crashes on malformed responses (graceful fallback)
- [ ] Tests pass for known gold evidence examples

## Monitoring & Tuning

After implementation, capture metrics for:
- How often LLM provides citations (% of responses with CITED_EVIDENCE)
- Average number of citations per verdict
- Correlation between cited evidence and verdict accuracy
- Parse failure rate (when extraction fails)

Use these metrics to refine:
- Prompt instructions for better citation compliance
- Parser robustness for various response formats
- Temperature settings for more consistent structured output

## Notes

- The structured prompt approach (Phase 1-3) should work for most models
- Function calling (Phase 4) is more reliable but only works with OpenAI models
- Consider A/B testing both approaches to see which gives better recall
- May need to adjust `max_tokens` if reasoning becomes too verbose