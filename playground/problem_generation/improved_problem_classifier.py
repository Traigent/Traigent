"""
Improved Problem Type Classifier for Traigent SDK.

This module provides an enhanced problem classifier that combines:
1. LLM-based classification for nuanced understanding (when available)
2. Keyword and pattern-based fallback for reliability
3. Context-aware classification with problem domain understanding
4. Better handling of ambiguous descriptions
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Try to import Claude Code SDK
try:
    from claude_code_sdk import ClaudeCodeOptions, query

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    print(
        "Warning: Claude Code SDK not available. Using keyword-based classification only."
    )

    # Define mock ClaudeCodeOptions class
    class ClaudeCodeOptions:
        def __init__(self, system_prompt=None, permission_mode=None, **kwargs):
            self.system_prompt = system_prompt
            self.permission_mode = permission_mode
            for k, v in kwargs.items():
                setattr(self, k, v)

    async def query(prompt: str, options=None):
        """Mock implementation for when Claude Code SDK is not available."""
        yield f"Mock response for: {prompt[:100]}..."


@dataclass
class ClassificationResult:
    """Result of problem type classification."""

    problem_type: str
    confidence: float
    reasoning: str
    alternative_types: List[Tuple[str, float]]  # (type, confidence) pairs
    detected_keywords: List[str]
    suggested_metrics: List[str]
    classification_method: str = "hybrid"  # "llm", "keyword", or "hybrid"

    def to_dict(self) -> Dict:
        """Convert to dictionary format."""
        return {
            "problem_type": self.problem_type,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "alternative_types": [
                {"type": t, "confidence": c} for t, c in self.alternative_types
            ],
            "detected_keywords": self.detected_keywords,
            "suggested_metrics": self.suggested_metrics,
            "classification_method": self.classification_method,
        }


# Problem type list for validation
VALID_PROBLEM_TYPES = [
    "classification",
    "generation",
    "information_extraction",
    "question_answering",
    "summarization",
    "ranking_retrieval",
    "translation_transformation",
    "code_generation",
    "reasoning",
]


class ImprovedProblemClassifier:
    """
    Enhanced problem classifier that combines LLM and keyword-based approaches.

    Features:
    - LLM-based classification for nuanced understanding
    - Keyword/pattern fallback for reliability
    - Context-aware classification
    - Confidence scoring based on multiple factors
    - Detailed reasoning for transparency
    """

    # Enhanced problem type definitions with more context
    PROBLEM_TYPES = {
        "classification": {
            "description": "Categorizing input text into predefined classes or labels",
            "keywords": [
                "classify",
                "categorize",
                "label",
                "tag",
                "sort",
                "identify",
                "sentiment",
                "intent",
                "moderate",
                "filter",
                "detect",
                "determine",
            ],
            "patterns": [
                r"classify\s+\w+\s+(?:into|as)",
                r"categorize\s+\w+\s+(?:as|into)",
                r"determine\s+(?:if|whether)\s+\w+\s+is",
                r"label\s+\w+\s+(?:with|as)",
                r"detect\s+(?:spam|fraud|anomal|toxic|offensive)",
                r"identify\s+(?:the\s+)?(?:type|category|class)",
                r"sentiment\s+(?:analysis|classification)",
                r"intent\s+(?:classification|detection)",
                r"spam\s+(?:detection|filtering)",
                r"is\s+(?:this|it)\s+\w+",
            ],
            "example_phrases": [
                "classify emails as spam or not spam",
                "categorize customer feedback by sentiment",
                "detect toxic comments",
                "identify document types",
            ],
            "metrics": [
                "classification",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "confusion_matrix",
                "roc_auc",
                "precision_at_k",
            ],
            "typical_outputs": [
                "class label",
                "category",
                "binary decision",
                "multi-class label",
            ],
        },
        "generation": {
            "description": "Creating new text based on prompts, context, or requirements",
            "keywords": [
                "generate",
                "create",
                "write",
                "compose",
                "produce",
                "draft",
                "synthesize",
                "formulate",
                "construct",
                "develop",
                "design",
                "provide",
                "guidance",
                "advice",
                "tips",
                "recommendations",
            ],
            "patterns": [
                r"generate\s+(?:a\s+)?(?:new\s+)?\w+",
                r"create\s+(?:a\s+)?(?:new\s+)?\w+",
                r"write\s+(?:a|an)\s+\w+",
                r"compose\s+\w+\s+(?:for|about)",
                r"produce\s+\w+\s+content",
                r"draft\s+(?:a|an)\s+\w+",
                r"come\s+up\s+with\s+\w+",
                r"help\s+me\s+(?:write|create|generate)",
                r"need\s+(?:a|an)\s+\w+\s+(?:for|about)",
                r"provide\s+(?:guidance|advice|tips|recommendations)",
                r"give\s+(?:me\s+)?(?:guidance|advice|tips)",
            ],
            "example_phrases": [
                "generate product descriptions",
                "create marketing copy",
                "write blog posts",
                "compose email responses",
                "draft technical documentation",
            ],
            "metrics": [
                "bleu_score",
                "rouge_score",
                "perplexity",
                "coherence",
                "relevance",
                "fluency",
                "diversity",
            ],
            "typical_outputs": [
                "generated text",
                "creative content",
                "structured document",
            ],
        },
        "information_extraction": {
            "description": "Extracting specific structured information from unstructured text",
            "keywords": [
                "extract",
                "parse",
                "find",
                "locate",
                "identify",
                "detect",
                "recognize",
                "pull",
                "get",
                "retrieve",
                "mine",
            ],
            "patterns": [
                r"extract\s+(?:all\s+)?\w+\s+from",
                r"parse\s+\w+\s+(?:from\s+)?(?:document|text)",
                r"find\s+(?:all\s+)?(?:the\s+)?\w+\s+in",
                r"identify\s+\w+\s+(?:entities|information)",
                r"get\s+(?:all\s+)?(?:the\s+)?\w+\s+from",
                r"pull\s+(?:out\s+)?\w+\s+from",
                r"named\s+entity\s+(?:recognition|extraction)",
                r"information\s+extraction",
                r"data\s+(?:extraction|mining)",
                r"detect\s+\w+\s+in\s+(?:unstructured\s+)?(?:text|documents?|data)",
                r"locate\s+\w+\s+in\s+(?:text|document|data)",
            ],
            "example_phrases": [
                "extract names and dates from documents",
                "parse contact information",
                "find all phone numbers in text",
                "identify key entities and relationships",
            ],
            "metrics": [
                "extraction_precision",
                "extraction_recall",
                "extraction_f1",
                "exact_match",
                "partial_match",
                "entity_accuracy",
            ],
            "typical_outputs": ["structured data", "entity list", "key-value pairs"],
        },
        "question_answering": {
            "description": "Providing accurate answers to specific questions based on context",
            "keywords": [
                "answer",
                "respond",
                "reply",
                "explain",
                "clarify",
                "solve",
                "address",
                "tell",
                "describe",
                "define",
                "question",
                "inquiry",
                "FAQ",
                "Q&A",
                "how",
                "what",
                "why",
                "when",
                "where",
                "who",
                "which",
            ],
            "patterns": [
                r"answer\s+(?:the\s+)?(?:following\s+)?questions?",
                r"respond\s+to\s+(?:user|customer)?\s*questions",
                r"(?:can\s+you\s+)?(?:please\s+)?(?:answer|tell|explain)",
                r"what\s+(?:is|are|was|were|causes?|makes?|happens?)",
                r"(?:how|why|when|where|who|which)\s+(?:does?|is|are|can|should|would|will)",
                r"what\s+\w+\s+\w+",
                r"(?:build|create)\s+(?:a|an)\s+Q&A\s+system",
                r"FAQ\s+(?:bot|system|generator)",
                r"question\s+answering\s+(?:system|model)",
            ],
            "example_phrases": [
                "answer questions about products",
                "build a Q&A system",
                "respond to customer inquiries",
                "create an FAQ bot",
            ],
            "metrics": [
                "exact_match",
                "f1_score",
                "answer_relevance",
                "answer_accuracy",
                "factual_accuracy",
                "semantic_similarity",
            ],
            "typical_outputs": ["direct answer", "explanation", "factual response"],
        },
        "summarization": {
            "description": "Condensing longer texts while preserving key information",
            "keywords": [
                "summarize",
                "condense",
                "brief",
                "abstract",
                "digest",
                "synopsis",
                "outline",
                "recap",
                "compress",
                "shorten",
            ],
            "patterns": [
                r"summarize\s+(?:the\s+)?(?:following\s+)?\w+",
                r"(?:create|generate)\s+(?:a\s+)?(?:summary|summaries|abstract|abstracts|synopsis|overview)",
                r"(?:create|generate)\s+(?:executive\s+)?summaries",
                r"condense\s+\w+\s+(?:into|to)",
                r"(?:provide|give)\s+(?:a\s+)?(?:brief|short)\s+(?:summary|overview)",
                r"abstract\s+(?:of|for)\s+\w+",
                r"key\s+(?:points|takeaways)\s+(?:from|of)",
                r"tldr|tl;dr",
                r"main\s+(?:points|ideas)",
                r"brief\s+overview",
                r"synopsis\s+of",
            ],
            "example_phrases": [
                "summarize this article",
                "create executive summaries",
                "condense meeting notes",
                "key takeaways from the document",
            ],
            "metrics": [
                "rouge_1",
                "rouge_2",
                "rouge_l",
                "bleu_score",
                "information_retention",
                "compression_ratio",
                "readability",
            ],
            "typical_outputs": ["summary text", "bullet points", "abstract"],
        },
        "ranking_retrieval": {
            "description": "Ordering items by relevance or retrieving most relevant items",
            "keywords": [
                "rank",
                "search",
                "retrieve",
                "find",
                "match",
                "recommend",
                "suggest",
                "order",
                "sort",
                "prioritize",
                "score",
            ],
            "patterns": [
                r"rank\s+\w+\s+by\s+(?:relevance|importance)",
                r"search\s+for\s+(?:similar|relevant)",
                r"retrieve\s+(?:most\s+)?relevant",
                r"find\s+(?:best|top)\s+(?:matches|results)",
                r"recommend\s+\w+\s+based\s+on",
                r"sort\s+by\s+(?:relevance|similarity)",
                r"most\s+(?:relevant|similar)\s+\w+",
                r"personalized\s+recommendations",
            ],
            "example_phrases": [
                "rank search results by relevance",
                "find similar documents",
                "recommend products to users",
                "retrieve most relevant articles",
            ],
            "metrics": [
                "ndcg",
                "map",
                "mrr",
                "precision_at_k",
                "recall_at_k",
                "hit_rate",
                "coverage",
                "diversity",
            ],
            "typical_outputs": ["ranked list", "relevance scores", "recommendations"],
        },
        "translation_transformation": {
            "description": "Converting text from one form, style, or language to another",
            "keywords": [
                "translate",
                "transform",
                "convert",
                "rewrite",
                "paraphrase",
                "adapt",
                "reformat",
                "rephrase",
                "modify",
                "change",
            ],
            "patterns": [
                r"translate\s+(?:from\s+)?\w+\s+(?:to|into)",
                r"transform\s+\w+\s+(?:style|format|tone)",
                r"convert\s+\w+\s+(?:to|into)",
                r"rewrite\s+\w+\s+(?:in|as|for)",
                r"change\s+(?:the\s+)?(?:tone|style|format)",
                r"paraphrase\s+(?:the\s+)?(?:following|this)",
                r"make\s+(?:it|this)\s+more\s+\w+",
                r"simplify\s+(?:the\s+)?(?:language|text)",
            ],
            "example_phrases": [
                "translate to Spanish",
                "convert technical to simple language",
                "transform casual to formal tone",
                "paraphrase this paragraph",
            ],
            "metrics": [
                "bleu_score",
                "semantic_similarity",
                "style_preservation",
                "fluency",
                "adequacy",
                "edit_distance",
            ],
            "typical_outputs": ["transformed text", "translation", "paraphrase"],
        },
        "code_generation": {
            "description": "Generating executable code or converting natural language to code",
            "keywords": [
                "code",
                "program",
                "script",
                "function",
                "implement",
                "algorithm",
                "sql",
                "query",
                "develop",
                "automate",
            ],
            "patterns": [
                r"(?:generate|write|create)\s+(?:code|program|script)",
                r"implement\s+(?:a|an)?\s*\w+",
                r"(?:create|write)\s+(?:a|an)?\s*function",
                r"code\s+(?:for|to)\s+\w+",
                r"(?:sql|database)\s+query",
                r"text\s*(?:to|2)\s*(?:code|sql)",
                r"natural\s+language\s+to\s+(?:code|sql)",
                r"automate\s+\w+\s+(?:with|using)\s+(?:code|script)",
                r"(?:python|javascript|java)\s+(?:code|script|function)",
            ],
            "example_phrases": [
                "generate Python code to sort a list",
                "create SQL query for user data",
                "implement binary search algorithm",
                "convert this logic to code",
            ],
            "metrics": [
                "exact_match",
                "execution_match",
                "syntax_validity",
                "functional_correctness",
                "code_quality",
                "efficiency",
            ],
            "typical_outputs": ["executable code", "SQL query", "script", "function"],
        },
        "reasoning": {
            "description": "Solving problems requiring logical, mathematical, or analytical thinking",
            "keywords": [
                "solve",
                "calculate",
                "reason",
                "deduce",
                "infer",
                "analyze",
                "prove",
                "derive",
                "compute",
                "figure out",
                "problem",
                "solution",
                "logic",
                "mathematical",
            ],
            "patterns": [
                r"solve\s+(?:this\s+)?\w+\s+problem",
                r"calculate\s+\w+\s+(?:based\s+on|from)",
                r"reason\s+(?:about|through)",
                r"(?:mathematical|logical)\s+(?:reasoning|problem)",
                r"step-by-step\s+(?:solution|reasoning|problem\s+solving)",
                r"problem\s+solving",
                r"figure\s+out\s+(?:how|why|what)",
                r"analyze\s+(?:and\s+)?(?:solve|explain)",
                r"work\s+(?:out|through)\s+(?:this|the)\s+problem",
            ],
            "example_phrases": [
                "solve this math problem",
                "reason through this logic puzzle",
                "calculate the optimal solution",
                "analyze and solve this case",
            ],
            "metrics": [
                "answer_accuracy",
                "reasoning_validity",
                "step_correctness",
                "logic_consistency",
                "solution_efficiency",
                "explanation_quality",
            ],
            "typical_outputs": [
                "solution",
                "proof",
                "calculation result",
                "reasoning steps",
            ],
        },
    }

    def __init__(self, use_llm: bool = True, model_name: str = "claude-sonnet-4-0"):
        """
        Initialize the improved problem classifier.

        Args:
            use_llm: Whether to use LLM-based classification when available
            model_name: The LLM model to use for classification
        """
        self.use_llm = use_llm and CLAUDE_SDK_AVAILABLE
        self._compile_patterns()

        if self.use_llm:
            print("Claude Code SDK-based classification enabled")
        else:
            print("Using keyword-based classification")

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching."""
        for _problem_type, config in self.PROBLEM_TYPES.items():
            config["compiled_patterns"] = [
                re.compile(pattern, re.IGNORECASE) for pattern in config["patterns"]
            ]

    def classify(self, description: str) -> ClassificationResult:
        """
        Classify a problem description using hybrid approach.

        Args:
            description: Natural language description of the problem

        Returns:
            ClassificationResult with problem type, confidence, and reasoning
        """
        # Try LLM-based classification first if enabled
        if self.use_llm:
            try:
                # Run async method in sync context
                import asyncio

                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is already running, create a task
                        # This happens in Jupyter notebooks or async contexts
                        import concurrent.futures

                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run, self._classify_with_llm(description)
                            )
                            llm_result = future.result()
                    else:
                        # Normal sync context
                        llm_result = loop.run_until_complete(
                            self._classify_with_llm(description)
                        )
                except RuntimeError:
                    # No event loop, create one
                    llm_result = asyncio.run(self._classify_with_llm(description))

                if llm_result.confidence >= 0.7:  # High confidence LLM result
                    return llm_result

                # For lower confidence, combine with keyword analysis
                keyword_result = self._classify_with_keywords(description)
                return self._combine_results(llm_result, keyword_result)

            except Exception as e:
                print(
                    f"LLM classification failed: {e}. Using keyword-based classification."
                )

        # Fallback to keyword-based classification
        return self._classify_with_keywords(description)

    async def _classify_with_llm(self, description: str) -> ClassificationResult:
        """Classify using Claude Code SDK."""
        # Format problem types for prompt
        problem_types_text = self._format_problem_types_for_prompt()

        # Create prompt for Claude
        prompt = f"""You are an expert at classifying natural language processing and machine learning problems.

Given a problem description, classify it into one of these problem types:
{problem_types_text}

Problem Description: {description}

Analyze this description carefully and provide a JSON response with:
1. "problem_type": The most appropriate problem type from the list above
2. "confidence": Your confidence level (0.0 to 1.0)
3. "reasoning": Detailed reasoning explaining your classification
4. "alternative_types": Up to 3 alternative problem types that could also apply
5. "key_indicators": Key phrases/keywords that led to your classification

The problem type MUST be one of: {", ".join(VALID_PROBLEM_TYPES)}

Respond with valid JSON only."""

        try:
            messages = []
            async for message in query(
                prompt=prompt,
                options=ClaudeCodeOptions(
                    system_prompt="You are an expert at problem classification. Always respond with valid JSON containing the requested fields.",
                    permission_mode="bypassPermissions",
                    max_turns=1,  # We only need one response
                ),
            ):
                messages.append(message)

            # Extract text from assistant messages
            response_text = ""
            for msg in messages:
                # Check if this is an assistant message with content
                if hasattr(msg, "type") and msg.type == "assistant":
                    if hasattr(msg, "message") and hasattr(msg.message, "content"):
                        # The content is likely a list of content blocks
                        content = msg.message.content
                        if isinstance(content, str):
                            response_text += content
                        elif isinstance(content, list):
                            for block in content:
                                if hasattr(block, "type") and block.type == "text":
                                    if hasattr(block, "text"):
                                        response_text += block.text
                                elif isinstance(block, str):
                                    response_text += block

            # Ensure we have a response
            if not response_text:
                print("Warning: Empty response from Claude SDK")
                return self._classify_with_keywords(description)

            # Parse JSON response
            import json

            result = json.loads(response_text)

            # Validate and extract fields
            problem_type = result.get("problem_type", "classification")
            if problem_type not in VALID_PROBLEM_TYPES:
                problem_type = "classification"  # Default fallback

            confidence = float(result.get("confidence", 0.7))
            reasoning = result.get(
                "reasoning", "Classified based on description analysis"
            )
            alternative_types_raw = result.get("alternative_types", [])
            key_indicators = result.get("key_indicators", [])

            # Convert alternative types to tuples with confidence
            alternatives = [
                (alt_type, 0.7 - 0.1 * i)  # Decreasing confidence for alternatives
                for i, alt_type in enumerate(alternative_types_raw)
                if alt_type in VALID_PROBLEM_TYPES
            ][:3]

            # Get problem type info
            problem_type_info = self.PROBLEM_TYPES.get(problem_type, {})

            return ClassificationResult(
                problem_type=problem_type,
                confidence=confidence,
                reasoning=reasoning,
                alternative_types=alternatives,
                detected_keywords=key_indicators,
                suggested_metrics=problem_type_info.get("metrics", []),
                classification_method="llm",
            )

        except Exception as e:
            print(f"Claude SDK classification error: {e}")
            # Return a fallback result
            return self._classify_with_keywords(description)

    def _classify_with_keywords(self, description: str) -> ClassificationResult:
        """Classify using keyword and pattern matching."""
        description_lower = description.lower()

        # Special case handling
        special_result = self._handle_special_cases(description_lower)
        if special_result:
            return special_result

        # Score each problem type
        scores = {}
        detected_keywords = {}
        pattern_matches = {}

        for problem_type, config in self.PROBLEM_TYPES.items():
            score, keywords, patterns = self._score_problem_type(
                description_lower, description, config
            )
            scores[problem_type] = score
            detected_keywords[problem_type] = keywords
            pattern_matches[problem_type] = patterns

        # Get best match and alternatives
        sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_type, best_score = sorted_types[0]

        # Calculate confidence based on score distribution
        confidence = self._calculate_confidence(best_score, sorted_types)

        # Get alternatives with significant scores
        alternatives = [
            (ptype, score)
            for ptype, score in sorted_types[1:]
            if score > 0.1  # Only include if score > 10% (more generous)
        ][:3]

        # Generate reasoning
        reasoning = self._generate_keyword_reasoning(
            description,
            best_type,
            best_score,
            detected_keywords[best_type],
            pattern_matches[best_type],
        )

        return ClassificationResult(
            problem_type=best_type,
            confidence=confidence,
            reasoning=reasoning,
            alternative_types=alternatives,
            detected_keywords=detected_keywords[best_type],
            suggested_metrics=self.PROBLEM_TYPES[best_type]["metrics"],
            classification_method="keyword",
        )

    def _handle_special_cases(
        self, description_lower: str
    ) -> Optional[ClassificationResult]:
        """Handle special cases with high confidence."""
        special_cases = [
            # Text to SQL variations
            (
                [
                    "text to sql",
                    "text2sql",
                    "text-to-sql",
                    "nl to sql",
                    "nl2sql",
                    "natural language to sql",
                    "english to sql",
                    "convert natural language queries into sql",
                ],
                "code_generation",
                "Text-to-SQL is a specialized code generation task",
            ),
            # How-to guides
            (
                ["how to", "guide for", "steps to", "tutorial on"],
                "generation",
                "Creating guides and tutorials is a content generation task",
            ),
            # Direct questions
            (
                [
                    "what is",
                    "what are",
                    "why is",
                    "why are",
                    "when is",
                    "when are",
                    "where is",
                    "where are",
                    "who is",
                    "who are",
                    "how does",
                    "how do",
                ],
                "question_answering",
                "Direct questions indicate a question answering task",
            ),
            # Sentiment analysis
            (
                ["sentiment analysis", "emotion detection", "opinion mining"],
                "classification",
                "Sentiment analysis is a classification task",
            ),
            # Named entity recognition
            (
                ["named entity recognition", "ner", "entity extraction"],
                "information_extraction",
                "NER is an information extraction task",
            ),
        ]

        for patterns, problem_type, reasoning in special_cases:
            matched_patterns = []
            for pattern in patterns:
                # For short patterns like "ner", ensure word boundaries
                if len(pattern) <= 3:
                    # Check for exact word match
                    import re

                    if re.search(r"\b" + re.escape(pattern) + r"\b", description_lower):
                        matched_patterns.append(pattern)
                else:
                    # For longer patterns, simple substring match is fine
                    if pattern in description_lower:
                        matched_patterns.append(pattern)

            if matched_patterns:
                return ClassificationResult(
                    problem_type=problem_type,
                    confidence=0.95,
                    reasoning=reasoning,
                    alternative_types=[],
                    detected_keywords=matched_patterns,
                    suggested_metrics=self.PROBLEM_TYPES[problem_type]["metrics"],
                    classification_method="keyword",
                )

        return None

    def _score_problem_type(
        self, description_lower: str, description: str, config: Dict
    ) -> Tuple[float, List[str], List[str]]:
        """Score a problem type based on keywords and patterns."""
        score = 0.0
        detected_keywords = []
        matched_patterns = []

        # Keyword matching (weight: 0.4)
        keyword_matches = 0
        specific_keywords = []
        general_keywords = []

        # Categorize keywords by specificity
        very_general = {
            "generate",
            "create",
            "write",
            "make",
            "build",
            "develop",
            "produce",
        }

        # Split description into words for word-boundary matching
        desc_words = description_lower.split()

        for keyword in config["keywords"]:
            # Check for whole word match or as part of compound words
            if keyword in desc_words or any(
                word.startswith(keyword + "s")
                or word.startswith(keyword + "ing")
                or word.startswith(keyword + "ed")
                for word in desc_words
            ):
                keyword_matches += 1
                detected_keywords.append(keyword)

                if keyword in very_general:
                    general_keywords.append(keyword)
                else:
                    specific_keywords.append(keyword)

        if keyword_matches > 0:
            # Weight specific keywords more heavily
            specific_weight = len(specific_keywords) * 1.0
            general_weight = (
                len(general_keywords) * 0.3
            )  # Reduce weight of general keywords

            total_weight = specific_weight + general_weight
            score += 0.4 * min(total_weight / 3, 1.0)

        # Pattern matching (weight: 0.5)
        pattern_matches = 0
        for pattern in config["compiled_patterns"]:
            if pattern.search(description):
                pattern_matches += 1
                matched_patterns.append(pattern.pattern)

        if pattern_matches > 0:
            score += 0.5 * min(pattern_matches / 2, 1.0)

        # Example phrase similarity (weight: 0.1)
        phrase_similarity = 0
        for phrase in config["example_phrases"]:
            # Check for partial matches
            phrase_words = set(phrase.lower().split())
            desc_words = set(description_lower.split())
            overlap = len(phrase_words & desc_words) / len(phrase_words)
            phrase_similarity = max(phrase_similarity, overlap)

        score += 0.1 * phrase_similarity

        return score, detected_keywords, matched_patterns

    def _calculate_confidence(
        self, best_score: float, sorted_types: List[Tuple[str, float]]
    ) -> float:
        """Calculate confidence based on score distribution."""
        if best_score == 0:
            return 0.0

        if len(sorted_types) == 1:
            return min(best_score * 2.5, 0.95)  # Boost single matches

        # Calculate margin between best and second-best
        second_score = sorted_types[1][1] if len(sorted_types) > 1 else 0
        margin = best_score - second_score

        # More generous base confidence calculation
        base_confidence = min(best_score * 3.0, 0.9)  # Triple the raw score
        margin_boost = min(margin * 4, 0.2)  # Up to 20% boost for clear winner

        # Ensure minimum confidence for reasonable matches
        final_confidence = max(base_confidence + margin_boost, best_score * 2.5)

        return min(final_confidence, 0.95)

    def _generate_keyword_reasoning(
        self,
        description: str,
        problem_type: str,
        score: float,
        keywords: List[str],
        patterns: List[str],
    ) -> str:
        """Generate reasoning for keyword-based classification."""
        type_info = self.PROBLEM_TYPES[problem_type]
        reasoning_parts = []

        # Describe what was detected
        if keywords:
            reasoning_parts.append(f"Detected keywords: {', '.join(keywords[:5])}")

        if patterns:
            reasoning_parts.append(f"Matched patterns for {problem_type} tasks")

        # Add problem type description
        reasoning_parts.append(
            f"This appears to be a {problem_type} problem: {type_info['description']}"
        )

        # Add confidence interpretation
        if score > 0.8:
            reasoning_parts.append(
                "High confidence based on multiple strong indicators."
            )
        elif score > 0.5:
            reasoning_parts.append("Moderate confidence based on clear indicators.")
        else:
            reasoning_parts.append(
                "Lower confidence - consider reviewing alternatives."
            )

        return " ".join(reasoning_parts)

    def _combine_results(
        self, llm_result: ClassificationResult, keyword_result: ClassificationResult
    ) -> ClassificationResult:
        """Combine LLM and keyword results for hybrid classification."""
        # If they agree, boost confidence
        if llm_result.problem_type == keyword_result.problem_type:
            combined_confidence = min(
                (llm_result.confidence + keyword_result.confidence) / 2 + 0.1, 0.99
            )

            return ClassificationResult(
                problem_type=llm_result.problem_type,
                confidence=combined_confidence,
                reasoning=f"{llm_result.reasoning} [Confirmed by keyword analysis]",
                alternative_types=llm_result.alternative_types,
                detected_keywords=list(
                    set(llm_result.detected_keywords + keyword_result.detected_keywords)
                ),
                suggested_metrics=llm_result.suggested_metrics,
                classification_method="hybrid",
            )

        # If they disagree, use weighted average
        if llm_result.confidence > keyword_result.confidence:
            primary = llm_result
            secondary = keyword_result
        else:
            primary = keyword_result
            secondary = llm_result

        # Add the other result as an alternative
        alternatives = primary.alternative_types.copy()
        if secondary.problem_type not in [alt[0] for alt in alternatives]:
            alternatives.insert(0, (secondary.problem_type, secondary.confidence))

        return ClassificationResult(
            problem_type=primary.problem_type,
            confidence=primary.confidence
            * 0.8,  # Reduce confidence due to disagreement
            reasoning=f"{primary.reasoning} [Alternative view: {secondary.problem_type}]",
            alternative_types=alternatives[:3],
            detected_keywords=list(
                set(primary.detected_keywords + secondary.detected_keywords)
            ),
            suggested_metrics=primary.suggested_metrics,
            classification_method="hybrid",
        )

    def _format_problem_types_for_prompt(self) -> str:
        """Format problem types for LLM prompt."""
        lines = []
        for ptype, info in self.PROBLEM_TYPES.items():
            lines.append(f"\n{ptype.upper()}:")
            lines.append(f"  Description: {info['description']}")
            lines.append(f"  Keywords: {', '.join(info['keywords'][:10])}")
            lines.append(f"  Examples: {', '.join(info['example_phrases'][:3])}")
            lines.append(f"  Typical outputs: {', '.join(info['typical_outputs'])}")

        return "\n".join(lines)

    def get_problem_type_info(self, problem_type: str) -> Dict:
        """Get detailed information about a problem type."""
        return self.PROBLEM_TYPES.get(problem_type, {})

    def get_all_problem_types(self) -> List[str]:
        """Get list of all available problem types."""
        return list(self.PROBLEM_TYPES.keys())

    def get_metrics_for_type(self, problem_type: str) -> List[str]:
        """Get suggested metrics for a problem type."""
        return self.PROBLEM_TYPES.get(problem_type, {}).get("metrics", [])

    def validate_classification(
        self, description: str, expected_type: str
    ) -> Tuple[bool, str]:
        """
        Validate if a description matches the expected problem type.

        Args:
            description: Problem description
            expected_type: Expected problem type

        Returns:
            Tuple of (is_valid, explanation)
        """
        result = self.classify(description)

        if result.problem_type == expected_type:
            return (
                True,
                f"Correctly classified as {expected_type} with {result.confidence:.2f} confidence",
            )

        # Check if expected type is in alternatives
        alt_types = [alt[0] for alt in result.alternative_types]
        if expected_type in alt_types:
            alt_conf = next(
                conf for typ, conf in result.alternative_types if typ == expected_type
            )
            return (
                True,
                f"Expected type {expected_type} is a valid alternative with {alt_conf:.2f} confidence",
            )

        return (
            False,
            f"Classified as {result.problem_type}, not {expected_type}. Reasoning: {result.reasoning}",
        )


def compare_classifiers(description: str):
    """
    Compare the old and new classifier results for debugging.

    Args:
        description: Problem description to classify
    """
    print(f"\n{'=' * 60}")
    print(f"Description: {description}")
    print(f"{'=' * 60}")

    # Try old classifier
    try:
        from problem_classifier_old import ProblemTypeClassifier as OldClassifier

        old_classifier = OldClassifier()
        old_result = old_classifier.classify(description)

        print("\nOLD CLASSIFIER:")
        print(f"  Type: {old_result.problem_type}")
        print(f"  Confidence: {old_result.confidence:.2f}")
        print(f"  Reasoning: {old_result.reasoning}")
        print(f"  Keywords: {', '.join(old_result.detected_keywords[:5])}")
    except Exception as e:
        print(f"\nOLD CLASSIFIER: Failed - {e}")

    # Try new classifier
    try:
        new_classifier = ImprovedProblemClassifier(
            use_llm=False
        )  # Test keyword-only first
        new_result = new_classifier.classify(description)

        print("\nIMPROVED CLASSIFIER (Keyword):")
        print(f"  Type: {new_result.problem_type}")
        print(f"  Confidence: {new_result.confidence:.2f}")
        print(f"  Reasoning: {new_result.reasoning}")
        print(f"  Method: {new_result.classification_method}")
    except Exception as e:
        print(f"\nIMPROVED CLASSIFIER: Failed - {e}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    # Test the improved classifier
    test_descriptions = [
        "I need to classify customer emails into different categories like complaints, questions, and feedback",
        "Generate a blog post about the benefits of machine learning in healthcare",
        "Extract all the names, dates, and locations mentioned in these legal documents",
        "Build a system that can answer questions about our product documentation",
        "Summarize this 50-page research paper into a 2-page executive summary",
        "Find the most relevant documents in our knowledge base for a given query",
        "Translate this technical manual from English to Spanish while maintaining formatting",
        "Write Python code that implements a binary search algorithm",
        "Solve this optimization problem to minimize costs while maximizing efficiency",
        "How to lose weight effectively and safely",
        "Convert natural language queries into SQL database queries",
        "What are the main causes of climate change?",
        "Create a sentiment analysis model for movie reviews",
        "I want to build a recommendation system for our e-commerce platform",
    ]

    # Initialize classifier
    classifier = ImprovedProblemClassifier(use_llm=False)  # Test without LLM dependency

    print("Testing Improved Problem Classifier")
    print("=" * 80)

    for desc in test_descriptions:
        result = classifier.classify(desc)
        print(f"\nDescription: {desc}")
        print(f"Classification: {result.problem_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Method: {result.classification_method}")
        if result.alternative_types:
            alts = [f"{t}({c:.2f})" for t, c in result.alternative_types]
            print(f"Alternatives: {', '.join(alts)}")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 80)
