"""
Problem Diversity Manager for Gap Analysis and Problem Planning.

This module analyzes existing problems to identify gaps and suggest new problems
for comprehensive coverage across domains, types, and difficulties.
"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from traigent.utils.secure_path import safe_write_text, validate_path

@dataclass
class ProblemProfile:
    """Profile of an existing problem."""

    name: str
    domain: str
    problem_type: str
    difficulty_level: str
    example_count: int
    input_structure: Dict[str, str]
    output_type: str
    metrics: List[str]
    technical_features: Set[str]
    coverage_gaps: List[str] = field(default_factory=list)


@dataclass
class ProblemOpportunity:
    """Opportunity for a new problem to fill gaps."""

    suggested_name: str
    domain: str
    problem_type: str
    difficulty_level: str
    rationale: str
    unique_features: List[str]
    target_capabilities: List[str]
    estimated_impact: str  # high, medium, low


@dataclass
class ProblemSpecification:
    """Complete specification for a new problem."""

    name: str
    description: str
    domain: str
    problem_type: str
    difficulty_level: str
    example_count: int
    input_structure: Dict[str, str]
    output_structure: Dict[str, str]
    evaluation_metrics: List[str]
    example_patterns: List[str]
    technical_requirements: List[str]


class ProblemDiversityManager:
    """
    Manages diversity across problems to ensure comprehensive coverage.

    Analyzes existing problems and suggests new ones to fill gaps in:
    - Domain coverage
    - Problem type variety
    - Difficulty distribution
    - Technical capability testing
    """

    def __init__(self, problems_dir: str = "examples/langchain_problems"):
        """
        Initialize diversity manager.

        Args:
            problems_dir: Directory containing existing problems
        """
        self._base_dir = Path.cwd()
        self.problems_dir = validate_path(problems_dir, self._base_dir)
        self.existing_problems: Dict[str, ProblemProfile] = {}

        # Define comprehensive taxonomies
        self.domains = {
            "technical": ["software", "api", "debugging", "architecture", "security"],
            "business": ["finance", "marketing", "strategy", "operations", "analytics"],
            "medical": [
                "diagnosis",
                "treatment",
                "research",
                "patient_care",
                "pharmacy",
            ],
            "legal": [
                "contracts",
                "compliance",
                "litigation",
                "intellectual_property",
                "regulatory",
            ],
            "educational": [
                "curriculum",
                "assessment",
                "tutoring",
                "research",
                "administration",
            ],
            "creative": [
                "writing",
                "design",
                "music",
                "storytelling",
                "content_creation",
            ],
            "scientific": [
                "research",
                "experimentation",
                "analysis",
                "theory",
                "publication",
            ],
            "customer_service": [
                "support",
                "satisfaction",
                "retention",
                "feedback",
                "escalation",
            ],
        }

        self.problem_types = {
            "classification": ["binary", "multi_class", "multi_label", "hierarchical"],
            "generation": ["text", "code", "structured", "creative", "technical"],
            "extraction": ["entity", "relation", "event", "attribute", "summary"],
            "question_answering": [
                "factual",
                "reasoning",
                "multi_hop",
                "conversational",
            ],
            "analysis": ["sentiment", "topic", "style", "quality", "comparison"],
            "transformation": [
                "translation",
                "paraphrase",
                "style_transfer",
                "simplification",
            ],
            "completion": ["text", "code", "dialogue", "story", "technical"],
            "reasoning": ["logical", "mathematical", "causal", "temporal", "spatial"],
        }

        self.difficulty_levels = ["easy", "medium", "hard", "very_hard", "expert"]

        self.technical_capabilities = [
            "multi_step_reasoning",
            "structured_output",
            "context_understanding",
            "ambiguity_handling",
            "domain_expertise",
            "creative_generation",
            "factual_accuracy",
            "logical_consistency",
            "temporal_reasoning",
            "spatial_reasoning",
            "mathematical_computation",
            "code_understanding",
            "cross_lingual",
            "multimodal_integration",
        ]

    def analyze_existing_problems(self) -> Dict[str, ProblemProfile]:
        """
        Analyze all existing problems in the problems directory.

        Returns:
            Dictionary mapping problem names to their profiles
        """
        self.existing_problems.clear()

        for problem_file in self.problems_dir.glob("*.py"):
            if problem_file.name.startswith("__"):
                continue

            try:
                profile = self._analyze_problem_file(problem_file)
                if profile:
                    self.existing_problems[profile.name] = profile
            except Exception as e:
                print(f"Error analyzing {problem_file}: {e}")

        return self.existing_problems

    def identify_gaps(self) -> List[ProblemOpportunity]:
        """
        Identify gaps in current problem coverage.

        Returns:
            List of problem opportunities to fill gaps
        """
        gaps = []

        # Analyze domain coverage
        domain_gaps = self._identify_domain_gaps()
        gaps.extend(domain_gaps)

        # Analyze problem type coverage
        type_gaps = self._identify_problem_type_gaps()
        gaps.extend(type_gaps)

        # Analyze difficulty distribution
        difficulty_gaps = self._identify_difficulty_gaps()
        gaps.extend(difficulty_gaps)

        # Analyze technical capability coverage
        capability_gaps = self._identify_capability_gaps()
        gaps.extend(capability_gaps)

        # Sort by estimated impact
        gaps.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}[x.estimated_impact])

        return gaps

    def suggest_new_problems(self, count: int = 30) -> List[ProblemSpecification]:
        """
        Suggest new problems to create comprehensive coverage.

        Args:
            count: Number of problems to suggest

        Returns:
            List of problem specifications
        """
        # Analyze existing problems
        self.analyze_existing_problems()

        # Identify gaps
        opportunities = self.identify_gaps()

        # Convert top opportunities to specifications
        specifications = []

        # Ensure diversity in suggestions
        domains_used = defaultdict(int)
        types_used = defaultdict(int)
        difficulties_used = defaultdict(int)

        for opp in opportunities:
            if len(specifications) >= count:
                break

            # Check diversity constraints
            if domains_used[opp.domain] >= count // 6:  # Max ~5 per domain
                continue
            if types_used[opp.problem_type] >= count // 4:  # Max ~7-8 per type
                continue
            if (
                difficulties_used[opp.difficulty_level] >= count // 3
            ):  # Even distribution
                continue

            spec = self._opportunity_to_specification(opp)
            specifications.append(spec)

            domains_used[opp.domain] += 1
            types_used[opp.problem_type] += 1
            difficulties_used[opp.difficulty_level] += 1

        # Fill remaining slots with balanced problems
        while len(specifications) < count:
            spec = self._generate_balanced_problem(
                specifications, domains_used, types_used, difficulties_used
            )
            specifications.append(spec)

            domains_used[spec.domain] += 1
            types_used[spec.problem_type] += 1
            difficulties_used[spec.difficulty_level] += 1

        return specifications

    def _analyze_problem_file(self, problem_file: Path) -> Optional[ProblemProfile]:
        """Analyze a single problem file."""
        try:
            # Read file content
            safe_path = validate_path(problem_file, self.problems_dir, must_exist=True)
            content = safe_path.read_text()

            # Extract metadata from docstring and code
            name = problem_file.stem
            domain = self._extract_domain(content, name)
            problem_type = self._extract_problem_type(content)
            difficulty = self._extract_difficulty(content)
            example_count = self._count_examples(content)
            input_structure = self._extract_input_structure(content)
            output_type = self._extract_output_type(content)
            metrics = self._extract_metrics(content)
            technical_features = self._extract_technical_features(content)

            return ProblemProfile(
                name=name,
                domain=domain,
                problem_type=problem_type,
                difficulty_level=difficulty,
                example_count=example_count,
                input_structure=input_structure,
                output_type=output_type,
                metrics=metrics,
                technical_features=technical_features,
            )

        except Exception as e:
            print(f"Error analyzing {problem_file}: {e}")
            return None

    def _extract_domain(self, content: str, filename: str) -> str:
        """Extract domain from content or filename."""
        # Check explicit domain mentions
        for domain in self.domains:
            if domain in filename.lower():
                return domain
            if f"domain: {domain}" in content.lower():
                return domain

        # Check for domain-specific keywords
        content_lower = content.lower()
        domain_scores = {}

        for domain, keywords in self.domains.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return "general"

    def _extract_problem_type(self, content: str) -> str:
        """Extract problem type from content."""
        content_lower = content.lower()

        # Direct type mentions
        for ptype in self.problem_types:
            if f"problem type: {ptype}" in content_lower:
                return ptype
            if f"{ptype} problem" in content_lower:
                return ptype

        # Infer from method names and structure
        if "classify" in content_lower or "categorize" in content_lower:
            return "classification"
        elif "generate" in content_lower or "create" in content_lower:
            return "generation"
        elif "extract" in content_lower:
            return "extraction"
        elif "answer" in content_lower or "question" in content_lower:
            return "question_answering"
        elif "analyze" in content_lower or "analysis" in content_lower:
            return "analysis"

        return "general"

    def _extract_difficulty(self, content: str) -> str:
        """Extract difficulty level from content."""
        content_lower = content.lower()

        for level in self.difficulty_levels:
            if f"difficulty: {level}" in content_lower:
                return level
            if f"difficulty_level: '{level}'" in content_lower:
                return level
            if f'difficulty_level: "{level}"' in content_lower:
                return level

        # Default based on content complexity
        if "expert" in content_lower or "advanced" in content_lower:
            return "expert"
        elif "complex" in content_lower or "challenging" in content_lower:
            return "hard"

        return "medium"

    def _count_examples(self, content: str) -> int:
        """Count number of examples in the problem."""
        # Look for example lists or counts
        example_matches = re.findall(r"examples?\s*[:=]\s*(\d+)", content.lower())
        if example_matches:
            return int(example_matches[0])

        # Count actual example definitions
        example_defs = len(re.findall(r"'id':\s*\d+", content))
        if example_defs > 0:
            return example_defs

        return 0

    def _extract_input_structure(self, content: str) -> Dict[str, str]:
        """Extract input structure from content."""
        # Look for input_data definitions
        input_matches = re.findall(r"input_data['\"]?\s*:\s*{([^}]+)}", content)

        if input_matches:
            # Parse the structure
            structure = {}
            for match in input_matches[:3]:  # Sample first few
                # Extract key-value pairs
                pairs = re.findall(r"['\"](\w+)['\"]:\s*([^,}]+)", match)
                for key, value in pairs:
                    if key not in structure:
                        # Infer type from value
                        if "str" in value or "text" in value:
                            structure[key] = "str"
                        elif "int" in value or "number" in value:
                            structure[key] = "int"
                        elif "list" in value or "array" in value:
                            structure[key] = "list"
                        else:
                            structure[key] = "any"

            return structure if structure else {"input": "str"}

        return {"input": "str"}

    def _extract_output_type(self, content: str) -> str:
        """Extract output type from content."""
        # Look for expected_output patterns
        if "expected_output" in content:
            if re.search(r"expected_output['\"]?\s*:\s*{", content):
                return "dict"
            elif re.search(r"expected_output['\"]?\s*:\s*\[", content):
                return "list"
            else:
                return "str"

        return "str"

    def _extract_metrics(self, content: str) -> List[str]:
        """Extract evaluation metrics from content."""
        metrics = []

        # Common metric patterns
        metric_names = [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "bleu",
            "rouge",
            "perplexity",
            "coherence",
            "relevance",
        ]

        content_lower = content.lower()
        for metric in metric_names:
            if metric in content_lower:
                metrics.append(metric)

        # Look for metrics list
        metrics_match = re.search(r"metrics['\"]?\s*[:=]\s*\[([^\]]+)\]", content)
        if metrics_match:
            metric_list = metrics_match.group(1)
            extracted = re.findall(r"['\"](\w+)['\"]", metric_list)
            metrics.extend(extracted)

        return list(set(metrics))

    def _extract_technical_features(self, content: str) -> Set[str]:
        """Extract technical features tested by the problem."""
        features = set()
        content_lower = content.lower()

        feature_indicators = {
            "multi_step_reasoning": ["multi-step", "reasoning", "chain of thought"],
            "structured_output": ["json", "structured", "schema", "format"],
            "context_understanding": ["context", "background", "situation"],
            "ambiguity_handling": ["ambiguous", "unclear", "multiple interpretations"],
            "domain_expertise": ["domain", "expertise", "specialized"],
            "creative_generation": ["creative", "novel", "original"],
            "factual_accuracy": ["factual", "accurate", "correct"],
            "logical_consistency": ["logical", "consistent", "coherent"],
        }

        for feature, indicators in feature_indicators.items():
            if any(ind in content_lower for ind in indicators):
                features.add(feature)

        return features

    def _identify_domain_gaps(self) -> List[ProblemOpportunity]:
        """Identify gaps in domain coverage."""
        gaps = []

        # Count problems per domain
        domain_counts = defaultdict(int)
        for problem in self.existing_problems.values():
            domain_counts[problem.domain] += 1

        # Find underrepresented domains
        for domain in self.domains:
            count = domain_counts.get(domain, 0)

            if count == 0:
                gaps.append(
                    ProblemOpportunity(
                        suggested_name=f"{domain}_assistant",
                        domain=domain,
                        problem_type="classification",  # Start with classification
                        difficulty_level="medium",
                        rationale=f"No problems exist for {domain} domain",
                        unique_features=[
                            f"{domain}_expertise",
                            "domain_specific_knowledge",
                        ],
                        target_capabilities=[
                            "domain_expertise",
                            "context_understanding",
                        ],
                        estimated_impact="high",
                    )
                )
            elif count < 2:
                # Suggest different problem type for diversity
                existing_types = [
                    p.problem_type
                    for p in self.existing_problems.values()
                    if p.domain == domain
                ]
                new_type = self._suggest_different_type(existing_types)

                gaps.append(
                    ProblemOpportunity(
                        suggested_name=f"{domain}_{new_type}",
                        domain=domain,
                        problem_type=new_type,
                        difficulty_level="hard",
                        rationale=f"Limited coverage in {domain} domain (only {count} problem)",
                        unique_features=[f"{domain}_advanced", new_type],
                        target_capabilities=[
                            "domain_expertise",
                            self._type_to_capability(new_type),
                        ],
                        estimated_impact="medium",
                    )
                )

        return gaps

    def _identify_problem_type_gaps(self) -> List[ProblemOpportunity]:
        """Identify gaps in problem type coverage."""
        gaps = []

        # Count problems per type
        type_counts = defaultdict(int)
        for problem in self.existing_problems.values():
            type_counts[problem.problem_type] += 1

        # Priority problem types that are missing
        priority_types = ["generation", "extraction", "question_answering", "reasoning"]

        for ptype in priority_types:
            if type_counts.get(ptype, 0) == 0:
                # Find suitable domain for this type
                domain = self._suggest_domain_for_type(ptype)

                gaps.append(
                    ProblemOpportunity(
                        suggested_name=f"{domain}_{ptype}",
                        domain=domain,
                        problem_type=ptype,
                        difficulty_level="medium",
                        rationale=f"No {ptype} problems exist",
                        unique_features=[ptype, f"{ptype}_capability"],
                        target_capabilities=[self._type_to_capability(ptype)],
                        estimated_impact="high",
                    )
                )

        return gaps

    def _identify_difficulty_gaps(self) -> List[ProblemOpportunity]:
        """Identify gaps in difficulty distribution."""
        gaps = []

        # Count problems per difficulty
        difficulty_counts = defaultdict(int)
        for problem in self.existing_problems.values():
            difficulty_counts[problem.difficulty_level] += 1

        # Check for missing easy problems
        if difficulty_counts.get("easy", 0) < 2:
            gaps.append(
                ProblemOpportunity(
                    suggested_name="simple_text_classification",
                    domain="general",
                    problem_type="classification",
                    difficulty_level="easy",
                    rationale="Insufficient easy problems for beginners",
                    unique_features=["beginner_friendly", "clear_examples"],
                    target_capabilities=["basic_understanding"],
                    estimated_impact="medium",
                )
            )

        # Check for expert problems
        if difficulty_counts.get("expert", 0) < 2:
            gaps.append(
                ProblemOpportunity(
                    suggested_name="advanced_reasoning_challenge",
                    domain="scientific",
                    problem_type="reasoning",
                    difficulty_level="expert",
                    rationale="Need more expert-level challenges",
                    unique_features=["complex_reasoning", "multi_domain"],
                    target_capabilities=["multi_step_reasoning", "domain_expertise"],
                    estimated_impact="medium",
                )
            )

        return gaps

    def _identify_capability_gaps(self) -> List[ProblemOpportunity]:
        """Identify gaps in technical capability coverage."""
        gaps = []

        # Collect all tested capabilities
        tested_capabilities = set()
        for problem in self.existing_problems.values():
            tested_capabilities.update(problem.technical_features)

        # High-priority capabilities
        priority_capabilities = [
            "structured_output",
            "multi_step_reasoning",
            "temporal_reasoning",
            "mathematical_computation",
        ]

        for capability in priority_capabilities:
            if capability not in tested_capabilities:
                domain, ptype = self._suggest_problem_for_capability(capability)

                gaps.append(
                    ProblemOpportunity(
                        suggested_name=f"{capability.replace('_', '_')}_{ptype}",
                        domain=domain,
                        problem_type=ptype,
                        difficulty_level="hard",
                        rationale=f"No problems test {capability}",
                        unique_features=[capability, f"{capability}_focused"],
                        target_capabilities=[capability],
                        estimated_impact="high",
                    )
                )

        return gaps

    def _opportunity_to_specification(
        self, opportunity: ProblemOpportunity
    ) -> ProblemSpecification:
        """Convert an opportunity to a full specification."""
        # Generate appropriate structures based on type
        input_structure, output_structure = self._generate_structures(
            opportunity.problem_type, opportunity.domain
        )

        # Generate description
        description = self._generate_description(
            opportunity.domain,
            opportunity.problem_type,
            opportunity.target_capabilities,
        )

        # Select appropriate metrics
        metrics = self._select_metrics(opportunity.problem_type)

        # Generate example patterns
        patterns = self._generate_example_patterns(
            opportunity.domain, opportunity.problem_type, opportunity.difficulty_level
        )

        return ProblemSpecification(
            name=opportunity.suggested_name,
            description=description,
            domain=opportunity.domain,
            problem_type=opportunity.problem_type,
            difficulty_level=opportunity.difficulty_level,
            example_count=1000,  # Target count
            input_structure=input_structure,
            output_structure=output_structure,
            evaluation_metrics=metrics,
            example_patterns=patterns,
            technical_requirements=opportunity.target_capabilities,
        )

    def _generate_balanced_problem(
        self,
        existing_specs: List[ProblemSpecification],
        domains_used: Dict[str, int],
        types_used: Dict[str, int],
        difficulties_used: Dict[str, int],
    ) -> ProblemSpecification:
        """Generate a balanced problem to fill remaining slots."""
        # Find least used domain
        all_domains = list(self.domains.keys())
        domain = min(all_domains, key=lambda d: domains_used.get(d, 0))

        # Find least used type
        all_types = list(self.problem_types.keys())
        ptype = min(all_types, key=lambda t: types_used.get(t, 0))

        # Find least used difficulty
        difficulty = min(
            self.difficulty_levels, key=lambda d: difficulties_used.get(d, 0)
        )

        # Create specification
        opportunity = ProblemOpportunity(
            suggested_name=f"{domain}_{ptype}_{len(existing_specs)}",
            domain=domain,
            problem_type=ptype,
            difficulty_level=difficulty,
            rationale="Balanced coverage",
            unique_features=[f"{domain}_{ptype}"],
            target_capabilities=[self._type_to_capability(ptype)],
            estimated_impact="medium",
        )

        return self._opportunity_to_specification(opportunity)

    def _suggest_different_type(self, existing_types: List[str]) -> str:
        """Suggest a problem type different from existing ones."""
        all_types = list(self.problem_types.keys())
        for ptype in all_types:
            if ptype not in existing_types:
                return ptype
        return "analysis"  # Fallback

    def _type_to_capability(self, problem_type: str) -> str:
        """Map problem type to primary capability."""
        mapping = {
            "classification": "context_understanding",
            "generation": "creative_generation",
            "extraction": "structured_output",
            "question_answering": "factual_accuracy",
            "analysis": "multi_step_reasoning",
            "transformation": "logical_consistency",
            "completion": "context_understanding",
            "reasoning": "multi_step_reasoning",
        }
        return mapping.get(problem_type, "context_understanding")

    def _suggest_domain_for_type(self, problem_type: str) -> str:
        """Suggest appropriate domain for a problem type."""
        suggestions = {
            "generation": "creative",
            "extraction": "business",
            "question_answering": "educational",
            "reasoning": "scientific",
            "analysis": "technical",
            "transformation": "technical",
            "completion": "technical",
        }
        return suggestions.get(problem_type, "general")

    def _suggest_problem_for_capability(self, capability: str) -> Tuple[str, str]:
        """Suggest domain and type for testing a capability."""
        suggestions = {
            "structured_output": ("technical", "extraction"),
            "multi_step_reasoning": ("scientific", "reasoning"),
            "temporal_reasoning": ("business", "analysis"),
            "mathematical_computation": ("educational", "reasoning"),
        }
        return suggestions.get(capability, ("general", "analysis"))

    def _generate_structures(
        self, problem_type: str, domain: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Generate input/output structures for a problem."""
        # Input structures by type
        if problem_type == "classification":
            input_structure = {"text": "str", "context": "Optional[str]"}
        elif problem_type == "generation":
            input_structure = {"prompt": "str", "constraints": "Optional[Dict]"}
        elif problem_type == "extraction":
            input_structure = {"document": "str", "schema": "Dict"}
        elif problem_type == "question_answering":
            input_structure = {"question": "str", "context": "str"}
        else:
            input_structure = {"input": "str"}

        # Output structures by type
        if problem_type == "classification":
            output_structure = {"class": "str", "confidence": "float"}
        elif problem_type == "generation":
            output_structure = {"generated_text": "str", "metadata": "Dict"}
        elif problem_type == "extraction":
            output_structure = {"entities": "List[Dict]", "relations": "List[Dict]"}
        elif problem_type == "structured":
            output_structure = {"data": "Dict", "valid": "bool"}
        else:
            output_structure = {"output": "str"}

        return input_structure, output_structure

    def _generate_description(
        self, domain: str, problem_type: str, capabilities: List[str]
    ) -> str:
        """Generate problem description."""
        capability_str = ", ".join(capabilities)

        templates = {
            "classification": f"Classify {domain} content into relevant categories",
            "generation": f"Generate {domain}-specific content based on prompts",
            "extraction": f"Extract structured information from {domain} documents",
            "question_answering": f"Answer questions about {domain} topics",
            "analysis": f"Analyze {domain} content for insights",
            "reasoning": f"Solve {domain} problems requiring logical reasoning",
        }

        base = templates.get(problem_type, f"Process {domain} content")
        return f"{base}. Tests {capability_str}."

    def _select_metrics(self, problem_type: str) -> List[str]:
        """Select appropriate metrics for problem type."""
        metrics_map = {
            "classification": ["accuracy", "f1_score", "precision", "recall"],
            "generation": ["coherence", "relevance", "fluency", "diversity"],
            "extraction": ["precision", "recall", "f1_score", "exact_match"],
            "question_answering": ["exact_match", "f1_score", "accuracy"],
            "analysis": ["accuracy", "completeness", "insight_quality"],
            "reasoning": ["correctness", "reasoning_steps", "logical_consistency"],
        }
        return metrics_map.get(problem_type, ["accuracy", "quality_score"])

    def _generate_example_patterns(
        self, domain: str, problem_type: str, difficulty: str
    ) -> List[str]:
        """Generate example patterns for a problem."""
        patterns = []

        # Base patterns by type
        if problem_type == "classification":
            patterns.extend(
                ["simple_case", "ambiguous_case", "edge_case", "multi_label"]
            )
        elif problem_type == "generation":
            patterns.extend(["short_form", "long_form", "constrained", "creative"])
        elif problem_type == "extraction":
            patterns.extend(["single_entity", "multiple_entities", "nested_structure"])

        # Add difficulty-specific patterns
        if difficulty in ["very_hard", "expert"]:
            patterns.extend(["complex_reasoning", "subtle_distinctions", "rare_cases"])
        elif difficulty == "easy":
            patterns.extend(["clear_examples", "typical_cases", "straightforward"])

        return patterns

    def export_gap_analysis(self, output_file: str = "problem_gaps.json"):
        """Export gap analysis to file."""
        self.analyze_existing_problems()
        gaps = self.identify_gaps()

        analysis = {
            "existing_problems": {
                name: {
                    "domain": p.domain,
                    "type": p.problem_type,
                    "difficulty": p.difficulty_level,
                    "examples": p.example_count,
                    "capabilities": list(p.technical_features),
                }
                for name, p in self.existing_problems.items()
            },
            "identified_gaps": [
                {
                    "name": g.suggested_name,
                    "domain": g.domain,
                    "type": g.problem_type,
                    "difficulty": g.difficulty_level,
                    "rationale": g.rationale,
                    "impact": g.estimated_impact,
                }
                for g in gaps
            ],
            "coverage_summary": {
                "domains": self._calculate_domain_coverage(),
                "types": self._calculate_type_coverage(),
                "difficulties": self._calculate_difficulty_coverage(),
            },
        }

        output_path = validate_path(output_file, Path.cwd())
        safe_write_text(output_path, json.dumps(analysis, indent=2), Path.cwd())

    def _calculate_domain_coverage(self) -> Dict[str, float]:
        """Calculate coverage percentage for each domain."""
        coverage = {}
        for domain in self.domains:
            count = sum(
                1 for p in self.existing_problems.values() if p.domain == domain
            )
            coverage[domain] = count / max(len(self.existing_problems), 1)
        return coverage

    def _calculate_type_coverage(self) -> Dict[str, float]:
        """Calculate coverage percentage for each problem type."""
        coverage = {}
        for ptype in self.problem_types:
            count = sum(
                1 for p in self.existing_problems.values() if p.problem_type == ptype
            )
            coverage[ptype] = count / max(len(self.existing_problems), 1)
        return coverage

    def _calculate_difficulty_coverage(self) -> Dict[str, float]:
        """Calculate coverage percentage for each difficulty level."""
        coverage = {}
        for difficulty in self.difficulty_levels:
            count = sum(
                1
                for p in self.existing_problems.values()
                if p.difficulty_level == difficulty
            )
            coverage[difficulty] = count / max(len(self.existing_problems), 1)
        return coverage
