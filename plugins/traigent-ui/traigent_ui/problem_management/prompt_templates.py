"""
High-quality prompt templates for example generation.

This module provides problem type-specific prompt templates with few-shot examples
to ensure consistent, high-quality example generation for LangChain optimization problems.
"""

from typing import List, Optional


class PromptTemplates:
    """Collection of optimized prompt templates for different problem types."""

    @staticmethod
    def get_classification_prompt(
        description: str,
        count: int,
        domain: str,
        categories: Optional[List[str]] = None,
    ) -> str:
        """Generate prompt for classification examples."""
        categories_str = (
            ", ".join(categories) if categories else "appropriate categories"
        )

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} classification examples for: {description}

Domain: {domain}
Categories: {categories_str}

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. Each example MUST follow this EXACT format
3. The "output" field MUST contain ONLY the category name, nothing else
4. Output ONLY the JSON examples, no explanations or text outside JSON

FORMAT (follow exactly):
{{
  "input": {{"text": "example text here"}},
  "output": "category_name"
}}

Example 1:
{{
  "input": {{"text": "I can't log into my account, it keeps saying invalid password"}},
  "output": "technical_support"
}}

Example 2:
{{
  "input": {{"text": "When will my order #12345 arrive? It's been 5 days"}},
  "output": "shipping_inquiry"
}}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @staticmethod
    def get_reasoning_prompt(
        description: str, count: int, domain: str, reasoning_type: str = "general"
    ) -> str:
        """Generate prompt for reasoning examples."""

        # Domain-specific examples
        if domain == "technical" and "assembly" in description.lower():
            example_format = """Example 1:
{{
  "input": {{
    "problem": "Given the assembly code:\\nMOV AX, 5\\nMOV BX, 3\\nADD AX, BX\\nWhat is the final value in AX?"
  }},
  "output": {{
    "steps": [
      "AX = 5 (MOV AX, 5)",
      "BX = 3 (MOV BX, 3)",
      "AX = AX + BX = 5 + 3 = 8 (ADD AX, BX)"
    ],
    "answer": "8",
    "explanation": "The ADD instruction adds BX to AX, storing result in AX"
  }}
}}

Example 2:
{{
  "input": {{
    "problem": "Trace this code:\\nMOV CX, 4\\nLOOP_START:\\nDEC CX\\nJNZ LOOP_START\\nHow many times does the loop execute?"
  }},
  "output": {{
    "steps": [
      "CX = 4 initially",
      "First iteration: CX = 3, jump taken",
      "Second iteration: CX = 2, jump taken",
      "Third iteration: CX = 1, jump taken",
      "Fourth iteration: CX = 0, jump not taken (exit)"
    ],
    "answer": "4",
    "explanation": "Loop executes 4 times until CX reaches 0"
  }}
}}"""
        elif domain == "mathematical":
            example_format = """Example 1:
{{
  "input": {{
    "problem": "If a train travels 120 miles in 2 hours, what is its average speed?"
  }},
  "output": {{
    "steps": [
      "Speed = Distance / Time",
      "Distance = 120 miles",
      "Time = 2 hours",
      "Speed = 120 / 2 = 60 mph"
    ],
    "answer": "60 mph",
    "explanation": "Average speed is total distance divided by total time"
  }}
}}"""
        else:
            example_format = """Example 1:
{{
  "input": {{
    "problem": "If all roses are flowers, and all flowers need water, what can we conclude about roses?"
  }},
  "output": {{
    "steps": [
      "All roses are flowers (given)",
      "All flowers need water (given)",
      "Therefore, all roses need water (transitive property)"
    ],
    "answer": "All roses need water",
    "explanation": "Using logical deduction from the given premises"
  }}
}}"""

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} reasoning problem examples for: {description}

Domain: {domain}
Type: {reasoning_type} reasoning

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. Each example MUST have this EXACT structure
3. The "output" MUST be a JSON object with "steps", "answer", and "explanation" fields
4. Each step MUST be a complete sentence
5. Output ONLY the JSON examples, no explanations outside JSON

REQUIRED FORMAT:
{{
  "input": {{"problem": "problem statement here"}},
  "output": {{
    "steps": ["Step 1 description", "Step 2 description", ...],
    "answer": "final answer",
    "explanation": "brief explanation"
  }}
}}

{example_format}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @staticmethod
    def get_generation_prompt(description: str, count: int, domain: str) -> str:
        """Generate prompt for text generation examples."""

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} text generation examples for: {description}

Domain: {domain}

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. The "output" field MUST contain the generated text (50-200 words)
3. Output ONLY the JSON examples, no explanations outside JSON
4. Generated text must be complete and coherent

FORMAT:
{{
  "input": {{"text": "generation request"}},
  "output": "generated text content"
}}

Example:
{{
  "input": {{"text": "Write a brief product description for an eco-friendly water bottle"}},
  "output": "Stay hydrated sustainably with our eco-friendly water bottle. Made from 100% recycled materials, this durable 24oz bottle features double-wall insulation to keep drinks cold for 24 hours or hot for 12. The leak-proof design ensures no spills in your bag."
}}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @staticmethod
    def get_qa_prompt(description: str, count: int, domain: str) -> str:
        """Generate prompt for question-answering examples."""

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} question-answering examples for: {description}

Domain: {domain}

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. Input MUST have "question" and "context" fields
3. Output MUST be a direct, concise answer (1-2 sentences)
4. Answer MUST be derivable from the context
5. Output ONLY the JSON examples, no explanations outside JSON

FORMAT:
{{
  "input": {{
    "question": "specific question",
    "context": "relevant context containing the answer"
  }},
  "output": "direct answer"
}}

Example:
{{
  "input": {{
    "question": "When was Python first released?",
    "context": "Python is a programming language created by Guido van Rossum and first released in 1991. It emphasizes code readability."
  }},
  "output": "Python was first released in 1991."
}}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @staticmethod
    def get_extraction_prompt(description: str, count: int, domain: str) -> str:
        """Generate prompt for information extraction examples."""

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} information extraction examples for: {description}

Domain: {domain}

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. Input MUST have a "text" field with extractable information
3. Output MUST be a JSON object with extracted fields
4. Extract ONLY information explicitly stated in the text
5. Output ONLY the JSON examples, no explanations outside JSON

FORMAT:
{{
  "input": {{"text": "text containing information to extract"}},
  "output": {{
    "field1": "extracted value",
    "field2": "extracted value"
  }}
}}

Example:
{{
  "input": {{"text": "John Smith (john@email.com) called at 3:30 PM about order #12345."}},
  "output": {{
    "name": "John Smith",
    "email": "john@email.com",
    "time": "3:30 PM",
    "order_id": "12345"
  }}
}}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @staticmethod
    def get_summarization_prompt(description: str, count: int, domain: str) -> str:
        """Generate prompt for summarization examples."""

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} summarization examples for: {description}

Domain: {domain}

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. Input text MUST be 100-200 words
3. Summary MUST be 20-40 words (20-30% of input)
4. Summary MUST capture key points and facts
5. Output ONLY the JSON examples, no explanations outside JSON

FORMAT:
{{
  "input": {{"text": "long text to summarize"}},
  "output": "concise summary"
}}

Example:
{{
  "input": {{"text": "The quarterly sales report shows a 15% increase in revenue compared to last quarter, driven by strong performance in Asia-Pacific with 25% growth. North America remained stable at 2% growth, while Europe declined by 5%. Three new products launched this quarter."}},
  "output": "Q3 revenue up 15%, led by APAC (25%). North America stable (2%), Europe down 5%. Three new products launched."
}}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @staticmethod
    def get_code_generation_prompt(description: str, count: int, domain: str) -> str:
        """Generate prompt for code generation examples."""

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} code generation examples for: {description}

Domain: {domain}

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. Input MUST have a "description" field describing what code to generate
3. Output MUST be working code (function, class, or algorithm)
4. Code MUST be syntactically correct and functional
5. Output ONLY the JSON examples, no explanations outside JSON

FORMAT:
{{
  "input": {{"description": "what to code"}},
  "output": "working code implementation"
}}

Example 1:
{{
  "input": {{"description": "Write a function to check if a string is a palindrome"}},
  "output": "function isPalindrome(str) {{\\n    const cleaned = str.toLowerCase().replace(/[^a-z0-9]/g, '');\\n    return cleaned === cleaned.split('').reverse().join('');\\n}}"
}}

Example 2:
{{
  "input": {{"description": "Create a function that finds the maximum value in an array"}},
  "output": "function findMax(arr) {{\\n    if (arr.length === 0) return undefined;\\n    return Math.max(...arr);\\n}}"
}}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @staticmethod
    def get_generic_prompt(
        description: str, count: int, problem_type: str, domain: str
    ) -> str:
        """Generate generic prompt for any problem type."""

        # Ensure we request small batches
        actual_count = min(count, 5)

        return f"""Generate EXACTLY {actual_count} examples for: {description}

Problem Type: {problem_type}
Domain: {domain}

STRICT REQUIREMENTS:
1. Output MUST be valid JSON
2. Follow the format exactly as shown
3. Output ONLY the JSON examples, no explanations outside JSON
4. Each example must be complete and self-contained

FORMAT:
{{
  "input": {{"text": "your input here" or other relevant fields}},
  "output": "expected output in appropriate format"
}}

Generate EXACTLY {actual_count} examples. Each on a new line. Output ONLY valid JSON:"""

    @classmethod
    def get_prompt_for_type(
        cls,
        problem_type: str,
        description: str,
        count: int,
        domain: str = "general",
        **kwargs,
    ) -> str:
        """Get the appropriate prompt template for a problem type."""

        prompt_map = {
            "classification": cls.get_classification_prompt,
            "reasoning": cls.get_reasoning_prompt,
            "generation": cls.get_generation_prompt,
            "question_answering": cls.get_qa_prompt,
            "information_extraction": cls.get_extraction_prompt,
            "summarization": cls.get_summarization_prompt,
            "code_generation": cls.get_code_generation_prompt,
        }

        prompt_func = prompt_map.get(problem_type, cls.get_generic_prompt)

        if prompt_func == cls.get_generic_prompt:
            return prompt_func(description, count, problem_type, domain)
        else:
            return prompt_func(description, count, domain, **kwargs)
