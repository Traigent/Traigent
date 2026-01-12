"""GPT-4.1 Replication Study.

This use-case replicates key experiments from OpenAI's GPT-4.1 announcement
to validate the claimed improvements in:
- Coding performance (SWE-bench, Aider diff format)
- Instruction following (MultiChallenge, IFEval patterns)
- Long context comprehension (MRCR, needle-in-haystack)
- Function calling reliability (ComplexFuncBench patterns)

Models compared: GPT-4.1, GPT-4.1-mini, GPT-4.1-nano, GPT-4o, GPT-4o-mini
"""
