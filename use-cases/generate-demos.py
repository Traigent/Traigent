#!/usr/bin/env python3
"""Generate asciinema cast files for all use-case demos."""

import json
import os
import subprocess
import time

# Timing constants (in seconds) - adjusted for comfortable viewing pace
TYPING_SPEED = 0.05  # Time per character when typing commands
OUTPUT_LINE_DELAY = 0.12  # Time between output lines
EMPTY_LINE_DELAY = 0.08  # Time for empty lines
COMMAND_PAUSE = 0.6  # Pause after typing a command
POST_COMMAND_DELAY = 0.4  # Delay before showing output
FINAL_PAUSE = 2.0  # Pause at end of recording


def generate_cast(script_path: str, output_path: str, title: str):
    """Generate an asciinema cast file from a script."""
    env = os.environ.copy()
    env["TERM"] = "xterm-256color"
    script_dir = os.path.dirname(os.path.abspath(script_path))

    result = subprocess.run(
        ["bash", script_path], capture_output=True, text=True, env=env, cwd=script_dir
    )

    output = result.stdout + result.stderr

    header = {
        "version": 2,
        "width": 80,
        "height": 24,
        "timestamp": int(time.time()),
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
        "title": title,
    }

    with open(output_path, "w") as f:
        f.write(json.dumps(header) + "\n")

        current_time = 0.0
        lines = output.split("\n")

        for line in lines:
            if line.startswith("$") or line.startswith("#"):
                # Type character by character for commands/comments
                for char in line:
                    f.write(json.dumps([current_time, "o", char]) + "\n")
                    current_time += TYPING_SPEED
                current_time += POST_COMMAND_DELAY
                f.write(json.dumps([current_time, "o", "\r\n"]) + "\n")
                current_time += COMMAND_PAUSE
            else:
                # Output lines appear with readable pacing
                if line:
                    f.write(json.dumps([current_time, "o", line + "\r\n"]) + "\n")
                    current_time += OUTPUT_LINE_DELAY
                else:
                    f.write(json.dumps([current_time, "o", "\r\n"]) + "\n")
                    current_time += EMPTY_LINE_DELAY

        f.write(json.dumps([current_time + FINAL_PAUSE, "o", ""]) + "\n")

    print(f"  Generated {output_path}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    demos = [
        (
            "gtm-acquisition/demo/demo-gtm.sh",
            "gtm-acquisition/demo/demo.cast",
            "GTM & Acquisition: SDR Outbound Optimization",
        ),
        (
            "operations/demo/demo-operations.sh",
            "operations/demo/demo.cast",
            "Operations: Workflow Automation Optimization",
        ),
        (
            "knowledge-rag/demo/demo-knowledge-rag.sh",
            "knowledge-rag/demo/demo.cast",
            "Knowledge & RAG: Document Q&A Optimization",
        ),
        (
            "product-technical/demo/demo-product-technical.sh",
            "product-technical/demo/demo.cast",
            "Product & Technical: Code Generation Optimization",
        ),
        (
            "customer-support/demo/demo-customer-support.sh",
            "customer-support/demo/demo.cast",
            "Customer Support: ShopEasy Bot Optimization",
        ),
    ]

    print("Generating asciinema cast files for use-case demos...")
    print()

    for script, output, title in demos:
        script_path = os.path.join(script_dir, script)
        output_path = os.path.join(script_dir, output)

        if os.path.exists(script_path):
            print(f"-> {title}")
            try:
                generate_cast(script_path, output_path, title)
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print(f"-> {title} (skipped - {script} not found)")

    print()
    print("Done! To convert to SVG, run:")
    print("  npm install -g svg-term-cli")
    print("  svg-term --in demo.cast --out demo.svg --window")
    print()
    print("Or use the record-demos.sh script in docs/demos/")


if __name__ == "__main__":
    main()
