#!/usr/bin/env python3
"""Generate asciinema cast files from demo scripts."""

import json
import subprocess
import time
import os

# Timing constants (in seconds) - adjust for comfortable viewing pace
TYPING_SPEED = 0.06         # Time per character when typing commands (was 0.03)
OUTPUT_LINE_DELAY = 0.15    # Time between output lines (was 0.05)
EMPTY_LINE_DELAY = 0.10     # Time for empty lines (was 0.02)
COMMAND_PAUSE = 0.8         # Pause after typing a command (was 0.5)
POST_COMMAND_DELAY = 0.6    # Delay before showing output (was 0.3)
FINAL_PAUSE = 2.0           # Pause at end of recording (was 1.0)


def generate_cast(script_path: str, output_path: str, title: str):
    """Generate an asciinema cast file from a script."""

    env = os.environ.copy()
    env['TERM'] = 'xterm-256color'
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))

    result = subprocess.run(
        ['bash', script_path],
        capture_output=True,
        text=True,
        env=env,
        cwd=script_dir
    )

    output = result.stdout + result.stderr

    header = {
        "version": 2,
        "width": 100,
        "height": 30,
        "timestamp": int(time.time()),
        "env": {"SHELL": "/bin/bash", "TERM": "xterm-256color"},
        "title": title
    }

    with open(output_path, 'w') as f:
        f.write(json.dumps(header) + '\n')

        current_time = 0.0
        lines = output.split('\n')

        for line in lines:
            if line.startswith('$') or line.startswith('#'):
                # Type character by character for commands/comments
                for char in line:
                    f.write(json.dumps([current_time, "o", char]) + '\n')
                    current_time += TYPING_SPEED
                current_time += POST_COMMAND_DELAY
                f.write(json.dumps([current_time, "o", "\r\n"]) + '\n')
                current_time += COMMAND_PAUSE
            else:
                # Output lines appear with readable pacing
                if line:
                    f.write(json.dumps([current_time, "o", line + "\r\n"]) + '\n')
                    current_time += OUTPUT_LINE_DELAY
                else:
                    f.write(json.dumps([current_time, "o", "\r\n"]) + '\n')
                    current_time += EMPTY_LINE_DELAY

        f.write(json.dumps([current_time + FINAL_PAUSE, "o", ""]) + '\n')

    print(f"  Generated {output_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demos_dir = os.path.dirname(script_dir)
    output_dir = os.path.join(demos_dir, 'output')

    os.makedirs(output_dir, exist_ok=True)

    demos = [
        ('demo-optimize.sh', 'optimize.cast', 'Traigent LLM Optimization'),
        ('demo-hooks.sh', 'hooks.cast', 'Traigent Optimization Callbacks'),
        ('demo-github-hooks.sh', 'github-hooks.cast', 'Traigent GitHub Hooks'),
    ]

    print("Generating asciinema cast files...")
    print()

    for script, output, title in demos:
        script_path = os.path.join(script_dir, script)
        output_path = os.path.join(output_dir, output)

        if os.path.exists(script_path):
            print(f"-> {title}")
            try:
                generate_cast(script_path, output_path, title)
            except Exception as e:
                print(f"   Error: {e}")
        else:
            print(f"-> {title} (skipped - {script} not found)")

    print()
    print("Done!")

if __name__ == '__main__':
    main()
