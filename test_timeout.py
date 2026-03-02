import subprocess

try:
    subprocess.run(["sleep", "2"], capture_output=True, text=True, timeout=1)
except subprocess.TimeoutExpired as exc:
    print(repr(exc.stdout))
    print(repr(exc.stderr))
