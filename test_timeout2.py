import subprocess

try:
    subprocess.run(["sleep", "2"], capture_output=True, text=True, timeout=1)
except subprocess.TimeoutExpired as exc:
    timeout_stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    print(repr(timeout_stdout))
