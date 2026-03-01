import subprocess

try:
    subprocess.run("echo hello && sleep 2", shell=True, capture_output=True, text=True, timeout=1)
except subprocess.TimeoutExpired as exc:
    print(repr(exc.stdout))
