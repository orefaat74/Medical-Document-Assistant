"""Launch and supervise the local FastAPI + Streamlit application."""

import argparse
import json
import os
import subprocess
import signal
import sys
import time
from pathlib import Path

import requests


def main() -> None:
    def stop(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-port", type=int, default=8501)
    parser.add_argument("--frontend-host", default="127.0.0.1")
    parser.add_argument("--api-port", type=int, default=8000)
    parser.add_argument("--api-host", default="127.0.0.1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    logs = root / "reports" / "phase5" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, MEDICAL_API_URL=f"http://127.0.0.1:{args.api_port}", MEDICAL_PROJECT_ROOT=str(root))
    children = []
    with (logs / "api.log").open("a", encoding="utf-8") as api_log, (logs / "frontend.log").open("a", encoding="utf-8") as frontend_log:
        try:
            children.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "medical_doc_assistant.api:app", "--host", args.api_host, "--port", str(args.api_port)],
                                             cwd=root, env=environment, stdout=api_log, stderr=subprocess.STDOUT))
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if children[0].poll() is not None:
                    raise RuntimeError("API failed to start; see reports/phase5/logs/api.log")
                try:
                    if requests.get(environment["MEDICAL_API_URL"] + "/health", timeout=2).ok:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError("API startup timed out.")
            children.append(subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(root / "streamlit_app.py"),
                                             "--server.address", args.frontend_host, "--server.port", str(args.frontend_port),
                                             "--server.headless", "true", "--browser.gatherUsageStats", "false", "--server.maxUploadSize", "10"],
                                            cwd=root, env=environment, stdout=frontend_log, stderr=subprocess.STDOUT))
            state = {"api_pid": children[0].pid, "frontend_pid": children[1].pid,
                     "api_url": environment["MEDICAL_API_URL"], "frontend_url": f"http://127.0.0.1:{args.frontend_port}"}
            (logs / "processes.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
            print(json.dumps(state, indent=2), flush=True)
            while all(child.poll() is None for child in children):
                time.sleep(1)
            raise RuntimeError("An application process exited; see the logs.")
        except KeyboardInterrupt:
            pass
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        child.terminate()
            for child in children:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()


if __name__ == "__main__":
    main()
