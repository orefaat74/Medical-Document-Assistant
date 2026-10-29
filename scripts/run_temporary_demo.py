"""Expose the running Streamlit app with a temporary Cloudflare review URL."""

import json
import re
import subprocess
import time
from pathlib import Path

from medical_doc_assistant.storage import utc_now, write_json


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    logs = root / "reports" / "phase5" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / "cloudflare.log"
    with logfile.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(["cloudflared", "tunnel", "--url", "http://127.0.0.1:8501", "--no-autoupdate"],
                                   stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        url = None
        while time.monotonic() < deadline and process.poll() is None:
            content = logfile.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", content)
            if match:
                url = match.group(0)
                break
            time.sleep(1)
        if not url:
            process.terminate()
            raise SystemExit("Temporary tunnel startup failed; see reports/phase5/logs/cloudflare.log")
        details = {"url": url, "kind": "temporary_cloudflare_review_tunnel", "started_at_utc": utc_now(),
                   "tunnel_pid": process.pid, "local_url": "http://127.0.0.1:8501",
                   "requires_local_application_and_tunnel": True, "persistent_hosting": False}
        write_json(root / "reports" / "phase5" / "live_demo.json", details)
        print(json.dumps(details, indent=2), flush=True)
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
