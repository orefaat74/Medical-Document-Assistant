"""Build the deployment image with persistent capstone build evidence."""

import json
import subprocess
from pathlib import Path

from medical_doc_assistant.storage import utc_now, write_json


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    directory = root / "reports" / "phase5"
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status = directory / "container_build.json"
    write_json(status, {"status": "building", "started_at_utc": utc_now()})
    with (logs / "docker_build.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(["docker", "build", "-t", "medical-doc-assistant:capstone", str(root)], stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(status, {"status": "failed", "finished_at_utc": utc_now(), "returncode": result.returncode})
        raise SystemExit(result.returncode)
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", "medical-doc-assistant:capstone"], text=True))[0]
    write_json(status, {"status": "built", "finished_at_utc": utc_now(), "image": "medical-doc-assistant:capstone",
                       "image_id": image["Id"], "size_bytes": image["Size"]})
