# Final application verification

Verified on 2026-10-28 with the supplied Windows/Python virtual environment and
the built Linux Docker image.

| Check | Result |
|---|---|
| Full software suite | **80 tests passed** |
| Local real API/browser flows | **14 checks passed** |
| Linux container real API flows | **9 checks passed** |
| Public HTTPS/browser/websocket flows | **5 checks passed** |
| Docker packaging | Image built; Compose container healthy |
| Dependency consistency | `pip check`: no broken requirements |
| Architecture and presentation | PNG/SVG diagram, eight-slide PPTX/PDF, and speaker notes generated and inspected |

Integration evidence is recorded in `integration_results.json`,
`container_integration_results.json`, and `public_demo_verification.json`.
The public review URL is recorded in `live_demo.json` and is explicitly temporary.

The browser checks used real model inference and verified example upload,
classification/source display, positive evidence chat, missing-answer fallback,
clearing, and PDF upload. API checks additionally exercised DOCX ingestion and
missing/cleared document isolation. No existing trained classifier was retuned
using these application tests.

The Docker build initially exposed an old-pip dependency-metadata issue on the
CPU PyTorch index. The image now upgrades pip and uses the exact CPU wheel plus
the normal dependency index. The successful image is recorded in
`container_build.json` (approximately 1.71 GB).

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/verify_capstone.py
.\.venv\Scripts\python.exe scripts/verify_capstone.py --api "http://127.0.0.1:8001" --frontend "http://127.0.0.1:7860" --skip-browser --output "reports/phase5/container_integration_results.json"
.\.venv\Scripts\python.exe scripts/verify_public_demo.py
.\.venv\Scripts\python.exe -m pip check
docker compose ps
```

The user selected the temporary live demo, Docker/persistent-hosting handoff, and
source package. No account-backed Hugging Face or GitHub publication was performed.
The source ZIP is generated without runtime uploads, caches, credentials, or
large trained binaries. Those model/data artifacts are supplied in the project folder.

The app is an educational English-text document assistant. These checks establish
software/serving consistency, not medical or multilingual/OCR clinical validation.
