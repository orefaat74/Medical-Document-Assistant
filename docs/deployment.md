# Deployment and operations

## Local application

```powershell
.\.venv\Scripts\python.exe scripts/run_app.py
```

- Frontend: <http://127.0.0.1:8501>
- API documentation: <http://127.0.0.1:8000/docs>
- API health: <http://127.0.0.1:8000/health>

The launcher supervises both processes and writes logs to
`reports/phase5/logs/`. Stop it with Ctrl+C. The default classifier is TextCNN;
other saved models are available in the frontend selector.

## Docker Compose

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f
```

- Frontend: <http://127.0.0.1:7860>
- Local API docs: <http://127.0.0.1:8001/docs>

```powershell
docker compose down
```

The image installs CPU PyTorch and downloads immutable public model revisions
during build. Trained classifiers are copied from the supplied local artifacts.
Training corpora, notebook caches, environment secrets, and runtime uploads are
excluded from the Docker context. A named volume stores demo runtime indexes.

## Persistent Hugging Face Spaces hosting

Sign in **locally** with a write-capable Hugging Face token:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
.\.venv\Scripts\python.exe scripts/deploy_huggingface.py YOUR_ACCOUNT/medical-document-assistant
```

The script stages the runtime source, trained model files, example, Dockerfile,
and model comparison. It creates/updates a Docker Space and prints its URL.
Tokens are read from the local login or environment; they are not included in the
staged files. The target Space is public by default.

The Docker frontend listens on port 7860, as required by Spaces. Public source
models are downloaded during the build; no inference API key is needed afterward.
Free CPU hosting can sleep and has limited resources. The demo runtime keeps
opaque document handles in one API process; handles are lost on restart.

## Temporary review link

With the local app already running:

```powershell
.\.venv\Scripts\python.exe scripts/run_temporary_demo.py
```

This uses `cloudflared` and records a temporary HTTPS URL in
`reports/phase5/live_demo.json`. It depends on the local application and tunnel
remaining active and is not permanent hosting.

## Configuration and data lifecycle

- `MEDICAL_PROJECT_ROOT` points to source/model assets.
- `MEDICAL_STORAGE_ROOT` points to demo document/index storage, default `.runtime/`.
- `MEDICAL_API_URL` tells Streamlit how to reach the API.
- `configs/phase5.json` defines file/text/page/question limits, classifier default,
  model CPU threads, document capacity, and a one-hour handle lifetime.
- Uploaded filenames are display metadata; they are not used as storage paths.
- Clearing an upload removes its opaque handle and deletes its internal index
  when no other active handle refers to the same text.

The application is an educational English-text demo, not clinical validation or
a production identity/authentication system. PDF scans need text extraction;
handwriting/OCR and evaluated Arabic models are documented extensions.

## Source repository handoff

The project contains `.gitignore`, source, tests, docs, and a generated source ZIP
under `deliverables/`. Large trained binaries and frozen datasets are supplied
locally and excluded from ordinary source tracking. A GitHub remote URL requires
the owner's GitHub account/repository publication step.
