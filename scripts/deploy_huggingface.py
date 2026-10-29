"""Stage the runtime files and deploy a Docker Space using an existing local HF login."""

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the capstone Docker app to Hugging Face Spaces")
    parser.add_argument("repo_id", help="Account/Space name, for example username/medical-document-assistant")
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise SystemExit("Sign in locally first with `huggingface-cli login` or set HF_TOKEN, then rerun this deployment command.")
    root = Path(__file__).resolve().parents[1]
    staging_parent = root / ".cache"
    staging_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="space-", dir=staging_parent) as temporary:
        destination = Path(temporary)
        files = ["Dockerfile", ".dockerignore", "pyproject.toml", "streamlit_app.py", "requirements-phase1.txt", "requirements-phase2.txt",
                 "requirements-phase3.txt", "requirements-phase4.txt", "requirements-phase5.txt"]
        for filename in files:
            shutil.copy2(root / filename, destination / filename)
        shutil.copy2(root / "deployment" / "huggingface" / "README.md", destination / "README.md")
        for directory in ("src", "configs", "scripts", "docs", "data/examples", "models/phase2", "models/phase4/bert/tokenizer"):
            shutil.copytree(root / directory, destination / directory, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (destination / "models" / "phase3").mkdir(parents=True)
        shutil.copy2(root / "models" / "phase3" / "text_cnn.pt", destination / "models" / "phase3" / "text_cnn.pt")
        shutil.copy2(root / "models" / "phase4" / "bert" / "model.pt", destination / "models" / "phase4" / "bert" / "model.pt")
        (destination / "reports" / "phase4" / "bert").mkdir(parents=True)
        shutil.copy2(root / "reports" / "phase4" / "bert" / "model_comparison.csv", destination / "reports" / "phase4" / "bert" / "model_comparison.csv")
        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(repo_id=args.repo_id, repo_type="space", folder_path=destination,
                          commit_message="Deploy Medical Document Assistant capstone")
    print("https://huggingface.co/spaces/" + args.repo_id)


if __name__ == "__main__":
    main()
