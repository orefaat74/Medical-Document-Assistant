# Trained model artifacts for a fresh clone

The application uses six custom trained files totaling approximately **53.5 MiB**.
They are not public Hugging Face base models and cannot be recreated by
`scripts/fetch_phase4_models.py`. They must be distributed with the project.

The repository is configured to store these files with **Git LFS**:

- `models/phase2/*.joblib`
- `models/phase3/text_cnn.pt`
- `models/phase4/bert/model.pt`

The adjacent BERT tokenizer is ordinary Git content. The MiniLM, SQuAD2, and
FLAN-T5 public QA dependencies are downloaded at image-build time from their
pinned Hugging Face revisions.

## Publisher: one-time setup

Run this on the computer that already has the trained models:

```powershell
git lfs install
git add .gitattributes .gitignore Dockerfile README.md configs/runtime_artifacts.json scripts/verify_runtime_artifacts.py docs/model_artifacts.md tests/test_runtime_artifacts.py
git add models/phase2/*.joblib models/phase3/text_cnn.pt models/phase4/bert/model.pt
git commit -m "Distribute trained runtime models with Git LFS"
git push
```

Check that the six model paths appear in:

```powershell
git lfs ls-files
```

Do not commit a cloud API token or Hugging Face token. These artifacts are the
project's own trained checkpoints and do not need an authentication token.

## User: after cloning

Install Git LFS if `git lfs version` does not work. On Windows it is normally
included with Git for Windows; otherwise install it from <https://git-lfs.com/>.

```powershell
git lfs install
git pull
git lfs pull
python scripts/verify_runtime_artifacts.py
docker compose up -d --build
```

Open <http://127.0.0.1:7860>. The first complete build also downloads pinned
public QA models and installs Python packages, so it needs internet access and can
take several minutes. Later builds reuse Docker layers and the runtime volume.

If the preflight says **Git LFS pointer only**, `git lfs pull` did not download the
binary object. Check internet access, repository LFS quota/access, and rerun it.
If it says **missing**, the project owner has not yet committed/pushed that model
through LFS, or the clone is not on the updated branch.

## Private file-transfer alternative

Instead of Git LFS, the owner can securely send the `models` directory to the
other computer. Preserve these exact relative paths, run the verification script,
and then build Docker. Do not retrain simply to resolve a missing-file error.
