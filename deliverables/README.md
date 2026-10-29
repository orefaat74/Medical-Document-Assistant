# Final handoff

- `Final_Presentation.pptx` — eight editable slides with speaker notes.
- `Final_Presentation.pdf` — PDF presentation copy.
- `Speaker_Notes.md` — 741 words, roughly 6–7 minutes including the live example.
- `Source_Code_Handoff.zip` — source, tests, configs, docs, deployment files, and plan.
- `manifest.json` — generated material counts and temporary demo URL.

Trained models and frozen datasets are in the project folder; large runtime assets
are not embedded in the ordinary source ZIP. Docker copies the trained classifiers
and downloads pinned public QA/embedding models during build.

The agreed finish is the temporary HTTPS demo plus source/deployment handoff.
Persistent Hugging Face hosting can be published with the supplied script after a
local account login. GitHub publication was not performed.
