---
title: Medical Document Assistant
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Medical Document Assistant

A source-quoted English document assistant with a FastAPI backend and Streamlit
interface. Upload or paste text, inspect source sections, and ask questions with
quoted evidence. Informational only; not medical advice.

The default classifier is the validation-selected TextCNN. All model comparison
figures describe the source-derived educational corpus, not clinical validation.

Model assets are downloaded from pinned public Hugging Face revisions during the
Docker build. Credits and scope are described in `docs/phase4_model_cards.md`.
