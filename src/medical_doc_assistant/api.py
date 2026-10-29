"""FastAPI endpoints for document upload, classification, extraction, and grounded QA."""

import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Response
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from medical_doc_assistant.ingestion import extract_source_fields, filename_only, normalize_document, read_document_details
from medical_doc_assistant.serving import ApplicationRuntime, DISCLAIMER, PROJECT_ROOT


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=200000)


class ClassificationInput(TextInput):
    model: str | None = None


class DocumentInput(ClassificationInput):
    source_name: str = Field(default="pasted document", max_length=200)


class QuestionInput(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=1000)


def _call(function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error).strip("'")) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def create_app(runtime: ApplicationRuntime | None = None) -> FastAPI:
    service = runtime or ApplicationRuntime(Path(os.getenv("MEDICAL_PROJECT_ROOT", str(PROJECT_ROOT))))
    app = FastAPI(title="Medical Document Assistant", version="1.0.0",
                  description="Classify English leaflet text, extract source sections, and ask document-grounded questions. " + DISCLAIMER)
    app.state.runtime = service

    def download(function: Any, *args: Any) -> Response:
        content, media_type, filename = _call(function, *args)
        return Response(content=content, media_type=media_type,
                        headers={"Content-Disposition": "attachment; filename*=UTF-8''" + quote(filename)})

    @app.get("/health")
    def health() -> dict[str, Any]:
        models = service.available_models()
        return {"status": "ok", "default_classifier": service.settings["default_classifier"],
                 "default_model_available": any(model["id"] == service.settings["default_classifier"] and model["available"] for model in models),
                 "evaluated_language": "en", "feature_version": 4, "disclaimer": DISCLAIMER}

    @app.get("/models")
    def models() -> dict[str, Any]:
        return {"models": service.available_models(), "disclaimer": DISCLAIMER}

    @app.post("/classify")
    def classify(payload: ClassificationInput) -> dict[str, Any]:
        return _call(service.classify, payload.text, payload.model)

    @app.post("/extract")
    def extract(payload: TextInput) -> dict[str, Any]:
        text = _call(normalize_document, payload.text, service.settings)
        return {**extract_source_fields(text), "disclaimer": DISCLAIMER}

    @app.post("/documents")
    def documents(payload: DocumentInput) -> dict[str, Any]:
        return _call(service.add_document, payload.text, payload.source_name, payload.model)

    @app.post("/documents/upload")
    async def upload(file: UploadFile = File(...), model: str | None = Form(None)) -> dict[str, Any]:
        content = await file.read(service.settings["maximum_upload_bytes"] + 1)
        await file.close()
        name = filename_only(file.filename or "uploaded document")
        details = await run_in_threadpool(_call, read_document_details, name, content, service.settings)
        return await run_in_threadpool(_call, service.add_document, details["text"], name, model, details, content)

    @app.get("/documents/{document_id}")
    def get_document(document_id: str) -> dict[str, Any]:
        return _call(service.document_view, document_id)

    @app.delete("/documents/{document_id}")
    def delete_document(document_id: str) -> dict[str, Any]:
        if not service.delete_document(document_id):
            raise HTTPException(status_code=404, detail="Document not found or already cleared.")
        return {"deleted": True, "document_id": document_id}

    @app.post("/ask")
    def ask(payload: QuestionInput) -> dict[str, Any]:
        return _call(service.ask, payload.document_id, payload.question)

    @app.get("/documents/{document_id}/original")
    def original(document_id: str) -> Response:
        return download(service.original_document, document_id)

    return app


app = create_app()
