"""Streamlit UI for overviews, follow-ups, dynamic suggestions, and source navigation."""

import base64
from pathlib import Path
from typing import Any

import requests
import streamlit as st
import streamlit.components.v1 as components

from medical_doc_assistant.source_navigation import highlighted_text, location_label

CLASS_NAMES = {"antibacterial": "Antibacterial", "anti_inflammatory_nsaid": "Anti-inflammatory (NSAID)",
               "corticosteroid": "Corticosteroid", "antifungal": "Antifungal", "acid_reducer": "Acid reducer",
               "antihypertensive": "Antihypertensive"}
FIELD_NAMES = {"indications": "Indications / uses", "dosage": "Dosage and administration", "warnings": "Warnings",
               "contraindications": "Contraindications", "side_effects": "Side effects", "interactions": "Interactions"}


def request(base: str, method: str, path: str, *, binary: bool = False, **kwargs: Any):
    try:
        response = requests.request(method, base + path, timeout=(10, 240), **kwargs)
    except requests.RequestException as error:
        raise RuntimeError("The backend is unavailable. Start the API service and try again.") from error
    if not response.ok:
        try:
            message = response.json().get("detail", response.text)
        except ValueError:
            message = response.text
        raise RuntimeError(str(message))
    return response.content if binary else response.json()


@st.cache_data(ttl=15)
def backend_models(base: str):
    return request(base, "GET", "/models")["models"]


def render_workspace(root: Path, base: str) -> None:
    def api(method, path, **kwargs):
        return request(base, method, path, **kwargs)

    def accept_document(document):
        previous = st.session_state.get("document")
        if previous:
            try:
                api("DELETE", "/documents/" + previous["document_id"])
            except RuntimeError:
                pass
        st.session_state.document = document
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.session_state.source_focus = []
        st.session_state.workspace_document_view = "Overview"
        st.session_state.workspace_source_location = 0
        st.session_state.pop("original_pdf", None)

    def jump_to_source(citation):
        document = st.session_state.document
        st.session_state.workspace_document_view = "Document text"
        st.session_state.source_focus = [(citation["char_start"], citation["char_end"])]
        st.session_state.workspace_source_location = next((index for index, block in enumerate(document["source_blocks"])
                                                          if block["char_start"] < citation["char_end"] and block["char_end"] > citation["char_start"]), 0)
        st.session_state.scroll_source = True

    def evidence(citations, prefix, expand=True):
        for index, citation in enumerate(citations):
            if expand:
                container = st.expander(citation.get("source_label", "Source evidence"))
            else:
                st.caption(citation.get("source_label", "Source evidence"))
                container = st.container()
            with container:
                st.text(citation["quote"])
                st.caption(f"Source characters {citation['char_start']}–{citation['char_end']}")
                st.button("Jump to source", key=prefix + "_source_" + str(index), on_click=jump_to_source, args=(citation,))

    for key, default in (("messages", []), ("pending_question", None), ("source_focus", [])):
        if key not in st.session_state:
            st.session_state[key] = default
    if st.session_state.get("document") and "overview" not in st.session_state.document:
        previous = st.session_state.document
        try:
            accept_document(api("POST", "/documents", json={"text": previous["text"], "source_name": previous["source_name"]}))
        except RuntimeError:
            st.session_state.pop("document", None)

    st.title("Medical Document Assistant")
    st.caption("Understand a document. Ask follow-up questions. Verify answers at their source.")
    st.info("Informational only — not medical advice. This demo evaluates English text and quotes the document you supply.")
    with st.sidebar:
        st.header("Document workspace")
        try:
            models = [model for model in backend_models(base) if model["available"]]
            names = {model["id"]: model["name"] for model in models}
            choices = list(names)
            classifier = st.selectbox("Classifier", choices, index=choices.index("text_cnn") if "text_cnn" in choices else 0,
                                     format_func=lambda value: names[value])
        except (RuntimeError, ValueError):
            classifier = "text_cnn"
            st.warning("API connection not ready. The default classifier is TextCNN.")
        st.caption("TextCNN is the validation-selected default. Categories are limited to the six trained leaflet classes.")
        if st.button("Use example leaflet", use_container_width=True):
            try:
                with st.spinner("Analyzing the source example..."):
                    text = (root / "data" / "examples" / "phase2_demo_leaflet.txt").read_text(encoding="utf-8")
                    accept_document(api("POST", "/documents", json={"text": text, "source_name": "FDA example leaflet", "model": classifier}))
                st.rerun()
            except (RuntimeError, OSError) as error:
                st.error(str(error))
        if st.session_state.get("document") and st.button("Clear document", use_container_width=True):
            try:
                api("DELETE", "/documents/" + st.session_state.document["document_id"])
            except RuntimeError:
                pass
            st.session_state.pop("document", None)
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption("10 MB file limit · 50 PDF pages · 200,000 text characters")

    with st.expander("Upload or paste a document", expanded=not bool(st.session_state.get("document"))):
        mode = st.radio("Input method", ["Upload file", "Paste text"], horizontal=True)
        with st.form("document_form"):
            if mode == "Upload file":
                uploaded = st.file_uploader("Choose a document", type=["pdf", "txt", "md", "html", "htm", "docx"])
                pasted, title = "", "pasted document"
            else:
                uploaded = None
                title = st.text_input("Document title", value="Pasted leaflet")
                pasted = st.text_area("Document text", height=220)
            submitted = st.form_submit_button("Analyze document", type="primary", use_container_width=True)
        if submitted:
            try:
                with st.spinner("Reading the source, building its overview, and checking its category..."):
                    if uploaded is not None:
                        document = api("POST", "/documents/upload", files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}, data={"model": classifier})
                    elif pasted.strip():
                        document = api("POST", "/documents", json={"text": pasted, "source_name": title, "model": classifier})
                    else:
                        raise RuntimeError("Choose a file or paste readable document text.")
                    accept_document(document)
                st.rerun()
            except RuntimeError as error:
                st.error(str(error))

    document = st.session_state.get("document")
    if not document:
        st.subheader("Start with a document")
        st.write("Upload a file, paste text, or use the real FDA example.")
        return
    classification = document["classification"]
    metrics = st.columns(3)
    metrics[0].metric("Source document", document["source_name"])
    metrics[1].metric("Predicted category", CLASS_NAMES.get(classification.get("predicted_class"), "English model only"))
    metrics[2].metric("Source words / sections", f"{document['words']:,} / {document['extraction']['section_count']}")
    st.caption("Classifier: " + classification.get("model_name", "unsupported language") + " · " + str(classification.get("latency_ms", "—")) + " ms")

    view = st.radio("Document view", ["Overview", "Source sections", "Document text", "Model comparison"],
                    horizontal=True, key="workspace_document_view")
    if view == "Overview":
        for issue_index, issue in enumerate(document["overview"]["consistency_issues"]):
            st.warning(issue["message"])
            evidence(issue["citations"], "consistency_" + str(issue_index))
        identity_columns = st.columns(3)
        for index, (name, field) in enumerate(document["overview"]["identity"].items()):
            with identity_columns[index % 3]:
                st.markdown("**" + name.replace("_", " ").title() + "**")
                st.write(field["text"])
                evidence(field["citations"], "identity_" + name)
        for name, field in document["overview"]["summary"].items():
            if field["present"]:
                with st.expander(name.replace("_", " ").title(), expanded=name in {"uses", "dosage"}):
                    st.write(field["text"])
                    evidence(field["citations"], "overview_" + name, expand=False)
    elif view == "Source sections":
        for name, field in document["extraction"]["fields"].items():
            with st.expander(FIELD_NAMES[name], expanded=name == "dosage"):
                st.text(field["text"] if field["present"] else "A dedicated source section was not found.")
    elif view == "Document text":
        st.markdown('<div id="document-source-focus"></div>', unsafe_allow_html=True)
        blocks = document["source_blocks"]
        selection = st.selectbox("Source location", range(len(blocks)), format_func=lambda index: location_label(blocks[index]), key="workspace_source_location")
        block = blocks[selection]
        st.markdown(highlighted_text(document["text"], block["char_start"], block["char_end"], st.session_state.source_focus), unsafe_allow_html=True)
        if st.session_state.pop("scroll_source", False):
            components.html("<script>window.parent.document.getElementById('document-source-focus')?.scrollIntoView({behavior:'smooth',block:'start'});</script>", height=0)
        if document["file_type"] == "pdf":
            if st.button("Open original PDF page"):
                st.session_state.original_pdf = api("GET", "/documents/" + document["document_id"] + "/original", binary=True)
            if st.session_state.get("original_pdf"):
                encoded = base64.b64encode(st.session_state.original_pdf).decode("ascii")
                components.html('<embed src="data:application/pdf;base64,' + encoded + '#page=' + str(block.get("page", 1)) + '" type="application/pdf" width="100%" height="600">', height=610)
        with st.expander("Complete extracted text"):
            st.text(document["text"])
        st.download_button("Download extracted text", document["text"], file_name="document_text.txt", mime="text/plain")
    else:
        try:
            import pandas as pd
            st.dataframe(pd.DataFrame([{"Model": model["name"], "Validation macro F1": model["validation_macro_f1"],
                                       "Test accuracy": model["test_accuracy"], "Test macro F1": model["test_macro_f1"]}
                                      for model in backend_models(base)]), hide_index=True, use_container_width=True)
        except RuntimeError as error:
            st.caption(str(error))

    st.divider()
    st.subheader("Ask this document")
    st.caption("Ask simply or follow up: “dosage”, “how often?”, “what about children?”, or “is it safe?” after pregnancy.")
    suggestions = document["suggestions"]
    for row_start in range(0, len(suggestions), 4):
        columns = st.columns(4)
        for column, suggestion in zip(columns, suggestions[row_start:row_start + 4]):
            if column.button(suggestion["label"], key="suggestion_" + suggestion["id"], use_container_width=True):
                st.session_state.pending_question = suggestion["question"]
    with st.expander("Test missing information"):
        st.caption("This checks whether a retail price is stated. A missing-price response is expected when it is absent.")
        if st.button("Price example"):
            st.session_state.pending_question = "What is the retail price stated in the document?"
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if message["role"] == "assistant":
                if message.get("context_used"):
                    st.caption("Interpreted as: " + message["resolved_question"])
                if message.get("missing_reason"):
                    st.caption(message["missing_reason"])
                evidence(message.get("citations", []), "chat_" + str(index))
                if message.get("latency_ms") is not None:
                    st.caption(str(message["latency_ms"]) + " ms · " + message.get("answer_mode", message.get("status", "")))
    typed = st.chat_input("Ask about this document, or follow up on the last answer")
    question = typed or st.session_state.pending_question
    if question:
        st.session_state.pending_question = None
        st.session_state.messages.append({"role": "user", "content": question})
        try:
            with st.spinner("Finding source evidence and checking the answer..."):
                response = api("POST", "/ask", json={"document_id": document["document_id"], "question": question})
            st.session_state.messages.append({**response, "role": "assistant", "content": response["answer"]})
        except RuntimeError as error:
            st.session_state.messages.append({"role": "assistant", "content": str(error), "citations": [], "status": "error"})
        st.rerun()
