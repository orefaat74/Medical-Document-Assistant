"""Read supported document files and extract source sections with exact offsets."""

import io
import re
import zipfile
from pathlib import Path
from typing import Any

from docx import Document
from docx.table import Table
from pypdf import PdfReader

from medical_doc_assistant.cleaning import FACT_FIELDS, SECTION_FIELDS, clean_text

ALIASES = {field.replace("_", " "): field for field in SECTION_FIELDS}
ALIASES.update({
    "dosage": "dosage_and_administration", "dosing": "dosage_and_administration", "directions": "dosage_and_administration",
    "side effects": "adverse_reactions", "uses": "indications_and_usage", "indications": "indications_and_usage",
    "warnings and precautions": "warnings_and_cautions", "warnings and cautions": "warnings_and_cautions",
    "pregnancy and breastfeeding": "pregnancy_or_breast_feeding", "pregnancy and breast feeding": "pregnancy_or_breast_feeding",
    "pregnancy and breast-feeding": "pregnancy_or_breast_feeding", "pregnancy or breastfeeding": "pregnancy_or_breast_feeding",
    "pregnancy or breast-feeding": "pregnancy_or_breast_feeding", "breastfeeding": "nursing_mothers",
    "breast feeding": "nursing_mothers", "breast-feeding": "nursing_mothers",
    "storage": "storage_and_handling", "storage and handling": "storage_and_handling",
    "price": "price", "retail price": "price",
    "active ingredient": "active_ingredient", "active ingredients": "active_ingredient",
    "inactive ingredients": "inactive_ingredient", "medicine name": "medicine_name", "brand name": "medicine_name",
    "interactions": "drug_interactions",
})
HEADINGS = re.compile(r"(?im)^[ \t]*(?:\d+(?:\.\d+)*\.?[ \t]+)?(?P<title>" +
                      "|".join(re.escape(title) for title in sorted(ALIASES, key=len, reverse=True)) +
                      r")[ \t]*(?::[ \t]*|(?=$))")


def filename_only(filename: str) -> str:
    return re.split(r"[/\\]", filename)[-1] or "uploaded document"


def read_document_file(filename: str, content: bytes, settings: dict[str, Any]) -> str:
    return read_document_details(filename, content, settings)["text"]


def _join_source_blocks(pieces: list[tuple[str, dict[str, Any]]], settings: dict[str, Any], file_type: str) -> dict[str, Any]:
    blocks, parts, offset = [], [], 0
    for raw, locator in pieces:
        text = clean_text(raw)
        if not text:
            continue
        if parts:
            offset += 1
        blocks.append({**locator, "char_start": offset, "char_end": offset + len(text)})
        parts.append(text)
        offset += len(text)
    text = normalize_document("\n".join(parts), settings)
    return {"text": text, "source_blocks": blocks, "file_type": file_type}


def text_document_details(text: str, settings: dict[str, Any]) -> dict[str, Any]:
    canonical = normalize_document(text, settings)
    return _join_source_blocks([(line, {"kind": "text_line", "line": index + 1})
                                for index, line in enumerate(canonical.splitlines())], settings, "text")


def read_document_details(filename: str, content: bytes, settings: dict[str, Any]) -> dict[str, Any]:
    if len(content) > settings["maximum_upload_bytes"]:
        raise ValueError("The upload exceeds the 10 MB document limit.")
    extension = Path(filename_only(filename)).suffix.lower()
    if extension not in settings["supported_file_types"]:
        raise ValueError("Use a text-based PDF, TXT/MD, HTML, or DOCX document.")
    try:
        if extension == ".pdf":
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted:
                raise ValueError("Use an unencrypted PDF document.")
            if len(reader.pages) > settings["maximum_pdf_pages"]:
                raise ValueError("The PDF exceeds the 50-page document limit.")
            pieces = [(page.extract_text(extraction_mode="plain") or "", {"kind": "pdf_page", "page": index + 1})
                      for index, page in enumerate(reader.pages)]
            if not any(text.strip() for text, _ in pieces):
                raise ValueError("This PDF has no extractable text. Paste the text or upload a text-based PDF.")
        elif extension == ".docx":
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                if archive.getinfo("word/document.xml").file_size > 10 * 1024 * 1024:
                    raise ValueError("The DOCX text is too large for this demo.")
            document = Document(io.BytesIO(content))
            pieces, paragraph_number, table_number = [], 0, 0
            for block in document.iter_inner_content():
                if isinstance(block, Table):
                    table_number += 1
                    pieces.extend((" | ".join(cell.text for cell in row.cells),
                                   {"kind": "docx_table_row", "table": table_number, "row": index + 1})
                                  for index, row in enumerate(block.rows))
                else:
                    paragraph_number += 1
                    pieces.append((block.text, {"kind": "docx_paragraph", "paragraph": paragraph_number}))
        else:
            text = normalize_document(content.decode("utf-8-sig"), settings)
            pieces = [(line, {"kind": "text_line", "line": index + 1}) for index, line in enumerate(text.splitlines())]
    except ValueError:
        raise
    except Exception as error:
        raise ValueError("The document could not be read. Check its file type and text content.") from error
    return _join_source_blocks(pieces, settings, extension.lstrip("."))


def normalize_document(text: str, settings: dict[str, Any]) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Upload or paste a nonempty document.")
    if len(text) > settings["maximum_document_characters"]:
        raise ValueError("The document exceeds the 200,000-character text limit.")
    normalized = clean_text(text)
    if not normalized:
        raise ValueError("No readable document text was found.")
    return normalized


def source_sections(text: str) -> list[dict[str, Any]]:
    matches = list(HEADINGS.finditer(text))
    sections = []
    for index, match in enumerate(matches):
        first = match.end()
        last = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        while first < last and text[first].isspace():
            first += 1
        while last > first and text[last - 1].isspace():
            last -= 1
        if last > first:
            sections.append({"field": ALIASES[match.group("title").lower()], "heading": match.group("title"),
                             "text": text[first:last], "char_start": first, "char_end": last})
    return sections


def extract_source_fields(text: str) -> dict[str, Any]:
    sections = source_sections(text)
    mapping = {"indications": FACT_FIELDS["indications_text"], "dosage": FACT_FIELDS["dosage_text"],
               "warnings": FACT_FIELDS["warnings_text"], "contraindications": FACT_FIELDS["contraindications_text"],
               "side_effects": FACT_FIELDS["side_effects_text"], "interactions": ("drug_interactions", "ask_doctor_or_pharmacist")}
    fields = {}
    for name, eligible in mapping.items():
        passages = [section for section in sections if section["field"] in eligible]
        unique = list(dict.fromkeys(passage["text"] for passage in passages))
        fields[name] = {"present": bool(passages), "text": "\n\n".join(unique), "passages": passages}
    return {"fields": fields, "sections": sections, "method": "source-heading section extraction", "section_count": len(sections)}
