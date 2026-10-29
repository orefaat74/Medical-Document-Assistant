"""Capstone web interface backed by the Medical Document Assistant FastAPI service."""

import os
from pathlib import Path

import streamlit as st
from medical_doc_assistant.web_workspace import render_workspace

ROOT = Path(__file__).resolve().parent
API_URL = os.getenv("MEDICAL_API_URL", "http://127.0.0.1:8000").rstrip("/")
st.set_page_config(page_title="Medical Document Assistant", page_icon="📄", layout="wide")
st.markdown("""<style>
.block-container {max-width:1200px; padding-top:2rem;}
h1, h2, h3 {color:#263450;}
[data-testid="stMetric"] {background:#f4f6fa; padding:16px; border-radius:12px;}
.stButton > button[kind="primary"] {background:#5e5aa1; border-color:#5e5aa1;}
</style>""", unsafe_allow_html=True)


render_workspace(ROOT, API_URL)
