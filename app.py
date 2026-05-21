"""
RAG Alergología — Dashboard Streamlit

Ejecución local:
    streamlit run app.py

Despliegue:
    Render free tier (render.yaml) o Streamlit Community Cloud (packages.txt)
"""

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from pinecone import Pinecone

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY   = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "rag-alergologia")

EMBED_MODEL = "gemini-embedding-2"
LLM_MODEL   = "gemini-2.5-flash"
TOP_K       = 5
MIN_SCORE   = 0.75
NO_INFO_MSG = "No encontré información suficiente en los documentos cargados."

DOCS_PATH  = Path("./docs")
DOCS_PATH.mkdir(exist_ok=True)

ACCEPTED_EXTS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp", ".md", ".markdown"}

ADMIN_EMAIL    = os.getenv("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")


# ── Clientes cacheados ─────────────────────────────────────────────────────────

@st.cache_resource
def _get_genai_client() -> genai.Client:
    return genai.Client(api_key=GOOGLE_API_KEY)


@st.cache_resource
def _get_index():
    """Conexión al índice de Pinecone. Se reutiliza entre reruns."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        names = [i.name for i in pc.list_indexes()]
        if INDEX_NAME not in names:
            return None
        return pc.Index(INDEX_NAME)
    except Exception:
        return None


# ── Funciones RAG ──────────────────────────────────────────────────────────────

def _embed_query(text: str) -> list:
    client = _get_genai_client()
    response = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return response.embeddings[0].values


def _retrieve(query: str) -> list[dict]:
    index = _get_index()
    if index is None:
        return []
    try:
        vec = _embed_query(query)
        res = index.query(vector=vec, top_k=TOP_K, include_metadata=True)
        return [
            {
                "score":        m.score,
                "filename":     m.metadata.get("filename", "—"),
                "page_number":  int(m.metadata.get("page_number", 0)),
                "text_chunk":   m.metadata.get("text_chunk", ""),
                "image_base64": m.metadata.get("image_base64", ""),
            }
            for m in res.matches
        ]
    except Exception as exc:
        st.error(f"Error al consultar Pinecone: {exc}")
        return []


def _build_prompt(query: str, chunks: list[dict]) -> str:
    ctx = "\n\n---\n\n".join(
        f"[Fuente {i}: {c['filename']}, página {c['page_number']}]\n{c['text_chunk']}"
        for i, c in enumerate(chunks, 1)
    )
    return (
        "Eres un asistente médico especializado en alergología. "
        "Responde SIEMPRE en español, basándote ÚNICAMENTE en el contexto proporcionado.\n\n"
        "Instrucciones:\n"
        "- Cita las fuentes que uses con el formato [Fuente N] dentro del texto.\n"
        "- Si la información no aparece en el contexto, indícalo explícitamente.\n"
        "- Usa terminología médica precisa y no inventes datos.\n\n"
        f"Contexto:\n{ctx}\n\n"
        f"Pregunta: {query}\n\n"
        "Respuesta:"
    )


def _generate(query: str, chunks: list[dict]) -> str:
    client = _get_genai_client()
    try:
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=_build_prompt(query, chunks),
        )
        return response.text
    except Exception as exc:
        return f"Error al generar la respuesta: {exc}"


def _rag(query: str) -> tuple[str, list[dict]]:
    """Devuelve (texto_respuesta, lista_de_fuentes)."""
    chunks = _retrieve(query)
    if not chunks or chunks[0]["score"] < MIN_SCORE:
        return NO_INFO_MSG, []
    return _generate(query, chunks), chunks


# ── Componente de fuentes ──────────────────────────────────────────────────────

def _show_sources(sources: list[dict]):
    if not sources:
        return

    st.markdown("**Fuentes consultadas:**")
    st.dataframe(
        pd.DataFrame([
            {
                "Documento": s["filename"],
                "Página":    s["page_number"],
                "Score":     round(s["score"], 4),
            }
            for s in sources
        ]),
        use_container_width=True,
        hide_index=True,
    )

    for s in sources:
        header = (
            f"📄 {s['filename']} — p. {s['page_number']}  "
            f"(score: {s['score']:.3f})"
        )
        with st.expander(header):
            col_img, col_txt = st.columns([1, 2])

            with col_img:
                if s.get("image_base64"):
                    st.image(
                        f"data:image/jpeg;base64,{s['image_base64']}",
                        caption=f"Página {s['page_number']}",
                        use_container_width=True,
                    )
                else:
                    st.warning("Imagen no disponible para esta fuente.")

            with col_txt:
                st.markdown("**Fragmento relevante:**")
                preview = s["text_chunk"]
                if len(preview) > 600:
                    preview = preview[:600] + "…"
                st.caption(preview)


# ── Layout de página ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Alergología",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Autenticación ──────────────────────────────────────────────────────────────

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col_center = st.columns([1, 1, 1])[1]
    with col_center:
        st.markdown("## 🏥 RAG Alergología")
        st.markdown("Ingresa tus credenciales para continuar.")
        email    = st.text_input("Correo", placeholder="usuario@ejemplo.com")
        password = st.text_input("Contraseña", type="password")
        if st.button("Ingresar", type="primary", use_container_width=True):
            if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Correo o contraseña incorrectos.")
    st.stop()

# Estado de sesión
if "messages" not in st.session_state:
    st.session_state.messages = []  # lista de {role, content, sources?}

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📋 Documentos")
    st.caption(f"Sesión: {ADMIN_EMAIL}")
    if st.button("Cerrar sesión", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()
    st.markdown("---")

    # Subida de archivos
    uploaded = st.file_uploader(
        "Subir documentos (PDF, imágenes o Markdown)",
        type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp", "md", "markdown"],
        accept_multiple_files=True,
        key="uploader",
        help="Los archivos se guardan en ./docs/ y deben re-indexarse manualmente.",
    )

    if uploaded:
        saved = []
        for f in uploaded:
            dest = DOCS_PATH / f.name
            dest.write_bytes(f.read())
            saved.append(f.name)
        if saved:
            st.success(f"Guardados: {', '.join(saved)}")

    # Botón de indexación
    if st.button("🔄 Indexar / Re-indexar", type="primary", use_container_width=True):
        with st.spinner("Procesando documentos… (puede tardar varios minutos)"):
            try:
                from ingest import ingest_folder   # importación diferida
                stats = ingest_folder(str(DOCS_PATH))
                st.success(
                    f"✅ Completado: **{stats['pdfs']}** PDFs · "
                    f"**{stats['images']}** imágenes · "
                    f"**{stats.get('markdowns', 0)}** Markdown · "
                    f"**{stats['vectors']}** vectores"
                )
                _get_index.clear()  # refrescar conexión al índice actualizado
            except Exception as exc:
                st.error(f"Error durante la indexación: {exc}")

    st.markdown("---")

    # Listado de documentos en ./docs/
    doc_files = sorted(
        f for f in DOCS_PATH.iterdir()
        if f.suffix.lower() in ACCEPTED_EXTS
    )
    if doc_files:
        st.markdown(f"**{len(doc_files)} archivo(s) en `./docs/`:**")
        for f in doc_files:
            st.text(f"• {f.name}")
    else:
        st.info("La carpeta `./docs/` está vacía.\nSube documentos y haz clic en **Indexar**.")

    st.markdown("---")

    # Estado del índice
    idx = _get_index()
    if idx is not None:
        st.success("✅ Índice Pinecone conectado")
    else:
        st.warning("⚠️ Índice no encontrado.\nIndexa documentos para crearlo.")

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Área principal ─────────────────────────────────────────────────────────────

st.title("🏥 Asistente de Alergología")
st.caption(
    "Consulta médica basada en los documentos indexados. "
    "Sube archivos y haz clic en **Indexar** desde el panel lateral antes de preguntar."
)

# Renderizar historial de conversación
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _show_sources(msg.get("sources", []))

# Entrada de chat
if prompt := st.chat_input("Escribe tu pregunta sobre alergología…"):
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generar y mostrar respuesta
    with st.chat_message("assistant"):
        with st.spinner("Buscando en documentos…"):
            response_text, sources = _rag(prompt)
        st.markdown(response_text)
        _show_sources(sources)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": response_text,
        "sources": sources,
    })
