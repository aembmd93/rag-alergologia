"""
Ingesta de documentos hacia Pinecone.

Uso:
    python ingest.py                  # procesa ./docs/
    python ingest.py --docs ./ruta    # procesa ruta personalizada

Requisitos de sistema:
    - tesseract-ocr instalado (apt-get / brew install tesseract)
    - Variables de entorno en .env (ver .env.example)
"""

import argparse
import base64
import hashlib
import io
import os
import time
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
from google import genai
from google.genai import types as genai_types
from PIL import Image
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ── Configuración ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME      = os.getenv("PINECONE_INDEX_NAME", "rag-alergologia")
PINECONE_REGION = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

EMBED_MODEL      = "gemini-embedding-2"
EMBED_DIM        = 3072

# 512 tokens × ~4 chars/token; overlap 64 tokens
CHUNK_CHARS      = 512 * 4
OVERLAP_CHARS    = 64  * 4
MIN_TEXT_OCR     = 50           # umbral de chars para activar OCR

RENDER_DPI       = 150
IMG_MAX_B64_KB   = 15           # presupuesto de imagen; con 3072-dim vectors limite Pinecone es 4 MB por lote
BATCH_SIZE       = 20           # vectores por upsert (~1.2 MB/lote con imagen+vector)
EMBED_SLEEP      = 0.5          # segundos entre llamadas (~120 RPM, seguro en free tier)
EMBED_MAX_RETRY  = 6            # reintentos ante 429
EMBED_RETRY_BASE = 60           # segundos de espera inicial ante 429 (se duplica cada intento)

IMAGE_EXTS    = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}
MARKDOWN_EXTS = {".md", ".markdown"}


# ── Clientes ───────────────────────────────────────────────────────────────────

_genai_client: genai.Client | None = None


def _setup_genai():
    global _genai_client
    _genai_client = genai.Client(api_key=GOOGLE_API_KEY)


def _setup_pinecone() -> object:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME in existing:
        current_dim = pc.describe_index(INDEX_NAME).dimension
        if current_dim != EMBED_DIM:
            print(
                f"[Pinecone] Índice '{INDEX_NAME}' tiene dimensión {current_dim} "
                f"pero el modelo requiere {EMBED_DIM}. Eliminando y recreando…"
            )
            pc.delete_index(INDEX_NAME)
            existing = []   # forzar creación abajo

    if INDEX_NAME not in existing:
        print(f"[Pinecone] Creando índice '{INDEX_NAME}' ({EMBED_DIM} dims) en '{PINECONE_REGION}'…")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=PINECONE_REGION),
        )
        for _ in range(60):
            if pc.describe_index(INDEX_NAME).status.get("ready"):
                break
            time.sleep(2)
        print("[Pinecone] Índice listo.")

    return pc.Index(INDEX_NAME)


# ── Embeddings ─────────────────────────────────────────────────────────────────

def _embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    delay = EMBED_RETRY_BASE
    for attempt in range(EMBED_MAX_RETRY):
        try:
            response = _genai_client.models.embed_content(
                model=EMBED_MODEL,
                contents=text,
                config=genai_types.EmbedContentConfig(task_type=task_type),
            )
            return response.embeddings[0].values
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                if attempt < EMBED_MAX_RETRY - 1:
                    print(f"\n    [429] Rate limit — esperando {delay}s (intento {attempt+1}/{EMBED_MAX_RETRY})…")
                    time.sleep(delay)
                    delay = min(delay * 2, 600)  # backoff exponencial, máximo 10 min
                else:
                    raise
            else:
                raise
    return []


# ── Chunking ───────────────────────────────────────────────────────────────────

def _chunk(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += CHUNK_CHARS - OVERLAP_CHARS
    return chunks


# ── Imagen → base64 ────────────────────────────────────────────────────────────

def _compress_to_b64(img: Image.Image, max_kb: int = IMG_MAX_B64_KB) -> str:
    """Convierte una imagen PIL a JPEG base64, comprimida para caber en max_kb.

    Intenta progresivamente: calidad reducida → mitad de tamaño → cuarto de tamaño.
    Retorna cadena vacía si ningún intento cabe (páginas de ruido puro, extremo teórico).
    """
    img_gray = img.convert("L")
    limit = max_kb * 1024

    def _try(image: Image.Image, qualities=(70, 50, 35, 20)) -> str | None:
        for q in qualities:
            buf = io.BytesIO()
            image.save(buf, "JPEG", quality=q, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode()
            if len(b64) <= limit:
                return b64
        return None

    result = _try(img_gray)
    if result:
        return result

    # 1/2 tamaño
    half = img_gray.resize((img_gray.width // 2, img_gray.height // 2), Image.LANCZOS)
    result = _try(half, (25, 15))
    if result:
        return result

    # 1/4 tamaño
    quarter = img_gray.resize((img_gray.width // 4, img_gray.height // 4), Image.LANCZOS)
    result = _try(quarter, (20, 10))
    if result:
        return result

    return ""  # sin imagen para esta página; no bloqueará la ingesta


def _page_to_b64(page: fitz.Page) -> str:
    mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
    img = Image.frombytes("L", [pix.width, pix.height], pix.samples)
    return _compress_to_b64(img)


# ── Extracción de texto ────────────────────────────────────────────────────────

def _extract_text(page: fitz.Page) -> str:
    text = page.get_text("text")
    if len(text.strip()) < MIN_TEXT_OCR:
        # Renderizar la página a mayor resolución para mejor OCR
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        try:
            text = pytesseract.image_to_string(img, lang="spa+eng")
        except Exception as exc:
            print(f"    [OCR] Falló: {exc}")
            text = ""
    return text.strip()


# ── IDs deterministas ──────────────────────────────────────────────────────────

def _vid(filename: str, page: int, chunk: int) -> str:
    raw = f"{filename}|p{page}|c{chunk}"
    return hashlib.md5(raw.encode()).hexdigest()


# ── Upsert por lote ────────────────────────────────────────────────────────────

def _flush(index, buf: list):
    if buf:
        index.upsert(vectors=buf)


# ── Procesamiento de archivos ──────────────────────────────────────────────────

def process_pdf(path: Path, index) -> int:
    """Procesa un PDF página a página y sube vectores a Pinecone."""
    filename = path.name
    doc = fitz.open(str(path))
    total_pages = len(doc)
    buf, total_vecs = [], 0

    print(f"  [PDF] {filename} — {total_pages} páginas")
    for pnum, page in enumerate(doc, start=1):
        print(f"    página {pnum}/{total_pages}", end="\r")
        text = _extract_text(page)
        if not text:
            continue

        b64 = _page_to_b64(page)
        chunks = _chunk(text)

        for ci, chunk in enumerate(chunks):
            try:
                emb = _embed(chunk)
                time.sleep(EMBED_SLEEP)
            except Exception as exc:
                print(f"\n    [Embed] p{pnum} c{ci}: {exc}")
                continue

            buf.append({
                "id": _vid(filename, pnum, ci),
                "values": emb,
                "metadata": {
                    "filename":     filename,
                    "page_number":  pnum,
                    "text_chunk":   chunk[:2000],
                    "image_base64": b64,
                },
            })

            if len(buf) >= BATCH_SIZE:
                _flush(index, buf)
                total_vecs += len(buf)
                buf = []

    _flush(index, buf)
    total_vecs += len(buf)
    doc.close()
    print(f"\n    → {total_vecs} vectores subidos")
    return total_vecs


def process_image(path: Path, index) -> int:
    """Aplica OCR a una imagen y sube vectores a Pinecone."""
    filename = path.name
    print(f"  [IMG] {filename}")
    img = Image.open(str(path))

    try:
        text = pytesseract.image_to_string(img, lang="spa+eng").strip()
    except Exception as exc:
        print(f"    [OCR] Falló: {exc}")
        return 0

    if not text:
        print(f"    Sin texto extraído.")
        return 0

    b64 = _compress_to_b64(img)
    chunks = _chunk(text)
    buf = []

    for ci, chunk in enumerate(chunks):
        try:
            emb = _embed(chunk)
            time.sleep(EMBED_SLEEP)
        except Exception as exc:
            print(f"    [Embed] c{ci}: {exc}")
            continue

        buf.append({
            "id": _vid(filename, 1, ci),
            "values": emb,
            "metadata": {
                "filename":     filename,
                "page_number":  1,
                "text_chunk":   chunk[:2000],
                "image_base64": b64,
            },
        })

    _flush(index, buf)
    print(f"    → {len(buf)} vectores subidos")
    return len(buf)


def process_markdown(path: Path, index) -> int:
    """Lee un archivo Markdown, lo chunkea y sube vectores a Pinecone."""
    filename = path.name
    print(f"  [MD]  {filename}")

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        print(f"    Archivo vacío.")
        return 0

    chunks = _chunk(text)
    buf = []

    for ci, chunk in enumerate(chunks):
        try:
            emb = _embed(chunk)
            time.sleep(EMBED_SLEEP)
        except Exception as exc:
            print(f"    [Embed] c{ci}: {exc}")
            continue

        buf.append({
            "id": _vid(filename, 1, ci),
            "values": emb,
            "metadata": {
                "filename":     filename,
                "page_number":  1,
                "text_chunk":   chunk[:2000],
                "image_base64": "",   # sin página renderizada
            },
        })

        if len(buf) >= BATCH_SIZE:
            _flush(index, buf)
            buf = []

    _flush(index, buf)
    total = len(buf)
    print(f"    → {total} vectores subidos")
    return total


# ── Punto de entrada principal ─────────────────────────────────────────────────

def ingest_folder(docs_path: str = "./docs") -> dict:
    """
    Procesa todos los PDFs, imágenes y Markdown en docs_path y los sube a Pinecone.

    Retorna dict con claves: pdfs, images, markdowns, vectors.
    Puede importarse desde app.py para disparar re-indexación desde la UI.
    """
    path = Path(docs_path)
    path.mkdir(parents=True, exist_ok=True)

    pdfs      = sorted(path.glob("*.pdf"))
    images    = sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    markdowns = sorted(f for f in path.iterdir() if f.suffix.lower() in MARKDOWN_EXTS)

    total_files = len(pdfs) + len(images) + len(markdowns)
    print(f"\n[Ingest] {len(pdfs)} PDFs, {len(images)} imágenes, {len(markdowns)} Markdown en '{docs_path}'")
    if not total_files:
        print("[Ingest] Carpeta vacía — nada que procesar.")
        return {"pdfs": 0, "images": 0, "markdowns": 0, "vectors": 0}

    _setup_genai()
    index = _setup_pinecone()

    total = 0
    for p in pdfs:
        total += process_pdf(p, index)
    for img in images:
        total += process_image(img, index)
    for md in markdowns:
        total += process_markdown(md, index)

    stats = {"pdfs": len(pdfs), "images": len(images), "markdowns": len(markdowns), "vectors": total}
    print(f"\n[Ingest] Completado: {stats}\n")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingesta documentos en Pinecone")
    ap.add_argument("--docs", default="./docs", help="Ruta a la carpeta de documentos")
    args = ap.parse_args()
    ingest_folder(args.docs)
