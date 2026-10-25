from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u202a-\u202e\ufeff]")

OBFUSCATION_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\[\s*at\s*\]", re.IGNORECASE), "@"),
    (re.compile(r"\(\s*at\s*\)", re.IGNORECASE), "@"),
    (re.compile(r"\{\s*at\s*\}", re.IGNORECASE), "@"),
    (re.compile(r"\[\s*arroba\s*\]", re.IGNORECASE), "@"),
    (re.compile(r"\(\s*arroba\s*\)", re.IGNORECASE), "@"),
    (re.compile(r"\{\s*arroba\s*\}", re.IGNORECASE), "@"),
    (re.compile(r"\[\s*dot\s*\]", re.IGNORECASE), "."),
    (re.compile(r"\(\s*dot\s*\)", re.IGNORECASE), "."),
    (re.compile(r"\{\s*dot\s*\}", re.IGNORECASE), "."),
    (re.compile(r"\[\s*punto\s*\]", re.IGNORECASE), "."),
    (re.compile(r"\(\s*punto\s*\)", re.IGNORECASE), "."),
    (re.compile(r"\{\s*punto\s*\}", re.IGNORECASE), "."),
    (re.compile(r"\barroba\b", re.IGNORECASE), "@"),
    # Tesseract suele confundir @ con el simbolo de copyright.
    (re.compile(r"[©®]"), "@"),
]

# Regex de candidatos laxos para smart retry / benchmark (seguro, sin relentizar)
LOOSE_EMAIL_RE = re.compile(
    r"[a-z0-9._%+\-]{2,}@[a-z0-9.\-]{2,}\.[a-z0-9]{2,}",
    re.IGNORECASE,
)
EMAIL_CUE_RE = re.compile(
    r"\b(correo|email|e-mail|mail|cv|curriculum|postular|postulate|enviar|envia|contacto)\b",
    re.IGNORECASE,
)

# Piezas compartidas para que la extraccion y la validacion final sean coherentes.
# Para OCR priorizamos precision: limitamos la parte local a caracteres que
# aparecen en emails reales. La lista RFC completa genera falsos positivos con
# simbolos decorativos, por ejemplo "|@dominio.com".
_LOCAL_CHARS = r"A-Za-z0-9._%+\-"

_LOCAL_PART = rf"[{_LOCAL_CHARS}]+(?:\.[{_LOCAL_CHARS}]+)*"
_LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?"
_TLD = r"[A-Za-z]{2,24}"
_SPACED_DOMAIN = rf"{_LABEL}(?:\s*\.\s*{_LABEL})*\s*\.\s*{_TLD}"
_FULL_DOMAIN = rf"{_LABEL}(?:\.{_LABEL})*\.{_TLD}"

# Admite espacios alrededor de @ y de los puntos (el OCR puede separar el texto).
EMAIL_RE = re.compile(rf"(?<![A-Za-z0-9._%+\-])({_LOCAL_PART})\s*@\s*({_SPACED_DOMAIN})")

FULL_EMAIL_RE = re.compile(rf"{_LOCAL_PART}@{_FULL_DOMAIN}")

# Extensiones de archivo que no son TLDs validos.
BAD_TLDS = frozenset(
    {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "bmp",
        "tif",
        "tiff",
        "svg",
        "pdf",
        "doc",
        "docx",
        "xls",
        "xlsx",
        "ppt",
        "pptx",
        "csv",
        "txt",
        "html",
        "htm",
        "css",
        "js",
        "php",
        "exe",
        "dll",
        "zip",
        "rar",
        "mp3",
        "mp4",
        "avi",
        "mov",
        "wav",
        "json",
        "xml",
        "log",
    }
)

# TLDs que Tesseract confunde con frecuencia. Solo se corrigen cuando el TLD
# original no es valido; nunca se toca la parte local ni el resto del dominio.
TLD_CORRECTIONS = {
    "c0m": "com",
    "c0rn": "com",
    "n3t": "net",
    "0rg": "org",
    "1nfo": "info",
}

# Tesseract tambien puede leer la @ como 0, © o ®. Solo reparamos el patron
# cuando hay contexto de email o el dominio es un proveedor conocido, para no
# convertir texto decorativo en un falso positivo.
AT_CONFUSION_RE = re.compile(
    r"(?P<local>[A-Za-z0-9._%+\-]{3,})[0©®]\s*"
    r"(?P<domain>[A-Za-z0-9\-]{2,}(?:\.[A-Za-z0-9\-]{2,})+)"
)
COMMON_EMAIL_DOMAINS = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "hotmail.com",
        "hotmail.com.ar",
        "hotmail.es",
        "outlook.com",
        "outlook.es",
        "yahoo.com",
        "yahoo.com.ar",
        "yahoo.es",
        "icloud.com",
        "live.com",
        "live.com.ar",
        "msn.com",
        "aol.com",
        "protonmail.com",
        "gmx.com",
        "mail.com",
    }
)
EMAIL_CONTEXT_RE = re.compile(
    r"\b(cv|correo|email|e-mail|mail|postul\w*|contacto|enviar|envia)\b",
    re.IGNORECASE,
)

_MAX_OCCURRENCES = 10_000


@dataclass(slots=True)
class EmailHit:
    email: str
    confidence: float
    occurrences: int


def normalize_text(text: str) -> str:
    """Unifica unicode, elimina caracteres invisibles y resuelve ofuscaciones."""
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\u2022", ".").replace("\u00b7", ".")
    for pattern, replacement in OBFUSCATION_REPLACEMENTS:
        text = pattern.sub(replacement, text)
    return text


def repair_ocr_confusions(text: str) -> str:
    """Repara una @ leida como 0/©/® cuando el contexto parece un email."""
    has_context = bool(EMAIL_CONTEXT_RE.search(text))

    def replace(match: re.Match[str]) -> str:
        local = match.group("local")
        domain = match.group("domain").lower()
        if not any(character.isalpha() for character in local):
            return match.group(0)
        if not has_context and domain not in COMMON_EMAIL_DOMAINS:
            return match.group(0)
        return f"{local}@{match.group('domain')}"

    return AT_CONFUSION_RE.sub(replace, text)


def clean_ocr_text(text: str) -> str:
    """Quita ruido tipico de OCR: logos, lineas decorativas y bucles de simbolos."""
    lines: list[str] = []
    previous: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        # Una linea sin letras ni numeros suele ser un icono, un borde o un bucle.
        if not any(character.isalnum() for character in line):
            continue
        # Evita que un bucle exacto del OCR duplique la misma linea.
        if line == previous:
            continue
        lines.append(line)
        previous = line

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def extract_emails(text: str, *, confidence: float = 0.85) -> list[EmailHit]:
    """Extrae y deduplica emails de un texto plano."""
    hits: dict[str, EmailHit] = {}
    score = round(min(max(confidence, 0.0), 1.0), 3)

    for match in EMAIL_RE.finditer(normalize_text(repair_ocr_confusions(text))):
        email = clean_email(match.group(0))
        if email is None:
            continue
        key = email.lower()
        if key in hits:
            hits[key].occurrences = min(hits[key].occurrences + 1, _MAX_OCCURRENCES)
            hits[key].confidence = max(hits[key].confidence, score)
        else:
            hits[key] = EmailHit(
                email=email,
                confidence=score,
                occurrences=1,
            )

    return _sorted(hits.values())


def merge_email_hits(*groups: Iterable[EmailHit]) -> list[EmailHit]:
    """Une resultados por linea y del texto completo, sin duplicar ni sumar de mas."""
    merged: dict[str, EmailHit] = {}

    for group in groups:
        for hit in group:
            key = hit.email.lower()
            confidence = round(min(max(hit.confidence, 0.0), 1.0), 3)
            occurrences = min(max(1, hit.occurrences), _MAX_OCCURRENCES)
            if key not in merged:
                merged[key] = EmailHit(
                    email=hit.email,
                    confidence=confidence,
                    occurrences=occurrences,
                )
                continue

            previous = merged[key]
            previous.occurrences = max(previous.occurrences, occurrences)
            previous.confidence = max(previous.confidence, confidence)

    return _sorted(merged.values())


def _repair_tld(domain: str) -> str:
    """Corrige confusiones tipicas de OCR en el TLD final."""
    label, dot, tld = domain.rpartition(".")
    if not dot:
        return domain
    corrected = TLD_CORRECTIONS.get(tld.lower())
    return f"{label}.{corrected}" if corrected else domain


def clean_email(raw: str) -> str | None:
    """Quita espacios y valida la estructura final del candidato."""
    candidate = re.sub(r"\s+", "", raw).strip(".,;:()[]<>\"'")
    if candidate.count("@") != 1:
        return None

    local, domain = candidate.rsplit("@", 1)
    domain = _repair_tld(domain)
    candidate = f"{local}@{domain}"
    if not 1 <= len(local) <= 64 or not domain:
        return None
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return None
    if len(domain) > 253 or "." not in domain:
        return None
    if "_" in domain or domain.startswith("-") or domain.endswith("-"):
        return None
    if any(not 1 <= len(label) <= 63 for label in domain.split(".")):
        return None

    tld = domain.rsplit(".", 1)[1].lower()
    if not tld.isalpha() or not 2 <= len(tld) <= 24 or tld in BAD_TLDS:
        return None

    if not FULL_EMAIL_RE.fullmatch(candidate):
        return None

    return candidate


def _sorted(hits: Iterable[EmailHit]) -> list[EmailHit]:
    return sorted(hits, key=lambda hit: (-hit.confidence, hit.email.lower()))
