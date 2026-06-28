"""Servicos de parsing e agrupamento de imagens por grupo e tempo."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import unicodedata


SUPPORTED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
TIME_ORDER = ("0h", "24h", "48h")
GROUP_PATTERN = re.compile(
    r"(?i)(?P<list>ni|i)\s*[-_ ]*\s*(?P<time>0h|24h|48h)\s*[-_ ]*\s*(?P<id>\d+(?:[.,]\d+)*)"
)
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
NUMERIC_TOKEN_PATTERN = re.compile(r"(?<![A-Za-z0-9])\d+(?:[.,]\d+)*(?![A-Za-z0-9])")
TIME_DIGIT_SET = {"0", "24", "48"}
TIME_CONNECTOR_TOKENS = {"h", "hr", "hrs", "hora", "horas", "time", "tempo", "t", "day", "dia", "d"}
LIST_TAG_ALIASES: dict[str, str] = {
    "i": "i",
    "inf": "i",
    "infectado": "i",
    "infectada": "i",
    "infected": "i",
    "inoculado": "i",
    "inoculada": "i",
    "inoc": "i",
    "ni": "ni",
    "ninf": "ni",
    "naoinf": "ni",
    "naoinfectado": "ni",
    "naoinfectada": "ni",
    "naoinoculado": "ni",
    "naoinoculada": "ni",
    "controle": "ni",
    "control": "ni",
    "ctrl": "ni",
    "uninfected": "ni",
    "noninfected": "ni",
}
TIME_TOKEN_ALIASES: dict[str, str] = {
    "0h": "0h",
    "h0": "0h",
    "t0": "0h",
    "tempo0": "0h",
    "time0": "0h",
    "baseline": "0h",
    "zero": "0h",
    "24h": "24h",
    "h24": "24h",
    "t24": "24h",
    "tempo24": "24h",
    "time24": "24h",
    "day1": "24h",
    "d1": "24h",
    "48h": "48h",
    "h48": "48h",
    "t48": "48h",
    "tempo48": "48h",
    "time48": "48h",
    "day2": "48h",
    "d2": "48h",
}


@dataclass
class GroupImage:
    path: Path
    list_tag: str
    time_tag: str | None
    image_id: str


def _strip_accents(text: str) -> str:
    """Remove acentos para facilitar comparacoes de tokens heterogeneos."""
    normalized = unicodedata.normalize("NFKD", str(text))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _norm_token(token: str) -> str:
    """Normaliza token para uma forma alfanumerica simples."""
    value = _strip_accents(token).lower().strip()
    return re.sub(r"[^a-z0-9]+", "", value)


def _tokenize_text(text: str) -> list[str]:
    """Quebra texto em tokens alfanumericos preservando a ordem de aparicao."""
    return [match.group(0) for match in TOKEN_PATTERN.finditer(str(text))]


def _build_parse_contexts(path: Path, base_folder: Path | None = None) -> list[str]:
    """Monta contextos textuais (arquivo + pastas) usados para inferir grupo e tempo."""
    contexts: list[str] = []
    seen: set[str] = set()

    def add_ctx(value: str) -> None:
        txt = str(value or "").strip()
        if not txt:
            return
        key = txt.lower()
        if key in seen:
            return
        seen.add(key)
        contexts.append(txt)

    add_ctx(path.stem)
    try:
        rel = path.relative_to(base_folder) if base_folder is not None else path
    except Exception:
        rel = path

    parent_parts = [part for part in rel.parent.parts if part not in ("", ".", "..")]
    if parent_parts:
        near_to_far = list(reversed(parent_parts))
        short_ctx = " ".join(near_to_far[:3])
        add_ctx(short_ctx)
        add_ctx(" ".join([path.stem] + near_to_far[:3]))
        if len(near_to_far) > 3:
            add_ctx(" ".join([path.stem] + near_to_far))

    return contexts


def _detect_list_tag(text: str) -> str | None:
    """Detecta marcador da lista (I/NI e aliases) em um trecho de texto."""
    tokens = [_norm_token(tok) for tok in _tokenize_text(text)]
    tokens = [tok for tok in tokens if tok]
    for tok in tokens:
        mapped = LIST_TAG_ALIASES.get(tok)
        if mapped is not None:
            return mapped

    for tok in tokens:
        if tok.startswith("nao") and ("infect" in tok or "inf" in tok):
            return "ni"
        if tok.startswith("non") and ("infect" in tok or "inf" in tok):
            return "ni"
        if tok.startswith("inf") or ("infect" in tok):
            return "i"
    return None


def _detect_time_tag(text: str) -> str | None:
    """Detecta tempo (0h/24h/48h) aceitando formatos e aliases comuns."""
    tokens = [_norm_token(tok) for tok in _tokenize_text(text)]
    tokens = [tok for tok in tokens if tok]
    for idx, tok in enumerate(tokens):
        mapped = TIME_TOKEN_ALIASES.get(tok)
        if mapped is not None:
            return mapped
        if tok in TIME_CONNECTOR_TOKENS and idx + 1 < len(tokens):
            next_tok = tokens[idx + 1]
            if next_tok in TIME_DIGIT_SET:
                return f"{next_tok}h"
        if tok in TIME_DIGIT_SET and idx + 1 < len(tokens):
            next_tok = tokens[idx + 1]
            if next_tok in TIME_CONNECTOR_TOKENS:
                return f"{tok}h"
        if tok.startswith("t") and tok[1:] in TIME_DIGIT_SET:
            return f"{tok[1:]}h"
        if tok.startswith("h") and tok[1:] in TIME_DIGIT_SET:
            return f"{tok[1:]}h"
    return None


def _is_time_like_token(tok: str) -> bool:
    if tok in TIME_TOKEN_ALIASES:
        return True
    if tok in TIME_DIGIT_SET:
        return True
    if tok in TIME_CONNECTOR_TOKENS:
        return True
    if tok.startswith("t") and tok[1:] in TIME_DIGIT_SET:
        return True
    if tok.startswith("h") and tok[1:] in TIME_DIGIT_SET:
        return True
    return False


def _extract_group_label(text: str) -> str | None:
    """Extrai um identificador textual de grupo descartando tokens de tempo e numeros puros."""
    kept: list[str] = []
    for raw_tok in _tokenize_text(text):
        tok = _norm_token(raw_tok)
        if not tok:
            continue
        if _is_time_like_token(tok):
            continue
        if tok.isdigit():
            continue
        kept.append(tok)

    if not kept:
        return None
    return "_".join(kept)


def _extract_numeric_id(text: str, time_tag: str | None = None) -> str | None:
    """Extrai o identificador numerico mais provavel do grupo."""
    skip_numeric = {str(time_tag[:-1])} if time_tag is not None else set()
    candidates: list[str] = []
    for match in NUMERIC_TOKEN_PATTERN.finditer(str(text)):
        raw = match.group(0)
        if raw in skip_numeric:
            continue
        normalized = normalize_group_position(raw)
        if normalized in skip_numeric:
            continue
        if normalized:
            candidates.append(normalized)
    if not candidates:
        return None
    return candidates[-1]


def _extract_textual_id(
    text: str,
    time_tag: str | None = None,
    ignored_tokens: set[str] | None = None,
) -> str | None:
    skip_numeric = {str(time_tag[:-1])} if time_tag is not None else set()
    ignored = set(ignored_tokens or ())
    kept: list[str] = []
    for raw_tok in _tokenize_text(text):
        tok = _norm_token(raw_tok)
        if not tok:
            continue
        if tok in ignored:
            continue
        if _is_time_like_token(tok):
            continue
        if tok in skip_numeric:
            continue
        if tok.isdigit():
            continue
        kept.append(tok)

    if not kept:
        return None

    candidate = kept[-1]
    if len(candidate) > 48:
        digest = hashlib.sha1(candidate.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate[:39]}_{digest}"
    return candidate


def normalize_group_position(raw_id: str) -> str:
    """
    Extrai o identificador de posicao do grupo a partir do token numerico.
    Exemplos esperados:
    - "1.11418" -> "1.1"
    - "1.11634" -> "1.1"
    - "1,1" -> "1.1"
    - "2.3" -> "2.3"
    """
    clean = str(raw_id).strip().replace(",", ".")
    if not clean:
        return clean

    if "." not in clean:
        return clean

    parts = [p for p in clean.split(".") if p]
    if len(parts) < 2:
        return clean

    head, frac = parts[0], parts[1]
    if not head.isdigit() or not frac.isdigit():
        return clean

    # PT: Nomes como 1.11418 trazem sufixos extras apos a posicao do grupo. | EN: Names such as 1.11418 contain extra suffixes after the group position.
    if len(parts) == 2 and len(frac) >= 4:
        return f"{head}.{frac[0]}"

    return f"{head}.{frac}"


def parse_group_image(path: Path, base_folder: Path | None = None) -> GroupImage | None:
    """Interpreta nome/caminho de uma imagem e retorna metadados de agrupamento."""
    match = GROUP_PATTERN.search(path.stem)
    if match is not None:
        list_tag = match.group("list").lower()
        time_tag = match.group("time").lower()
        image_id = normalize_group_position(match.group("id"))
        return GroupImage(path=path, list_tag=list_tag, time_tag=time_tag, image_id=image_id)

    contexts = _build_parse_contexts(path, base_folder=base_folder)
    if not contexts:
        return None

    time_tag: str | None = None
    for ctx in contexts:
        if time_tag is None:
            time_tag = _detect_time_tag(ctx)
        if time_tag is not None:
            break

    list_tag = _extract_group_label(path.stem)
    if list_tag is None:
        for ctx in contexts[1:]:
            list_tag = _extract_group_label(ctx)
            if list_tag is not None:
                break
    if list_tag is None:
        for ctx in contexts:
            list_tag = _detect_list_tag(ctx)
            if list_tag is not None:
                break
    if list_tag is None:
        list_tag = "grupo"

    image_id = _extract_numeric_id(path.stem, time_tag=time_tag)
    if image_id is None:
        for ctx in contexts[1:]:
            image_id = _extract_numeric_id(ctx, time_tag=time_tag)
            if image_id is not None:
                break

    ignored_tokens = {tok for tok in str(list_tag).split("_") if tok}
    if image_id is None:
        image_id = _extract_textual_id(path.stem, time_tag=time_tag, ignored_tokens=ignored_tokens)
    if image_id is None:
        for ctx in contexts[1:]:
            image_id = _extract_textual_id(ctx, time_tag=time_tag, ignored_tokens=ignored_tokens)
            if image_id is not None:
                break
    if image_id is None:
        try:
            rel = path.relative_to(base_folder) if base_folder is not None else path.name
        except Exception:
            rel = path.name
        digest = hashlib.sha1(str(rel).lower().encode("utf-8")).hexdigest()[:8]
        image_id = f"item_{digest}"

    return GroupImage(path=path, list_tag=list_tag, time_tag=time_tag, image_id=image_id)


def discover_image_groups(
    folder: Path,
    progress_callback=None,
) -> tuple[dict[str, dict[str, Path]], list[tuple[str, str, Path, Path]]]:
    """Agrupa imagens da pasta em slots 0h/24h/48h e identifica duplicatas."""
    groups: dict[str, dict[str, Path]] = {}
    duplicates: list[tuple[str, str, Path, Path]] = []
    untimed_by_group: dict[str, list[Path]] = {}
    # PT: Escaneia somente a pasta selecionada (sem subpastas), para evitar misturar imagens de outros conjuntos.
    # EN: Scans only the selected folder (without subfolders) to avoid mixing images from other datasets.
    all_paths = sorted(folder.glob("*"), key=lambda p: p.name.lower())
    total = len(all_paths)
    if progress_callback is not None:
        progress_callback(0, total)

    for idx, path in enumerate(all_paths, start=1):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
            if progress_callback is not None and (idx == total or (idx % 40) == 0):
                progress_callback(idx, total)
            continue

        parsed = parse_group_image(path, base_folder=folder)
        if parsed is None:
            if progress_callback is not None and (idx == total or (idx % 40) == 0):
                progress_callback(idx, total)
            continue

        group_key = f"{parsed.list_tag} {parsed.image_id}"
        if parsed.time_tag is None:
            untimed_by_group.setdefault(group_key, []).append(path)
            if progress_callback is not None and (idx == total or (idx % 40) == 0):
                progress_callback(idx, total)
            continue

        slot = groups.setdefault(group_key, {})
        if parsed.time_tag not in slot:
            slot[parsed.time_tag] = path
        else:
            duplicates.append((group_key, parsed.time_tag, slot[parsed.time_tag], path))

        if progress_callback is not None and (idx == total or (idx % 40) == 0):
            progress_callback(idx, total)

    # PT: Fallback para imagens sem tempo explicito: preenche 0h/24h/48h em ordem, preservando arquivos com tempo detectado.
    # EN: Fallback for images without an explicit time point: fills 0h/24h/48h in order while preserving files whose time point was detected.
    for group_key, pending_paths in untimed_by_group.items():
        slot = groups.setdefault(group_key, {})
        ordered_pending = sorted(pending_paths, key=lambda p: p.name.lower())
        for pending_path in ordered_pending:
            next_time = next((t for t in TIME_ORDER if t not in slot), None)
            if next_time is None:
                kept_path = slot.get(TIME_ORDER[-1]) or next(iter(slot.values()), pending_path)
                duplicates.append((group_key, "sem_tempo", kept_path, pending_path))
                continue
            slot[next_time] = pending_path

    if progress_callback is not None and total == 0:
        progress_callback(0, 0)
    return groups, duplicates
