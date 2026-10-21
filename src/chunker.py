import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional


BOILERPLATE_PATTERNS = [
    r"^Skip to navigation\s*$",
    r"^Scroll to top\s*$",
    r"^Ask a question\|Copy page\|.*$",
    r"^This page is for (API Reference|Documentation|Developer Guide)\.\s*$",
]


@dataclass
class DocChunk:
    chunk_id: str
    doc_path: str
    url: str
    title: str
    section: str
    category: str
    has_code: bool
    char_count: int
    token_estimate: int
    content: str
    formatted_for_embedding: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_frontmatter(raw_md: str) -> tuple[Dict[str, str], str]:
    """Extract YAML frontmatter and return (metadata_dict, markdown_body)."""
    meta: Dict[str, str] = {}
    if raw_md.startswith("---\n"):
        end_idx = raw_md.find("\n---\n", 4)
        if end_idx != -1:
            fm_block = raw_md[4:end_idx]
            for line in fm_block.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = raw_md[end_idx + 5 :]
            return meta, body
    return meta, raw_md


def clean_markdown_body(body: str) -> str:
    """Clean navigation boilerplate from crawled Sarvam docs while preserving code and tables."""
    # Remove 'On this page' TOC block up to 'Scroll to top'
    body = re.sub(
        r"On this page\s*\n+(?:\s*\*.*\n+)+Scroll to top\s*\n*",
        "\n",
        body,
        flags=re.MULTILINE,
    )
    cleaned_lines = []
    for line in body.splitlines():
        skip = False
        for pat in BOILERPLATE_PATTERNS:
            if re.match(pat, line.strip()):
                skip = True
                break
        if not skip:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def infer_category(rel_path: str, title: str, text: str) -> str:
    """Classify a Sarvam documentation file into a product/API category."""
    s = f"{rel_path} {title}".lower()
    if "text-to-speech" in s or "tts" in s or "bulbul" in s:
        return "text-to-speech"
    if "speech-to-text" in s or "stt" in s or "saaras" in s or "saarika" in s:
        return "speech-to-text"
    if "voice-cloning" in s or "cloning" in s or "voice-library" in s:
        return "voice-cloning"
    if "dubbing" in s:
        return "dubbing"
    if "voice-agents" in s or "samvaad" in s or "agent" in s:
        return "voice-agents"
    if "docai" in s or "document" in s or "parse" in s:
        return "document-ai"
    if "translate" in s or "transliterat" in s or "mayura" in s or "lid" in s:
        return "translation-nlp"
    if "chat" in s or "sarvam-m" in s or "indus" in s:
        return "chat-llm"
    if "cowork" in s:
        return "cowork"
    if "authentication" in s or "api-reference" in s:
        return "api-core"
    return "general"


def split_markdown_sections(body: str, max_chars: int = 1600, overlap_chars: int = 200) -> List[tuple[str, str]]:
    """
    Split markdown by H1/H2/H3 headers while keeping code blocks intact.
    Returns list of (section_heading, section_content).
    """
    lines = body.splitlines()
    sections: List[tuple[str, List[str]]] = []
    current_heading = "Overview"
    current_lines: List[str] = []
    in_code_block = False

    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block

        header_match = not in_code_block and re.match(r"^(#{1,3})\s+(.+)$", line.strip())
        if header_match:
            if current_lines:
                content_str = "\n".join(current_lines).strip()
                if len(content_str) >= 60:
                    sections.append((current_heading, current_lines))
                current_lines = []
            current_heading = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", header_match.group(2)).strip()
            current_lines.append(line)
        else:
            current_lines.append(line)

    if current_lines:
        content_str = "\n".join(current_lines).strip()
        if len(content_str) >= 40:
            sections.append((current_heading, current_lines))

    # Sub-chunk any section that exceeds max_chars
    final_chunks: List[tuple[str, str]] = []
    for heading, sec_lines in sections:
        full_sec = "\n".join(sec_lines).strip()
        if len(full_sec) <= max_chars:
            final_chunks.append((heading, full_sec))
        else:
            paragraphs = full_sec.split("\n\n")
            buf = ""
            part = 1
            for para in paragraphs:
                if len(buf) + len(para) + 2 <= max_chars:
                    buf = f"{buf}\n\n{para}".strip() if buf else para
                else:
                    if buf:
                        sub_heading = f"{heading} (Part {part})" if part > 1 else heading
                        final_chunks.append((sub_heading, buf))
                        part += 1
                        overlap = buf[-overlap_chars:] if overlap_chars > 0 else ""
                        buf = f"{overlap}\n\n{para}".strip()
                    else:
                        # Single paragraph larger than max_chars
                        final_chunks.append((f"{heading} (Part {part})", para[:max_chars]))
                        part += 1
                        buf = para[max_chars - overlap_chars :] if len(para) > max_chars else ""
            if buf and len(buf) >= 40:
                sub_heading = f"{heading} (Part {part})" if part > 1 else heading
                final_chunks.append((sub_heading, buf))

    return final_chunks


def build_sarvam_corpus(
    docs_root: Path,
    max_files: Optional[int] = None,
    include_subdomains: bool = False,
    max_chunks_per_file: int = 5,
) -> List[DocChunk]:
    """
    Crawl local sarvam-docs directory, parse frontmatter & markdown sections,
    and format chunks with EmbeddingGemma 2's `title: {title} | text: {content}` template.
    """
    md_files = sorted(docs_root.rglob("*.md"))
    filtered_files: List[Path] = []
    for f in md_files:
        rel = f.relative_to(docs_root).as_posix()
        if rel.startswith("_llms") or rel.startswith("_meta") or rel in ("llms.txt.md", "llms-full.txt.md"):
            continue
        if not include_subdomains and rel.startswith("_subdomains"):
            continue
        filtered_files.append(f)

    # Prioritize core API reference & product guides first so any top-N slice has full category coverage
    priority_docs = [
        "api-reference/authentication.md",
        "api-reference.md",
        "documentation/creative-models/creative-text-to-speech.md",
        "documentation/creative-models/creative-speech-to-text.md",
        "documentation/creative-models/creative-voice-cloning.md",
        "documentation/creative-models/creative-dubbing.md",
        "documentation/getting-started/models/sarvam-translate.md",
        "documentation/getting-started/models/sarvam-m.md",
        "documentation/docai/sarvam-parse.md",
        "documentation/voice-agents/introduction.md",
    ]
    for pdoc in reversed(priority_docs):
        p_path = docs_root / pdoc
        if p_path.exists():
            if p_path in filtered_files:
                filtered_files.remove(p_path)
            filtered_files.insert(0, p_path)

    # Also include high-value root guides from sarvam repo if present
    repo_root = docs_root.parent
    for extra_guide in [
        "README_TTS.md",
        "TAMIL_API_GUIDE.md",
        "HOW_IT_WAS_DONE.md",
        "GUIDE.md",
    ]:
        extra_path = repo_root / extra_guide
        if extra_path.exists():
            filtered_files.insert(0, extra_path)

    if max_files is not None:
        filtered_files = filtered_files[:max_files]

    chunks: List[DocChunk] = []
    seen_hashes = set()

    for file_path in filtered_files:
        try:
            raw_md = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        meta, body = parse_frontmatter(raw_md)
        try:
            rel_path = file_path.relative_to(docs_root).as_posix()
        except ValueError:
            rel_path = file_path.name

        title = meta.get("title") or rel_path.replace(".md", "").replace("/", " - ")
        title = title.replace(" | Sarvam API Docs", "").strip()
        url = meta.get("url") or f"https://github.com/ajithvnr2001/sarvam-docs/blob/main/{rel_path}"

        cleaned = clean_markdown_body(body)
        if len(cleaned) < 60:
            continue

        sec_tuples = split_markdown_sections(cleaned, max_chars=1100, overlap_chars=150)
        file_chunk_count = 0
        for idx, (sec_heading, sec_text) in enumerate(sec_tuples):
            if max_chunks_per_file and file_chunk_count >= max_chunks_per_file:
                break
            # Deduplicate near-identical chunks across mirrored routes
            norm_key = re.sub(r"\s+", " ", sec_text[:400]).strip().lower()
            if norm_key in seen_hashes:
                continue
            seen_hashes.add(norm_key)
            file_chunk_count += 1

            has_code = "```" in sec_text or "curl " in sec_text or "api-subscription-key" in sec_text or "POST " in sec_text
            category = infer_category(rel_path, title, sec_text)
            full_title = f"{title} — {sec_heading}" if sec_heading and sec_heading != title else title
            formatted = f"title: {full_title} | text: {sec_text}"

            chunk_id = f"sarvam_{len(chunks):04d}"
            chunks.append(
                DocChunk(
                    chunk_id=chunk_id,
                    doc_path=rel_path,
                    url=url,
                    title=full_title,
                    section=sec_heading,
                    category=category,
                    has_code=has_code,
                    char_count=len(sec_text),
                    token_estimate=max(1, len(sec_text) // 4),
                    content=sec_text,
                    formatted_for_embedding=formatted,
                )
            )

    return chunks


def save_chunks_jsonl(chunks: List[DocChunk], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def load_chunks_jsonl(in_path: Path) -> List[DocChunk]:
    chunks: List[DocChunk] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(DocChunk(**json.loads(line)))
    return chunks
