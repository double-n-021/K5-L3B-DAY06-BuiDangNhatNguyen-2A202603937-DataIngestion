"""Read and normalize source data."""
import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage

# Bắt các ký tự điều khiển ASCII (trừ \t, \n, \r) sinh ra do lỗi bảng mã CMap ở hình vẽ PDF
_CONTROL_CHAR_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f]")

# Regex nhận diện dòng tiêu đề mục (Abstract, 1 Introduction, 3.1 Setup, Appendix A...)
_HEADING_RE = re.compile(
    r"^(?:"
    r"Abstract|ABSTRACT|"
    r"Introduction|Background|Conclusion|Acknowledgements|Acknowledgments|Broader Impact|"
    r"Reproducibility Statement|Ethics Statement|REPRODUCIBILITY STATEMENT|ETHICS STATEMENT|"
    r"(?:\d+(?:\.\d+)*|[A-I](?:\.\d+)*)\s+[A-Z][A-Za-z0-9 \-:,/()]{2,75}"
    r")$"
)

# File PDF của ICLR (bài ReAct) khi đọc qua pypdf bị giãn cách chữ cái đầu ở các heading
_ICLR_SPACED_HEADING_MAP = {
    "1 I NTRODUCTION": "1 Introduction",
    "2 REAC T: S YNERGIZING REASONING + AC TING": "2 ReAct: Synergizing Reasoning + Acting",
    "3 K NOWLEDGE -I NTENSIVE REASONING TASKS": "3 Knowledge-Intensive Reasoning Tasks",
    "3.1 S ETUP": "3.1 Setup",
    "3.2 M ETHODS": "3.2 Methods",
    "3.3 R ESULTS AND OBSERVATIONS": "3.3 Results and Observations",
    "4 D ECISION MAKING TASKS": "4 Decision Making Tasks",
    "5 R ELATED WORK": "5 Related Work",
    "6 C ONCLUSION": "6 Conclusion",
    "A A DDITIONAL RESULTS": "A Additional Results",
    "A.1 GPT-3 E XPERIMENTS": "A.1 GPT-3 Experiments",
    "A.2 REAC T OBTAINS UP -TO-DATE KNOWLEDGE ON HOTPOT QA": "A.2 ReAct Obtains Up-to-Date Knowledge on HotpotQA",
    "A.3 H UMAN -IN-THE -LOOP BEHAVIOR CORRECTION ON ALFWORLD": "A.3 Human-in-the-Loop Behavior Correction on ALFWorld",
    "B E XPERIMENT DETAILS": "B Experiment Details",
    "B.1 H OTPOT QA F INETUNING DETAILS": "B.1 HotpotQA Finetuning Details",
    "B.2 A LFWORLD IM-S TYLE DETAILS": "B.2 ALFWorld IM-Style Details",
    "E M ORE ANALYSIS": "E More Analysis",
    "E.1 S UCCESS AND FAILURE MODES ANALYSIS": "E.1 Success and Failure Modes Analysis",
}


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace without discarding useful structure."""
    if not text:
        return ""

    # Dùng NFKC trước để tách các chữ ghép ligature (ﬁ -> fi, ﬂ -> fl) của LaTeX, sau đó đưa về NFC
    text = unicodedata.normalize("NFKC", text)
    text = unicodedata.normalize("NFC", text)

    # Xóa byte rỗng, dấu gạch mềm (soft hyphen) và đồng nhất kiểu xuống dòng
    text = text.replace("\x00", "").replace("\u00a0", " ").replace("\u00ad", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Lọc bỏ các dòng bị lỗi font từ hình vẽ nhúng (như Figure 1, 4, 5 trong bài ReAct)
    clean_raw_lines: list[str] = []
    for raw_line in text.split("\n"):
        if _CONTROL_CHAR_RE.search(raw_line):
            continue
        clean_raw_lines.append(raw_line)
    text = "\n".join(clean_raw_lines)

    # Xóa các dòng watermark/header hội nghị lặp lại ở đầu trang và câu bản quyền ở chân trang
    text = re.sub(r"Published as a conference paper at ICLR 2023\s*", "", text)
    text = re.sub(
        r"Provided proper attribution is provided, Google hereby grants permission to\s*"
        r"reproduce the tables and figures in this paper solely for use in journalistic or\s*"
        r"scholarly works\.\s*",
        "",
        text,
    )
    text = re.sub(r"arXiv:\d{4}\.\d{4,5}v\d+\s*\[[^\]]+\]\s*\d+\s+[A-Za-z]+\s+\d{4}\s*", "", text)

    # Nối lại các từ bị gạch ngang ngắt dòng ở cuối lề phải (vd: "transduc-\ntion" -> "transduction")
    text = re.sub(r"([A-Za-z]{2,})-\s*\n\s*([a-z]{2,})", r"\1\2", text)

    # Gom khoảng trắng thừa trên từng dòng, bỏ dòng chỉ chứa số trang và chuẩn hóa lại heading ICLR
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if re.fullmatch(r"\d{1,2}", line):
            continue
        if line in _ICLR_SPACED_HEADING_MAP:
            line = _ICLR_SPACED_HEADING_MAP[line]
        lines.append(line)

    # Tái tạo lại dấu ngắt đoạn (\n\n) quanh các dòng tiêu đề mục và chú thích bảng/hình
    rebuilt: list[str] = []
    for line in lines:
        if not line:
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            continue
        if _HEADING_RE.match(line) or line.startswith(("Table ", "Figure ")):
            if rebuilt and rebuilt[-1] != "":
                rebuilt.append("")
            rebuilt.append(line)
            if _HEADING_RE.match(line):
                rebuilt.append("")
        else:
            rebuilt.append(line)

    # Gom các dòng thuộc cùng một đoạn văn; nếu dòng ngắn kết thúc bằng dấu chấm câu thì ngắt đoạn
    paragraphs: list[str] = []
    buf: list[str] = []
    for line in rebuilt:
        if line == "":
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
        else:
            buf.append(line)
            if len(line) < 68 and line.endswith((".", ":", "?")):
                paragraphs.append(" ".join(buf))
                buf = []
    if buf:
        paragraphs.append(" ".join(buf))

    normalized = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_page_section(normalized_text: str, default_section: str = "Overview") -> str:
    """Lấy tiêu đề mục xuất hiện trên trang; nếu trang không có mục mới thì giữ mục của trang trước."""
    for para in normalized_text.split("\n\n"):
        line = para.strip()
        if _HEADING_RE.match(line):
            return line
    return default_section


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    valid_pages: set[int] | None = None,
    base_metadata: dict[str, Any] | None = None,
    error_log: list[dict[str, Any]] | None = None,
) -> list[DocumentPage]:
    """Extract each physical page; keep its page number and source.

    Detect and record pages with no text. OCR, when needed, is a separate step.
    """
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    active_section = "Overview & Abstract"
    extra_meta = dict(base_metadata) if base_metadata else {}

    for page_idx, pdf_page in enumerate(reader.pages, start=1):
        try:
            raw_text = pdf_page.extract_text() or ""
        except Exception as exc:
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EXTRACTION_ERROR",
                    "detail": str(exc),
                })
            continue

        # Ghi nhận nếu gặp trang trống không có text
        if not raw_text.strip():
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EMPTY_PAGE",
                    "detail": "Physical page yielded 0 extractable characters.",
                })
            continue

        # Bỏ qua các trang không thuộc nội dung chính (References, Visualization, Prompt dump) theo manifest
        if valid_pages is not None and page_idx not in valid_pages:
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EXCLUDED_NON_CONTENT_PAGE",
                    "detail": f"Excluded per manifest rules (raw_chars={len(raw_text)}).",
                })
            continue

        # Kiểm tra và ghi log nếu trang có các dòng rác từ hình vẽ bị lỗi bảng mã CMap
        garbled_lines = [ln for ln in raw_text.splitlines() if _CONTROL_CHAR_RE.search(ln)]
        if garbled_lines and error_log is not None:
            error_log.append({
                "document_id": document_id,
                "source": source,
                "page": page_idx,
                "event": "GARBLED_FIGURE_CMAP_CLEANED",
                "detail": f"Stripped {len(garbled_lines)} embedded figure lines containing ASCII control bytes.",
            })

        norm_text = normalize_text(raw_text)
        if not norm_text:
            if error_log is not None:
                error_log.append({
                    "document_id": document_id,
                    "source": source,
                    "page": page_idx,
                    "event": "EMPTY_AFTER_NORMALIZATION",
                    "detail": "Page became empty after normalization.",
                })
            continue

        active_section = extract_page_section(norm_text, default_section=active_section)
        page_meta = {
            **extra_meta,
            "page_id": f"{document_id}_p{page_idx:02d}",
            "section": active_section,
            "raw_chars": len(raw_text),
            "normalized_chars": len(norm_text),
        }

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_idx,
                text=norm_text,
                metadata=page_meta,
            )
        )

    return pages
