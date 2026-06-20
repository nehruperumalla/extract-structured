#!/usr/bin/env python3
"""
extract_structured.py

Extract structured data from unstructured documents (HTML, Markdown, plain text)
using the Anthropic Claude API with native structured outputs (JSON schema).

Usage:
    python extract_structured.py <input_file> <schema_file> [options]

Examples:
    python extract_structured.py report.html schema.json
    python extract_structured.py notes.md schema.json --output result.json
    python extract_structured.py data.txt schema.json --model claude-sonnet-4-5
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import anthropic

# ---------------------------------------------------------------------------
# Optional dependency: markdownify for HTML → Markdown conversion.
# Install with:  pip install markdownify
# Falls back to a lightweight regex-based stripper if unavailable.
# ---------------------------------------------------------------------------
try:
    from markdownify import markdownify as md_convert  # type: ignore

    _HAS_MARKDOWNIFY = True
except ImportError:
    _HAS_MARKDOWNIFY = False


# ---------------------------------------------------------------------------
# HTML → Markdown conversion
# ---------------------------------------------------------------------------

def _html_to_markdown_regex(html: str) -> str:
    """
    Minimal regex-based HTML → Markdown fallback.
    Handles the most common structural elements while preserving text content.
    """
    text = html

    # Remove <script> and <style> blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # Block-level headings
    for level in range(6, 0, -1):
        text = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            lambda m, l=level: f"\n{'#' * l} {m.group(1).strip()}\n",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

    # Paragraphs and divs → blank-line separated
    text = re.sub(r"<(?:p|div)[^>]*>(.*?)</(?:p|div)>", r"\n\1\n", text, flags=re.IGNORECASE | re.DOTALL)

    # Bold / strong
    text = re.sub(r"<(?:b|strong)[^>]*>(.*?)</(?:b|strong)>", r"**\1**", text, flags=re.IGNORECASE | re.DOTALL)

    # Italic / em
    text = re.sub(r"<(?:i|em)[^>]*>(.*?)</(?:i|em)>", r"*\1*", text, flags=re.IGNORECASE | re.DOTALL)

    # Inline code
    text = re.sub(r"<code[^>]*>(.*?)</code>", r"`\1`", text, flags=re.IGNORECASE | re.DOTALL)

    # Pre blocks
    text = re.sub(r"<pre[^>]*>(.*?)</pre>", r"\n```\n\1\n```\n", text, flags=re.IGNORECASE | re.DOTALL)

    # Links:  <a href="URL">TEXT</a>  →  [TEXT](URL)
    text = re.sub(
        r'<a[^>]+href=["\']([^"\']*)["\'][^>]*>(.*?)</a>',
        r"[\2](\1)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Images:  <img … alt="ALT" …>  →  ![ALT](src)
    text = re.sub(
        r'<img[^>]+src=["\']([^"\']*)["\'][^>]*alt=["\']([^"\']*)["\'][^>]*/?>', r"![\2](\1)", text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'<img[^>]+alt=["\']([^"\']*)["\'][^>]+src=["\']([^"\']*)["\'][^>]*/?>', r"![\1](\2)", text, flags=re.IGNORECASE
    )

    # Unordered list items
    text = re.sub(r"<li[^>]*>(.*?)</li>", r"\n- \1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</?[uo]l[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Table cells and rows (basic)
    text = re.sub(r"<th[^>]*>(.*?)</th>", r" **\1** |", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<td[^>]*>(.*?)</td>", r" \1 |", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<tr[^>]*>", "| ", text, flags=re.IGNORECASE)
    text = re.sub(r"</tr>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(?:table|thead|tbody|tfoot)[^>]*>", "\n", text, flags=re.IGNORECASE)

    # Horizontal rules
    text = re.sub(r"<hr[^>]*/?>", "\n---\n", text, flags=re.IGNORECASE)

    # Line breaks
    text = re.sub(r"<br[^>]*/?>", "\n", text, flags=re.IGNORECASE)

    # Strip remaining tags
    text = re.sub(r"<[^>]+>", "", text)

    # Decode common HTML entities
    entity_map = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
        "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
    }
    for entity, char in entity_map.items():
        text = text.replace(entity, char)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown, preferring markdownify if available."""
    if _HAS_MARKDOWNIFY:
        return md_convert(
            html,
            heading_style="ATX",
            bullets="-",
            strip=["script", "style"],
        )
    print(
        "  [info] markdownify not installed; using built-in HTML converter.\n"
        "         For best results: pip install markdownify",
        file=sys.stderr,
    )
    return _html_to_markdown_regex(html)


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".html", ".htm", ".md", ".markdown", ".txt"}


def load_document(path: Path) -> tuple[str, str]:
    """
    Read the file and return (content_as_markdown_or_text, detected_type).
    detected_type is one of: 'html', 'markdown', 'text'
    """
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    raw = path.read_text(encoding="utf-8", errors="replace")

    if suffix in (".html", ".htm"):
        print(f"  [info] Detected HTML — converting to Markdown …", file=sys.stderr)
        content = html_to_markdown(raw)
        doc_type = "html"
    elif suffix in (".md", ".markdown"):
        content = raw
        doc_type = "markdown"
    else:  # .txt
        content = raw
        doc_type = "text"

    return content, doc_type


# ---------------------------------------------------------------------------
# Schema loading & validation
# ---------------------------------------------------------------------------

def load_schema(path: Path) -> dict:
    """Load and lightly validate a JSON Schema from a file."""
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema file is not valid JSON: {exc}") from exc

    if not isinstance(schema, dict):
        raise ValueError("Schema must be a JSON object (dict).")
    if schema.get("type") != "object":
        raise ValueError(
            'Top-level schema "type" must be "object" for structured outputs.'
        )

    return schema


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def extract_with_claude(
    content: str,
    doc_type: str,
    schema: dict,
    model: str,
    max_tokens: int,
) -> dict:
    """
    Call Claude with structured outputs (JSON schema) to extract data.
    Returns the parsed JSON dict.
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    type_description = {
        "html": "an HTML document that has been converted to Markdown",
        "markdown": "a Markdown document",
        "text": "a plain-text document",
    }.get(doc_type, "a document")

    system_prompt = (
        "You are a precise data-extraction assistant. "
        "Your sole task is to extract structured information from the provided document "
        "and return it as valid JSON matching the supplied schema. "
        "Extract ALL information that matches each field; if a field cannot be found, "
        "use null or an empty value appropriate to the field's type. "
        "Do not invent data that is not present in the document."
    )

    user_prompt = (
        f"Extract structured data from the following {type_description}.\n\n"
        f"---\n{content}\n---"
    )

    print(f"  [info] Calling Claude ({model}) with structured outputs …", file=sys.stderr)

    response = client.beta.messages.create(
        model=model,
        betas=["structured-outputs-2025-11-13"],
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": schema,
            }
        },
    )

    # Extract text content from the response
    text_blocks = [block.text for block in response.content if hasattr(block, "text")]
    raw_text = "\n".join(text_blocks).strip()

    # The API guarantees valid JSON when using structured outputs, but we parse
    # defensively in case of any surrounding whitespace/fences.
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text, flags=re.DOTALL).strip()
    return json.loads(clean)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured data from HTML/Markdown/text documents using Claude.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input_file", type=Path, help="Path to the input document (.html, .htm, .md, .markdown, .txt)")
    parser.add_argument("schema_file", type=Path, help="Path to the JSON Schema file defining the output structure")
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Path to write the extracted JSON (default: print to stdout)",
    )
    parser.add_argument(
        "--model", "-m",
        default="claude-sonnet-4-5",
        help="Claude model to use (default: claude-sonnet-4-5)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for the response (default: 4096)",
    )
    parser.add_argument(
        "--show-markdown",
        action="store_true",
        help="If the input is HTML, print the converted Markdown before extraction",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- Validate inputs ---
    if not args.input_file.exists():
        print(f"Error: input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    if not args.schema_file.exists():
        print(f"Error: schema file not found: {args.schema_file}", file=sys.stderr)
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    # --- Load document ---
    print(f"\nLoading document: {args.input_file}", file=sys.stderr)
    try:
        content, doc_type = load_document(args.input_file)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  [info] Document type: {doc_type} | Characters: {len(content):,}", file=sys.stderr)

    if args.show_markdown and doc_type == "html":
        print("\n--- Converted Markdown ---", file=sys.stderr)
        print(content, file=sys.stderr)
        print("--- End Markdown ---\n", file=sys.stderr)

    # --- Load schema ---
    print(f"Loading schema: {args.schema_file}", file=sys.stderr)
    try:
        schema = load_schema(args.schema_file)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"  [info] Schema fields: {list(schema.get('properties', {}).keys())}", file=sys.stderr)

    # --- Extract ---
    try:
        result = extract_with_claude(
            content=content,
            doc_type=doc_type,
            schema=schema,
            model=args.model,
            max_tokens=args.max_tokens,
        )
    except anthropic.APIStatusError as exc:
        print(f"Claude API error ({exc.status_code}): {exc.message}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as exc:
        print(f"Failed to parse Claude's response as JSON: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Output ---
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
        print(f"\nExtracted data written to: {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()