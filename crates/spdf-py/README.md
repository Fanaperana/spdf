# spdf (Python)

Python bindings for [spdf](https://github.com/Fanaperana/spdf) — a fast,
spatially-precise PDF parser written in Rust.

## Install (from source, for now)

```bash
pip install maturin
maturin develop --release -m crates/spdf-py/Cargo.toml
# Optional: bundle the Tesseract OCR engine
maturin develop --release -m crates/spdf-py/Cargo.toml --features tesseract
```

A pre-built wheel is not yet published. Once published it will be
`pip install spdf`.

## Quick start

```python
import spdf

result = spdf.parse("invoice.pdf")
for page in result["pages"]:
    print(page["page"], "→", len(page["textItems"]), "items")
    print(page["text"])
```

## API

```python
spdf.parse(source, **options) -> dict
spdf.version() -> str
spdf.__version__  # same as version()
```

`source` may be a `str`, `pathlib.Path`, `bytes`, or `bytearray`.

### Options

| kwarg                  | type   | default | description                                        |
| ---------------------- | ------ | ------- | -------------------------------------------------- |
| `ocr`                  | `bool` | `True`  | Run OCR on sparse-text pages / embedded images.    |
| `ocr_language`         | `str`  | `"eng"` | Tesseract language code(s), comma-separated.       |
| `ocr_server_url`       | `str`  | —       | HTTP OCR server endpoint.                          |
| `tessdata_path`        | `str`  | —       | Tesseract language data directory.                 |
| `max_pages`            | `int`  | `1000`  | Hard cap on pages processed.                       |
| `target_pages`         | `str`  | —       | Page selector, e.g. `"1-5,10,15-20"`.              |
| `dpi`                  | `int`  | `150`   | Rendering DPI.                                     |
| `password`             | `str`  | —       | PDF decryption password.                           |
| `precise_bounding_box` | `bool` | `True`  | Include `boundingBoxes` per page.                  |
| `preserve_small_text`  | `bool` | `False` | Keep sub-2pt glyphs (form labels).                 |
| `detect_tables`        | `bool` | `False` | Attach structured tables to each page.             |
| `timeout_secs`         | `int`  | —       | Abort if wall-clock exceeds this many seconds.     |

### Return value

Parsed result as a plain `dict` matching the CLI's JSON schema (camelCase
keys). See [the main README](../../README.md#json-output) for the full
shape.

## Thread safety

`spdf.parse` releases the GIL for the duration of the parse, so it can
be driven concurrently from a `ThreadPoolExecutor`. Each call uses its
own internal rayon pool for per-page work.
