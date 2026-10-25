from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.ocr import OcrError
from app.services.pipeline import ExtractionPipeline
from app.services.preprocess import ImageTooLargeError, ImageValidationError


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae emails de una imagen con OCR local.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--text", action="store_true", help="Mostrar el texto completo detectado por OCR."
    )
    args = parser.parse_args()

    try:
        with ExtractionPipeline() as pipeline:
            result = pipeline.process_bytes(args.image.read_bytes(), include_text=args.text)
    except (OcrError, ImageValidationError, ImageTooLargeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    payload = {
        "file": str(args.image),
        "emails": [
            {
                "email": hit.email,
                "confidence": hit.confidence,
                "occurrences": hit.occurrences,
            }
            for hit in result.emails
        ],
        "meta": {
            "width": result.width,
            "height": result.height,
            "elapsed_ms": result.elapsed_ms,
            "engine": result.engine,
            "model": result.model,
            "ocr_confidence": result.ocr_confidence,
        },
    }
    if args.text:
        payload["text"] = result.text

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
