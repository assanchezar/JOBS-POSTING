from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.services.ocr import OcrError
from app.services.pipeline import ExtractionPipeline
from app.services.preprocess import ImageTooLargeError, ImageValidationError
from scripts.make_fixtures import build_fixtures


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark del pipeline sobre un directorio de imagenes."
    )
    parser.add_argument("--dir", type=Path, default=Path("tests/fixtures/generated"))
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    if not args.dir.exists():
        build_fixtures(args.dir)

    images = sorted(
        path
        for path in args.dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    )
    if not images:
        raise SystemExit(f"No hay imagenes en {args.dir}")

    total_hits = 0
    total_ms = 0.0
    started = time.perf_counter()

    try:
        with ExtractionPipeline() as pipeline:
            for _ in range(args.repeat):
                for path in images:
                    result = pipeline.process_bytes(path.read_bytes())
                    total_hits += len(result.emails)
                    total_ms += result.elapsed_ms
                    found = ", ".join(hit.email for hit in result.emails) or "-"
                    print(
                        f"{path.name:20s} {result.elapsed_ms:8.1f} ms  "
                        f"{len(result.emails)} emails  {found}"
                    )

            wall_ms = (time.perf_counter() - started) * 1000
            count = len(images) * args.repeat
            print("\nResumen")
            print(f"  motor:           {pipeline.engine.model}")
            print(f"  imagenes:        {count}")
            print(f"  emails totales:  {total_hits}")
            print(f"  media pipeline:  {total_ms / count:.1f} ms")
            print(f"  pared media:     {wall_ms / count:.1f} ms")
    except (OcrError, ImageValidationError, ImageTooLargeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
