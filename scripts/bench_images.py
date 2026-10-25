from __future__ import annotations

import argparse
import atexit
import csv
import json
import random
import re
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.email_extract import normalize_text
from app.services.pipeline import ExtractionPipeline

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})

# Candidato laxo: sirve para detectar posibles emails que la validacion estricta
# descarto, no para devolverlos como resultado.
LOOSE_EMAIL_RE = re.compile(
    r"[a-z0-9._%+\-]{2,}@[a-z0-9.\-]{2,}\.[a-z0-9]{2,}",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"\+?\d[\d\s().\-]{6,}\d")
AT_CUE_RE = re.compile(r"@|\[\s*at\s*\]|\(\s*at\s*\)|\{\s*at\s*\}|\barroba\b", re.IGNORECASE)
EMAIL_CUE_RE = re.compile(
    r"\b(correo|email|e-mail|mail|cv|curriculum|postular|postulate|enviar|envia|contacto)\b",
    re.IGNORECASE,
)

_worker_pipeline: ExtractionPipeline | None = None


@dataclass(slots=True)
class ImageResult:
    file: str
    width: int = 0
    height: int = 0
    elapsed_ms: float = 0.0
    emails: list[dict[str, Any]] = field(default_factory=list)
    ocr_confidence: float | None = None
    text_length: int = 0
    text: str = ""
    loose_candidates: list[str] = field(default_factory=list)
    phone_numbers: list[str] = field(default_factory=list)
    has_at: bool = False
    email_cue: bool = False
    error: str = ""

    @property
    def email_addresses(self) -> list[str]:
        return [item["email"] for item in self.emails]

    @property
    def possible_misses(self) -> list[str]:
        extracted = {email.lower() for email in self.email_addresses}
        return [candidate for candidate in self.loose_candidates if candidate not in extracted]

    @property
    def at_without_email(self) -> bool:
        return self.has_at and not self.emails

    @property
    def phone_only(self) -> bool:
        return bool(self.phone_numbers) and not self.emails

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "elapsed_ms": self.elapsed_ms,
            "email_count": len(self.emails),
            "emails": self.emails,
            "email_addresses": self.email_addresses,
            "ocr_confidence": self.ocr_confidence,
            "text_length": self.text_length,
            "text": self.text,
            "loose_candidates": self.loose_candidates,
            "possible_misses": self.possible_misses,
            "has_at": self.has_at,
            "at_without_email": self.at_without_email,
            "email_cue": self.email_cue,
            "phone_numbers": self.phone_numbers,
            "phone_only": self.phone_only,
            "error": self.error,
        }


def _get_pipeline() -> ExtractionPipeline:
    global _worker_pipeline
    if _worker_pipeline is None:
        _worker_pipeline = ExtractionPipeline()
    return _worker_pipeline


def _close_pipeline() -> None:
    global _worker_pipeline
    if _worker_pipeline is not None:
        _worker_pipeline.close()
        _worker_pipeline = None


def _configure(no_fallback: bool) -> None:
    if not no_fallback:
        return
    settings.ocr_binarize_fallback = False
    settings.ocr_denoise_fallback = False
    settings.ocr_bottom_strip_fallback = False
    # Sin psm alternativo: mide solo el primer pase base.
    settings.tesseract_fallback_confidence = 0.0


def _init_worker(no_fallback: bool = False) -> None:
    _configure(no_fallback)
    _get_pipeline()
    atexit.register(_close_pipeline)


def loose_candidates(text: str) -> list[str]:
    normalized = normalize_text(text)
    normalized = re.sub(r"\s*@\s*", "@", normalized)
    normalized = re.sub(r"\s*\.\s*", ".", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return sorted(
        {match.lower().strip(".,;:()[]<>\"'") for match in LOOSE_EMAIL_RE.findall(normalized)}
    )


def phone_candidates(text: str) -> list[str]:
    found: set[str] = set()
    for match in PHONE_RE.finditer(text):
        raw = match.group(0).strip()
        digits = re.sub(r"\D", "", raw)
        if 8 <= len(digits) <= 15:
            found.add(raw)
    return sorted(found)


def process_one(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    try:
        data = path.read_bytes()
        result = _get_pipeline().process_bytes(data, include_text=True)
        text = result.text
        emails = [
            {
                "email": hit.email,
                "confidence": hit.confidence,
                "occurrences": hit.occurrences,
            }
            for hit in result.emails
        ]
        return ImageResult(
            file=path.name,
            width=result.width,
            height=result.height,
            elapsed_ms=result.elapsed_ms,
            emails=emails,
            ocr_confidence=result.ocr_confidence,
            text_length=len(text),
            text=text,
            loose_candidates=loose_candidates(text),
            phone_numbers=phone_candidates(text),
            has_at=bool(AT_CUE_RE.search(text)),
            email_cue=bool(EMAIL_CUE_RE.search(text)),
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        return ImageResult(
            file=path.name,
            error=f"{type(exc).__name__}: {exc}",
        ).to_dict()


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((percentile / 100.0) * (len(ordered) - 1))
    return ordered[min(max(index, 0), len(ordered) - 1)]


def _load_ground_truth(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit("El ground truth debe ser un objeto JSON: {archivo: [emails]}")

    ground_truth: dict[str, list[str]] = {}
    for filename, expected in raw.items():
        if isinstance(expected, str):
            values = [expected]
        elif isinstance(expected, list):
            values = [str(value) for value in expected]
        else:
            raise SystemExit(f"Valor invalido para {filename}: se esperaba lista o texto")
        ground_truth[str(filename)] = sorted(
            {value.strip().lower() for value in values if value.strip()}
        )
    return ground_truth


def _apply_ground_truth(
    results: list[dict[str, Any]],
    ground_truth: dict[str, list[str]],
) -> dict[str, Any] | None:
    if not ground_truth:
        return None

    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_matches = 0
    images_with_gt = 0

    for result in results:
        filename = result["file"]
        if filename not in ground_truth:
            continue
        images_with_gt += 1
        expected = set(ground_truth[filename])
        actual = {email.lower() for email in result["email_addresses"]}
        matched = sorted(expected & actual)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        true_positive += len(matched)
        false_positive += len(extra)
        false_negative += len(missing)
        if not missing and not extra:
            exact_matches += 1
        result["ground_truth"] = {
            "expected": sorted(expected),
            "matched": matched,
            "missing": missing,
            "extra": extra,
            "exact_match": not missing and not extra,
        }

    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "images": images_with_gt,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "exact_match_rate": round(exact_matches / images_with_gt, 4) if images_with_gt else None,
    }


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [result for result in results if result["error"]]
    ok = [result for result in results if not result["error"]]
    with_email = [result for result in ok if result["emails"]]
    possible_misses = [result for result in ok if result["possible_misses"]]
    at_without_email = [result for result in ok if result["at_without_email"]]
    email_cue_without_email = [
        result for result in ok if result["email_cue"] and not result["emails"]
    ]
    phone_only = [result for result in ok if result["phone_only"]]
    elapsed = [result["elapsed_ms"] for result in ok if result["elapsed_ms"] > 0]
    confidences = [
        result["ocr_confidence"] for result in ok if result["ocr_confidence"] is not None
    ]
    total_emails = sum(len(result["emails"]) for result in ok)

    return {
        "total_images": len(results),
        "processed": len(ok),
        "errors": len(errors),
        "images_with_email": len(with_email),
        "coverage_pct": round(len(with_email) / len(ok) * 100, 2) if ok else 0.0,
        "total_emails": total_emails,
        "email_count_distribution": {
            "0": sum(1 for result in ok if not result["emails"]),
            "1": sum(1 for result in ok if len(result["emails"]) == 1),
            "2+": sum(1 for result in ok if len(result["emails"]) >= 2),
        },
        "possible_miss_images": len(possible_misses),
        "at_without_email_images": len(at_without_email),
        "email_cue_without_email_images": len(email_cue_without_email),
        "phone_only_images": len(phone_only),
        "avg_elapsed_ms": round(statistics.mean(elapsed), 1) if elapsed else None,
        "median_elapsed_ms": round(statistics.median(elapsed), 1) if elapsed else None,
        "p95_elapsed_ms": round(_percentile(elapsed, 95), 1) if elapsed else None,
        "avg_ocr_confidence": round(statistics.mean(confidences), 4) if confidences else None,
        "low_resolution_images": sum(
            1 for result in ok if result["width"] < 800 or result["height"] < 800
        ),
    }


def _text_excerpt(text: str, limit: int = 220) -> str:
    compact = " ".join(text.split())
    return compact[:limit]


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "file",
        "width",
        "height",
        "elapsed_ms",
        "email_count",
        "emails",
        "ocr_confidence",
        "text_length",
        "possible_misses",
        "at_without_email",
        "email_cue",
        "phone_numbers",
        "phone_only",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            row = dict(result)
            row["emails"] = "; ".join(result["email_addresses"])
            row["possible_misses"] = "; ".join(result["possible_misses"])
            row["phone_numbers"] = "; ".join(result["phone_numbers"])
            writer.writerow(row)


def _write_review_csv(path: Path, results: list[dict[str, Any]]) -> None:
    fields = [
        "file",
        "emails",
        "possible_misses",
        "ocr_confidence",
        "text_excerpt",
        "expected_emails",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if not (result["emails"] or result["possible_misses"] or result["at_without_email"]):
                continue
            writer.writerow(
                {
                    "file": result["file"],
                    "emails": "; ".join(result["email_addresses"]),
                    "possible_misses": "; ".join(result["possible_misses"]),
                    "ocr_confidence": result["ocr_confidence"],
                    "text_excerpt": _text_excerpt(result["text"]),
                    "expected_emails": "",
                }
            )


def _write_ground_truth_template(
    path: Path,
    results: list[dict[str, Any]],
) -> None:
    template = {
        result["file"]: []
        for result in results
        if result["emails"] or result["possible_misses"] or result["at_without_email"]
    }
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(
    path: Path,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    ground_truth_summary: dict[str, Any] | None,
) -> None:
    possible_misses = [result for result in results if result["possible_misses"]]
    at_without_email = [result for result in results if result["at_without_email"]]
    email_cue_without_email = [
        result for result in results if result["email_cue"] and not result["emails"]
    ]
    phone_only = [result for result in results if result["phone_only"]]
    errors = [result for result in results if result["error"]]

    lines = [
        "# Reporte OCR de ofertas laborales",
        "",
        f"Generado: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "## Resumen",
        "",
        (
            "- Modo: base (sin fallbacks)"
            if summary.get("mode") == "base"
            else "- Modo: completo (con fallbacks)"
        ),
        f"- Imagenes: {summary['total_images']}",
        f"- Procesadas: {summary['processed']}",
        f"- Errores: {summary['errors']}",
        (
            f"- Imagenes con al menos un email: {summary['images_with_email']} "
            f"({summary['coverage_pct']}%)"
        ),
        f"- Emails extraidos: {summary['total_emails']}",
        f"- Distribucion: {summary['email_count_distribution']}",
        f"- Posibles omisiones: {summary['possible_miss_images']} imagenes",
        f"- Imagenes con @ pero sin email: {summary['at_without_email_images']}",
        (
            f"- Imagenes con senales de correo pero sin email: "
            f"{summary['email_cue_without_email_images']}"
        ),
        f"- Imagenes solo con telefono/WhatsApp: {summary['phone_only_images']}",
        f"- Tiempo medio: {summary['avg_elapsed_ms']} ms",
        f"- Tiempo mediano: {summary['median_elapsed_ms']} ms",
        f"- Percentil 95: {summary['p95_elapsed_ms']} ms",
        f"- Confianza OCR media: {summary['avg_ocr_confidence']}",
        f"- Imagenes de baja resolucion (<800 px): {summary['low_resolution_images']}",
        "",
    ]

    if ground_truth_summary:
        lines.extend(
            [
                "## Ground truth",
                "",
                f"- Imagenes evaluadas: {ground_truth_summary['images']}",
                f"- Precision: {ground_truth_summary['precision']}",
                f"- Recall: {ground_truth_summary['recall']}",
                f"- F1: {ground_truth_summary['f1']}",
                f"- Exactitud por imagen: {ground_truth_summary['exact_match_rate']}",
                "",
            ]
        )

    lines.extend(["## Posibles omisiones", ""])
    if possible_misses:
        lines.extend(
            [
                "| Archivo | Emails extraidos | Candidatos laxos | Confianza | Texto |",
                "|---|---|---|---:|---|",
            ]
        )
        for result in possible_misses[:40]:
            emails = ", ".join(result["email_addresses"]) or "-"
            candidates = ", ".join(result["possible_misses"])
            text = _text_excerpt(result["text"], 120).replace("|", "/")
            lines.append(
                f"| {result['file']} | {emails} | {candidates} | "
                f"{result['ocr_confidence']} | {text} |"
            )
        if len(possible_misses) > 40:
            lines.append(f"| ... | ... | ... | ... | +{len(possible_misses) - 40} mas |")
    else:
        lines.append("No se detectaron candidatos laxos sin extraer.")
    lines.append("")

    lines.extend(["## Imagenes con @ pero sin email", ""])
    if at_without_email:
        for result in at_without_email[:30]:
            text = _text_excerpt(result["text"], 160).replace("|", "/")
            lines.append(f"- `{result['file']}`: {text}")
        if len(at_without_email) > 30:
            lines.append(f"- ... y {len(at_without_email) - 30} mas")
    else:
        lines.append("Ninguna.")
    lines.append("")

    lines.extend(["## Imagenes con senales de correo pero sin email", ""])
    if email_cue_without_email:
        for result in email_cue_without_email[:30]:
            text = _text_excerpt(result["text"], 160).replace("|", "/")
            lines.append(f"- `{result['file']}`: {text}")
        if len(email_cue_without_email) > 30:
            lines.append(f"- ... y {len(email_cue_without_email) - 30} mas")
    else:
        lines.append("Ninguna.")
    lines.append("")

    lines.extend(["## Imagenes solo con telefono/WhatsApp", ""])
    lines.append(
        f"{len(phone_only)} imagenes tienen numeros de telefono detectados pero ningun email."
    )
    for result in phone_only[:20]:
        lines.append(f"- `{result['file']}`: {', '.join(result['phone_numbers'][:3])}")
    if len(phone_only) > 20:
        lines.append(f"- ... y {len(phone_only) - 20} mas")
    lines.append("")

    lines.extend(["## Errores", ""])
    if errors:
        for result in errors[:30]:
            lines.append(f"- `{result['file']}`: {result['error']}")
    else:
        lines.append("Ninguno.")
    lines.append("")

    lines.extend(
        [
            "## Recomendaciones",
            "",
            (
                "- Revisar manualmente las imagenes marcadas como posibles omisiones; "
                "suelen ser emails partidos por saltos de linea o con confusiones de OCR."
            ),
            (
                "- Si predominan imagenes solo con telefono/WhatsApp, conviene agregar "
                "extraccion de telefonos como campo separado."
            ),
            (
                "- Para imagenes de baja resolucion, probar `OCR_UPSCALE_MIN_SIDE` mas alto "
                "y revisar `OCR_DESKEW`/binarizacion."
            ),
            (
                "- Si la latencia importa, desactivar las pasadas de respaldo o limitar "
                "`MAX_IMAGE_SIDE`."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _print_progress(index: int, total: int, result: dict[str, Any], verbose: bool) -> None:
    if verbose or index == total or index % 25 == 0 or result["error"]:
        found = len(result["emails"])
        print(
            f"[{index}/{total}] {result['file']} emails={found} "
            f"ms={result['elapsed_ms']:.0f} error={bool(result['error'])}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Procesa un directorio de imagenes y genera reportes de extraccion."
    )
    parser.add_argument("--images-dir", type=Path, default=Path("tests/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--limit", type=int, default=0, help="0 procesa todas")
    parser.add_argument(
        "--files-from",
        type=Path,
        default=None,
        help="Archivo de texto con un nombre de imagen por linea.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Toma una muestra aleatoria de N imagenes (0 = todas).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla para --sample.")
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Desactiva binarizacion, denoise y psm alternativo; mide solo el pase base.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--ground-truth", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    images = sorted(
        path
        for path in args.images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.files_from:
        names = {
            line.strip()
            for line in args.files_from.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        missing = sorted(names - {path.name for path in images})
        if missing:
            print(f"Aviso: {len(missing)} archivos de --files-from no existen.", flush=True)
        images = [path for path in images if path.name in names]
    if args.limit > 0:
        images = images[: args.limit]
    if args.sample > 0 and args.sample < len(images):
        images = sorted(random.Random(args.seed).sample(images, args.sample))
    if not images:
        raise SystemExit(f"No hay imagenes en {args.images_dir}")

    _configure(args.no_fallback)

    ground_truth = (
        _load_ground_truth(args.ground_truth)
        if args.ground_truth and args.ground_truth.exists()
        else {}
    )
    if args.ground_truth and not args.ground_truth.exists():
        print(f"Aviso: no existe el ground truth {args.ground_truth}", flush=True)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    try:
        if args.workers <= 1:
            for index, path in enumerate(images, 1):
                result = process_one(str(path))
                results.append(result)
                _print_progress(index, len(images), result, args.verbose)
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(args.no_fallback,),
            ) as executor:
                futures = {executor.submit(process_one, str(path)): path for path in images}
                for index, future in enumerate(as_completed(futures), 1):
                    path = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # noqa: BLE001
                        result = ImageResult(
                            file=path.name,
                            error=f"{type(exc).__name__}: {exc}",
                        ).to_dict()
                    results.append(result)
                    _print_progress(index, len(images), result, args.verbose)
    finally:
        _close_pipeline()

    results.sort(key=lambda item: item["file"])
    ground_truth_summary = _apply_ground_truth(results, ground_truth)
    summary = _build_summary(results)
    summary["ground_truth"] = ground_truth_summary
    summary["wall_ms"] = round((time.perf_counter() - started) * 1000, 1)
    summary["mode"] = "base" if args.no_fallback else "full"
    summary["seed"] = args.seed
    summary["sample_requested"] = args.sample

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ocr_images_report.json"
    csv_path = args.output_dir / "ocr_images_report.csv"
    md_path = args.output_dir / "ocr_images_report.md"
    review_path = args.output_dir / "review_candidates.csv"
    template_path = args.output_dir / "ground_truth_template.json"

    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "images_dir": str(args.images_dir),
        "summary": summary,
        "results": results,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, results)
    _write_markdown(md_path, summary, results, ground_truth_summary)
    _write_review_csv(review_path, results)
    _write_ground_truth_template(template_path, results)

    print("\nResumen")
    print(f"  modo:                     {summary['mode']}")
    print(f"  imagenes:                 {summary['total_images']}")
    print(f"  procesadas:               {summary['processed']}")
    print(f"  errores:                  {summary['errors']}")
    print(
        f"  con email:                {summary['images_with_email']} ({summary['coverage_pct']}%)"
    )
    print(f"  emails extraidos:         {summary['total_emails']}")
    print(f"  posibles omisiones:       {summary['possible_miss_images']}")
    print(f"  @ sin email:              {summary['at_without_email_images']}")
    print(f"  senales correo sin email: {summary['email_cue_without_email_images']}")
    print(f"  solo telefono/WhatsApp:   {summary['phone_only_images']}")
    print(f"  tiempo medio:             {summary['avg_elapsed_ms']} ms")
    print(f"  tiempo mediano:           {summary['median_elapsed_ms']} ms")
    print(f"  percentil 95:             {summary['p95_elapsed_ms']} ms")
    print(f"  confianza OCR media:      {summary['avg_ocr_confidence']}")
    if ground_truth_summary:
        print(f"  precision:                {ground_truth_summary['precision']}")
        print(f"  recall:                   {ground_truth_summary['recall']}")
        print(f"  f1:                       {ground_truth_summary['f1']}")
        print(f"  exactitud por imagen:     {ground_truth_summary['exact_match_rate']}")
    print(
        f"\nReportes:\n  {json_path}\n  {csv_path}\n  {md_path}\n  {review_path}\n  {template_path}"
    )


if __name__ == "__main__":
    main()
