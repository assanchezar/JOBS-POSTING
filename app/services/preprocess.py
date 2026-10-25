from __future__ import annotations

import io

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from app.core.config import settings

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})

# Desactivamos el limite interno de PIL: controlamos el tamano antes de decodificar.
Image.MAX_IMAGE_PIXELS = None


class ImageValidationError(ValueError):
    """La imagen no se puede procesar."""


class ImageTooLargeError(ValueError):
    """La imagen supera los limites configurados."""


def load_image(data: bytes) -> Image.Image:
    """Valida y normaliza una imagen recibida como bytes."""
    if not data:
        raise ImageValidationError("El archivo esta vacio.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = (probe.format or "").upper()
            if fmt not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    f"Formato no soportado: {fmt or 'desconocido'}. "
                    f"Usa {', '.join(sorted(ALLOWED_FORMATS))}."
                )
            probe.verify()
    except ImageValidationError:
        raise
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        raise ImageValidationError("El archivo no es una imagen valida.") from exc

    img: Image.Image | None = None
    try:
        img = Image.open(io.BytesIO(data))
        width, height = img.size
        if width <= 0 or height <= 0:
            raise ImageValidationError("La imagen tiene dimensiones invalidas.")
        if width * height > settings.max_image_pixels:
            raise ImageTooLargeError(
                f"La imagen tiene {width * height} pixeles; el maximo es {settings.max_image_pixels}."
            )
        img.load()
    except ImageTooLargeError:
        if img is not None:
            img.close()
        raise
    except ImageValidationError:
        if img is not None:
            img.close()
        raise
    except MemoryError as exc:
        if img is not None:
            img.close()
        raise ImageTooLargeError("La imagen es demasiado grande para procesarla.") from exc
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        ValueError,
    ) as exc:
        if img is not None:
            img.close()
        raise ImageValidationError("El archivo no es una imagen valida.") from exc

    try:
        transposed = ImageOps.exif_transpose(img)
        if transposed is not img:
            img.close()
            img = transposed

        if img.mode in {"RGBA", "LA", "P"}:
            converted = img.convert("RGBA")
            background = Image.new("RGB", converted.size, (255, 255, 255))
            background.paste(converted, mask=converted.split()[-1])
            converted.close()
            img.close()
            img = background
        elif img.mode != "RGB":
            converted = img.convert("RGB")
            img.close()
            img = converted

        if max(img.size) > settings.max_side:
            img.thumbnail((settings.max_side, settings.max_side), Image.Resampling.LANCZOS)
    except MemoryError as exc:
        img.close()
        raise ImageTooLargeError("La imagen es demasiado grande para procesarla.") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        img.close()
        raise ImageValidationError("No se pudo normalizar la imagen.") from exc

    return img


# La versatil cache de calculo de deskew para peupe imgs
DESKEW_CACHE_THRESHOLD = 512
DESKEW_CACHE_SIZE = 2048
DESKEW_MIN_IMPROVEMENT = 1.08
DESKEW_SIDE = 320


def otsu_threshold(gray: Image.Image) -> int:
    """Calcula un umbral global de Otsu sobre una imagen en escala de grises."""
    histogram = gray.histogram()
    total = sum(histogram)
    if total <= 0:
        return 127

    sum_all = sum(index * count for index, count in enumerate(histogram))
    sum_background = 0
    weight_background = 0
    best_threshold = 127
    best_variance = -1.0

    for threshold, count in enumerate(histogram):
        weight_background += count
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += threshold * count
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        between_class = (
            weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        )
        if between_class > best_variance:
            best_variance = between_class
            best_threshold = threshold

    return best_threshold


def binarize_image(image: Image.Image, threshold: int | None = None) -> Image.Image:
    """Convierte la imagen a blanco y negro con Otsu o un umbral explicito."""
    gray = image.convert("L")
    value = otsu_threshold(gray) if threshold is None else threshold
    value = min(max(int(value), 0), 255)
    return gray.point(lambda pixel: 255 if pixel > value else 0, mode="L")


def _projection_score(binary: Image.Image) -> float:
    """Mide picos horizontales: las lineas de texto alineadas maximizan la varianza."""
    width, height = binary.size
    if width <= 0 or height <= 0:
        return 0.0

    dark = binary.point(lambda pixel: 1 if pixel < 128 else 0, mode="L").tobytes()
    counts = [sum(dark[row * width : (row + 1) * width]) for row in range(height)]
    if not counts:
        return 0.0

    mean = sum(counts) / len(counts)
    return sum((count - mean) ** 2 for count in counts)


def estimate_skew_angle(
    image: Image.Image,
    *,
    max_angle: float | None = None,
    step: float | None = None,
) -> float:
    """Estima el angulo de correccion de una pagina inclinada usando proyecciones.

    Devuelve el angulo que hay que aplicar con ``Image.rotate`` para dejar las
    lineas horizontales. Un resultado de 0.0 significa que no hace falta rotar.
    """
    max_angle = settings.ocr_deskew_max_angle if max_angle is None else max_angle
    step = settings.ocr_deskew_step if step is None else step
    if max_angle <= 0 or step <= 0:
        return 0.0

    try:
        gray = image.convert("L")
        gray = ImageOps.autocontrast(gray, cutoff=1)

        longest = max(gray.size)
        if longest > DESKEW_SIDE:
            scale = DESKEW_SIDE / longest
            gray = gray.resize(
                (max(1, round(gray.width * scale)), max(1, round(gray.height * scale))),
                Image.Resampling.BILINEAR,
            )

        # Ignora los bordes: suelen dominar la proyeccion y falsear el angulo.
        margin_x = max(1, gray.width // 10)
        margin_y = max(1, gray.height // 10)
        if gray.width > 2 * margin_x and gray.height > 2 * margin_y:
            gray = gray.crop((margin_x, margin_y, gray.width - margin_x, gray.height - margin_y))

        threshold = otsu_threshold(gray)
        binary = gray.point(lambda pixel: 255 if pixel > threshold else 0, mode="L")
        zero_score = _projection_score(binary)
        if zero_score <= 0:
            return 0.0

        steps = max(1, round(max_angle / step))
        best_angle = 0.0
        best_score = zero_score
        for index in range(-steps, steps + 1):
            angle = index * step
            if abs(angle) < 1e-9:
                continue
            rotated = binary.rotate(
                angle,
                resample=Image.Resampling.NEAREST,
                expand=False,
                fillcolor=255,
            )
            score = _projection_score(rotated)
            if score > best_score:
                best_score = score
                best_angle = angle

        if best_score < zero_score * DESKEW_MIN_IMPROVEMENT:
            return 0.0
        return round(best_angle, 2)
    except (OSError, ValueError):
        return 0.0


def prepare_for_ocr(
    image: Image.Image,
    *,
    binarize: bool = False,
    denoise: bool = False,
    deskew: bool | None = None,
) -> Image.Image:
    """Prepara la imagen para Tesseract con Pillow, sin numpy ni OpenCV."""
    try:
        prepared = image.convert("L")
        prepared = ImageOps.autocontrast(prepared, cutoff=1)

        use_deskew = settings.ocr_deskew if deskew is None else deskew
        if use_deskew:
            angle = estimate_skew_angle(prepared)
            if angle:
                rotated = prepared.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=True,
                    fillcolor=255,
                )
                prepared.close()
                prepared = rotated

        longest = max(prepared.size)
        if settings.ocr_upscale and longest < settings.ocr_upscale_min_side:
            scale = min(2.0, settings.max_side / longest)
            if scale > 1.0:
                resized = prepared.resize(
                    (round(prepared.width * scale), round(prepared.height * scale)),
                    Image.Resampling.LANCZOS,
                )
                prepared.close()
                prepared = resized

        # El deskew con expand puede superar el limite; se vuelve a acotar.
        if max(prepared.size) > settings.max_side:
            prepared.thumbnail(
                (settings.max_side, settings.max_side),
                Image.Resampling.LANCZOS,
            )

        if denoise:
            prepared = prepared.filter(ImageFilter.MedianFilter(3))

        prepared = prepared.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=3))
        if binarize:
            prepared = binarize_image(prepared)
        if settings.ocr_border > 0:
            prepared = ImageOps.expand(prepared, border=settings.ocr_border, fill=255)
        return prepared
    except MemoryError as exc:
        raise ImageTooLargeError("La imagen es demasiado grande para preparar el OCR.") from exc
    except (OSError, ValueError) as exc:
        raise ImageValidationError("No se pudo preparar la imagen para OCR.") from exc
