# Email OCR API

API local con FastAPI que extrae emails de imagenes reales de ofertas laborales con Tesseract (local, sin APIs de nube).

## Estado del pipeline

- Modelo: `tessdata` original.
- Pipeline completo: preprocesado (deskew, upscale, binarize, denoise, bottom-strip) + smart_retry.
- Smart retry (`_smart_retry_text`): opera solo sobre texto existente, sin nuevo OCR; activa solo si hay contexto (`email`/`contacto`) y candidatos laxos (`LOOSE_EMAIL_RE`).
- Seguridad: `secrets.compare_digest()` para `API_KEYS`, limite de bytes (`_read_limited`), `BoundedSemaphore` para concurrencia.

## Mejoras aplicadas (v0.6.1)

- Correccion `_extract_bottom_strip`: cierre seguro de `crop` y `prepared`.
- Caché `language`: `_cached_languages` evita llamadas repetidas a Tesseract.
- Deskew acelerado: referencia de 480px a 320px (`DESKEW_SIDE`).
- Smart retry: regex puro, sin relentizar. Cobertura 56.0% (100 img), 0 errores.

## Benchmark final

| Metrica | Base (844 img) | Profundo (100 img, full) |
|---|---|---|
| Cobertura | 44.79% (390 emails) | 56.0% (57 emails) |
| Posibles omisiones | 7 | 3 |
| Confianza OCR media | 0.7706 | 0.7767 |
| Tiempo medio | 3302 ms | 5932-6815 ms |
| Errores | 0 | 0 |

## Uso

```bash
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Endpoint `POST /v1/extract` con `API_KEYS` opcional.
