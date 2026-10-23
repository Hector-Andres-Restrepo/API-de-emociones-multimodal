"""Validación y lectura segura de archivos subidos (sin persistir imágenes)."""
import io
import os
import shutil
import uuid
from typing import Optional

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image

from ..errors import ApiError

JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def is_image_bytes(data: bytes) -> bool:
    return data.startswith(JPEG_MAGIC) or data.startswith(PNG_MAGIC)


def is_mp4_bytes(head: bytes) -> bool:
    return len(head) >= 12 and head[4:8] == b"ftyp"


async def read_limited(file: UploadFile, max_bytes: int, field: str) -> bytes:
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", f"El archivo supera el límite de {max_bytes // (1024 * 1024)} MB.", {"field": field})
    return data


def decode_image(data: bytes, field: str, max_pixels: int) -> np.ndarray:
    if not is_image_bytes(data):
        raise ApiError(415, "INVALID_FILE_FORMAT", "Formato no admitido. Use JPEG o PNG.", {"field": field})
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception:
        raise ApiError(415, "INVALID_FILE_FORMAT", "La imagen está corrupta o no es JPEG/PNG válido.", {"field": field})
    if w * h > max_pixels:
        raise ApiError(400, "VALIDATION_ERROR", "Resolución de imagen excesiva.", {"field": field, "maxPixels": max_pixels})
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ApiError(415, "INVALID_FILE_FORMAT", "No se pudo decodificar la imagen.", {"field": field})
    return img


def new_job_dir(tmp_dir: str) -> str:
    path = os.path.join(tmp_dir, f"job-{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=True)
    return path


def remove_dir(path: Optional[str]) -> None:
    if path:
        shutil.rmtree(path, ignore_errors=True)


async def save_video(file: UploadFile, job_dir: str, max_bytes: int, field: str = "video") -> str:
    """Escribe el video en disco por bloques, con límite de tamaño y verificación MP4."""
    dest = os.path.join(job_dir, "video.mp4")
    total, first = 0, b""
    with open(dest, "wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            if not first:
                first = chunk[:16]
                if not is_mp4_bytes(first):
                    raise ApiError(415, "INVALID_FILE_FORMAT", "Formato no admitido. Use MP4.", {"field": field})
            total += len(chunk)
            if total > max_bytes:
                raise ApiError(413, "PAYLOAD_TOO_LARGE", f"El video supera el límite de {max_bytes // (1024 * 1024)} MB.", {"field": field})
            out.write(chunk)
    if total == 0:
        raise ApiError(400, "VALIDATION_ERROR", "Archivo de video vacío.", {"field": field})
    return dest
