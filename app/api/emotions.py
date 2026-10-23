"""Endpoints REST de análisis (sección 4 del contrato)."""
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..deps import (DISCLAIMER_FACE, DISCLAIMER_INCONGRUENCE, DISCLAIMER_TEXT, authenticate, require_text,
                    require_visual, run_inference, validate_text)
from ..errors import ApiError, request_id_of
from ..fusion import fuse_full, valence_of
from ..labels import rounded, top
from ..services import media
from ..services.video import JobFailure, analyze_video, precheck_face_count, probe_video

router = APIRouter(prefix="/api/v1", dependencies=[Depends(authenticate)])


class TextIn(BaseModel):
    text: str
    language: str = "es"


def _rid(request: Request) -> str:
    return request_id_of(request)


def _sample_fps(value: Optional[float]) -> float:
    v = 2.0 if value is None else float(value)
    if not 1.0 <= v <= 5.0:
        raise ApiError(400, "VALIDATION_ERROR", "sampleFps debe estar entre 1 y 5.", {"field": "sampleFps"})
    return v


def _visual_weight(request: Request, value: Optional[float]) -> float:
    w = request.app.state.settings.default_visual_weight if value is None else float(value)
    if not 0.0 <= w <= 1.0:
        raise ApiError(400, "VALIDATION_ERROR", "visualWeight debe estar entre 0 y 1.", {"field": "visualWeight"})
    return w


# ------------------------------------------------------------------ texto
@router.post("/emotions/text")
async def analyze_text(request: Request, body: TextIn):
    st = request.app.state
    text = validate_text(body.text, body.language, st.settings)
    eng = require_text(request)
    t0 = time.perf_counter()
    probs, warns = await run_in_threadpool(run_inference, st, eng.predict, text)
    emo, conf = top(probs)
    return {
        "requestId": _rid(request), "modality": "TEXT", "emotion": emo, "confidence": round(conf, 4),
        "probabilities": rounded(probs), "inferenceMs": int((time.perf_counter() - t0) * 1000),
        "modelVersions": {"text": eng.version}, "warnings": [*warns, DISCLAIMER_TEXT],
    }


# ------------------------------------------------------------------ imagen
@router.post("/emotions/image")
async def analyze_image(request: Request, image: UploadFile = File(...)):
    st = request.app.state
    data = await media.read_limited(image, st.settings.max_image_bytes, "image")
    img = media.decode_image(data, "image", st.settings.max_image_pixels)
    del data  # no se conserva el contenido
    eng = require_visual(request)
    t0 = time.perf_counter()

    def work():
        faces = []
        for i, box in enumerate(eng.detect_faces(img)[: st.settings.max_faces], start=1):
            probs, _ = eng.embed(img, box)
            emo, conf = top(probs)
            faces.append({"faceId": f"face-{i}", "emotion": emo, "confidence": round(conf, 4), "probabilities": rounded(probs)})
        return faces

    faces = await run_in_threadpool(run_inference, st, work)
    warnings = [DISCLAIMER_FACE]
    if not faces:
        warnings.insert(0, "NO_FACE_DETECTED")
    if len(faces) > 1:
        warnings.append("Varios rostros: se reportan por separado y no se fusionan entre sí.")
    return {
        "requestId": _rid(request), "modality": "IMAGE", "facesDetected": len(faces), "faces": faces,
        "inferenceMs": int((time.perf_counter() - t0) * 1000),
        "modelVersions": {"visual": eng.version}, "warnings": warnings,
    }


# ------------------------------------------------------------------ video
def _public_video_result(job) -> dict:
    out = {k: v for k, v in (job.result or {}).items() if not k.startswith("_")}
    return out


@router.post("/emotions/video", status_code=202)
async def submit_video(request: Request, video: UploadFile = File(...), sampleFps: Optional[float] = Form(None)):
    st = request.app.state
    fps = _sample_fps(sampleFps)
    eng = require_visual(request)
    job_dir = media.new_job_dir(st.settings.tmp_dir)
    try:
        path = await media.save_video(video, job_dir, st.settings.max_video_bytes)
        _, duration = probe_video(path)
        if duration > st.settings.max_video_seconds + 0.5:
            raise ApiError(400, "VALIDATION_ERROR", f"El video supera {int(st.settings.max_video_seconds)} segundos.", {"field": "video"})
        job = st.jobs.submit("video", lambda: analyze_video(path, fps, eng, st.settings), job_dir)
    except BaseException:
        media.remove_dir(job_dir)
        raise
    return {"requestId": _rid(request), "analysisId": job.analysis_id, "status": job.status,
            "resultUrl": f"/api/v1/emotions/video/{job.analysis_id}", "modelVersions": {"visual": eng.version}}


@router.get("/emotions/video/{analysis_id}")
async def get_video(request: Request, analysis_id: str):
    st = request.app.state
    job = st.jobs.get(analysis_id, "video")
    if not job:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "Identificador de análisis inexistente o expirado.", {"analysisId": analysis_id})
    body = {"requestId": _rid(request), "analysisId": job.analysis_id, "status": job.status,
            "modelVersions": {"visual": st.visual.version}}
    if job.status == "COMPLETED":
        r = _public_video_result(job)
        r.setdefault("temporalModel", {})["windowFrames"] = st.visual.window
        body.update(r)
        body["inferenceMs"] = job.inference_ms
    elif job.status == "FAILED":
        body["error"] = job.error
        body["dominantExpression"], body["timeline"] = None, None
    else:
        body["dominantExpression"], body["timeline"] = None, None
    return body


# ------------------------------------------------------------------ multimodal
def _multimodal(text_probs, vis_probs, face_id, w, thr, versions, warnings) -> dict:
    t_emo, t_conf = top(text_probs)
    v_emo, v_conf = top(vis_probs)
    f = fuse_full(text_probs, vis_probs, w, thr)
    return {
        "modality": "MULTIMODAL",
        "text": {"emotion": t_emo, "valence": valence_of(t_emo), "confidence": round(t_conf, 4), "probabilities": rounded(text_probs)},
        "visual": {"emotion": v_emo, "valence": valence_of(v_emo), "confidence": round(v_conf, 4), "faceId": face_id,
                   "probabilities": rounded(vis_probs)},
        "fusion": {"method": "WEIGHTED_LATE_FUSION", "visualWeight": w, "textWeight": round(1 - w, 6),
                   "emotion": f["emotion"], "confidence": round(f["confidence"], 4),
                   "probabilities": rounded(f["probabilities"]), "incongruence": f["incongruence"],
                   "incongruenceThreshold": thr},
        "modelVersions": versions,
        "warnings": warnings,
    }


def _fusion_warnings(incongruent_possible: bool = True) -> list:
    return [DISCLAIMER_INCONGRUENCE, DISCLAIMER_FACE, DISCLAIMER_TEXT,
            "Pesos, umbral y valencias provienen de una propuesta experimental, no de una regla validada."]


@router.post("/emotions/analyze")
async def analyze_multimodal(
    request: Request,
    text: str = Form(...),
    language: str = Form("es"),
    image: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    visualWeight: Optional[float] = Form(None),
    sampleFps: Optional[float] = Form(None),
):
    st = request.app.state
    s = st.settings
    if (image is None) == (video is None):
        raise ApiError(400, "VALIDATION_ERROR", "Se exige exactamente uno de 'image' o 'video'.", {"fields": ["image", "video"]})
    validate_text(text, language, s)
    w = _visual_weight(request, visualWeight)
    t_eng, v_eng = require_text(request), require_visual(request)
    versions = {"text": t_eng.version, "visual": v_eng.version}
    t0 = time.perf_counter()
    text_probs, text_warns = await run_in_threadpool(run_inference, st, t_eng.predict, text)
    del text  # el texto no se conserva: solo se guardan sus probabilidades

    def count_error(n: int) -> ApiError:
        return ApiError(422, "FACE_COUNT_NOT_SUPPORTED_FOR_FUSION",
                        "La fusión requiere exactamente un rostro detectable.", {"facesDetected": n})

    if image is not None:
        data = await media.read_limited(image, s.max_image_bytes, "image")
        img = media.decode_image(data, "image", s.max_image_pixels)
        del data

        def work():
            boxes = v_eng.detect_faces(img)
            if len(boxes) != 1:
                raise count_error(len(boxes))
            return v_eng.embed(img, boxes[0])[0]

        vis_probs = await run_in_threadpool(run_inference, st, work)
        out = _multimodal(text_probs, vis_probs, "face-1", w, s.incongruence_threshold, versions,
                          [*text_warns, *_fusion_warnings()])
        out.update(requestId=_rid(request), inferenceMs=int((time.perf_counter() - t0) * 1000))
        return out

    # video: 202 + consulta posterior; comprobación síncrona del número de rostros
    fps = _sample_fps(sampleFps)
    job_dir = media.new_job_dir(s.tmp_dir)
    try:
        path = await media.save_video(video, job_dir, s.max_video_bytes)
        _, duration = probe_video(path)
        if duration > s.max_video_seconds + 0.5:
            raise ApiError(400, "VALIDATION_ERROR", f"El video supera {int(s.max_video_seconds)} segundos.", {"field": "video"})
        n = await run_in_threadpool(run_inference, st, precheck_face_count, path, v_eng)
        if n != 1:
            raise count_error(n)

        def job_fn():
            r = analyze_video(path, fps, v_eng, s)
            if r["multiFaceFrames"] > 0:
                raise JobFailure("FACE_COUNT_NOT_SUPPORTED_FOR_FUSION", "Se detectaron varios rostros en el video; la fusión exige uno solo.")
            if r.get("_meanProbs") is None:
                raise JobFailure("FACE_COUNT_NOT_SUPPORTED_FOR_FUSION", "No se detectó rostro en el video; no se produce fusión.")
            out = _multimodal(text_probs, r["_meanProbs"], "face-1", w, s.incongruence_threshold, versions,
                              [*text_warns, *_fusion_warnings()])
            out["visual"]["temporalModel"] = {"type": "LSTM", "windowFrames": v_eng.window,
                                              "framesAnalyzed": r["temporalModel"]["framesAnalyzed"],
                                              "framesWithFace": r["temporalModel"]["framesWithFace"]}
            out["samplingFps"], out["durationSeconds"] = fps, r["durationSeconds"]
            return out

        job = st.jobs.submit("analyze", job_fn, job_dir)
    except BaseException:
        media.remove_dir(job_dir)
        raise
    return JSONResponse(
        {"requestId": _rid(request), "analysisId": job.analysis_id, "status": job.status,
         "resultUrl": f"/api/v1/emotions/analyze/{job.analysis_id}", "modelVersions": versions},
        status_code=202,
    )


@router.get("/emotions/analyze/{analysis_id}")
async def get_analyze(request: Request, analysis_id: str):
    st = request.app.state
    job = st.jobs.get(analysis_id, "analyze")
    if not job:
        raise ApiError(404, "ANALYSIS_NOT_FOUND", "Identificador de análisis inexistente o expirado.", {"analysisId": analysis_id})
    body = {"requestId": _rid(request), "analysisId": job.analysis_id, "status": job.status}
    if job.status == "COMPLETED":
        body.update(job.result)
        body["inferenceMs"] = job.inference_ms
        body["requestId"] = _rid(request)
    else:
        body["modelVersions"] = {"text": st.text.version, "visual": st.visual.version}
        if job.status == "FAILED":
            body["error"] = job.error
    return body
