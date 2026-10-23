"""Genera evidencias REALES por modalidad contra la API en ejecución (no usa mocks).

    python scripts/collect_evidence.py --token $T --image samples/rostro.jpg --video samples/clip.mp4 \
        --text "Hoy me siento muy feliz" [--base http://localhost:8000] [--out evidence]

Guarda en --out: JSON de cada respuesta (con requestId, modelVersions e inferenceMs) y SUMMARY.md con una tabla.
Con --image corre texto, imagen y multimodal con imagen; con --video corre video y multimodal con video.
Requiere consentimiento de las personas que aparecen en imagen/video. No se guardan los medios.
"""
import argparse
import json
import pathlib
import time

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--token", required=True)
    ap.add_argument("--text", default="Hoy me siento muy feliz")
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--out", default="evidence")
    ap.add_argument("--no-multimodal-video", action="store_true", help="omite /analyze con video (ahorra tiempo)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    h = {"Authorization": f"Bearer {a.token}"}
    api = f"{a.base}/api/v1"
    rows = []

    def save(name, resp, extra=None):
        data = resp.json()
        if extra:
            data.update(extra)
        (out / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append((name, resp.status_code, data.get("modelVersions"), data.get("inferenceMs"), data.get("wallClockSeconds")))
        print(name, resp.status_code, data.get("inferenceMs"), "ms")
        return data

    def wait_job(c, url, t0):
        while True:
            r = c.get(f"{a.base}{url}", headers=h)
            if r.json().get("status") in ("COMPLETED", "FAILED"):
                return r, {"wallClockSeconds": round(time.perf_counter() - t0, 2)}
            time.sleep(1)

    with httpx.Client(timeout=600) as c:
        ready = c.get(f"{a.base}/health/ready")
        (out / "health_ready.json").write_text(json.dumps(ready.json(), indent=2), encoding="utf-8")
        save("01_text", c.post(f"{api}/emotions/text", json={"text": a.text, "language": "es"}, headers=h))
        if a.image:
            with open(a.image, "rb") as f:
                save("02_image", c.post(f"{api}/emotions/image", files={"image": f}, headers=h))
            with open(a.image, "rb") as f:
                save("05_analyze_image", c.post(f"{api}/emotions/analyze", data={"text": a.text}, files={"image": f}, headers=h))
        if a.video:
            t0 = time.perf_counter()
            with open(a.video, "rb") as f:
                sub = c.post(f"{api}/emotions/video", files={"video": f}, data={"sampleFps": "2"}, headers=h)
            sub_data = save("03_video_submit", sub)
            r, extra = wait_job(c, sub_data["resultUrl"], t0)
            save("04_video_result", r, extra)
            if not a.no_multimodal_video:
                t0 = time.perf_counter()
                with open(a.video, "rb") as f:
                    sub = c.post(f"{api}/emotions/analyze", data={"text": a.text}, files={"video": f}, headers=h)
                sub_data = save("05b_analyze_video_submit", sub)
                if sub.status_code == 202:
                    r, extra = wait_job(c, sub_data["resultUrl"], t0)
                    save("05c_analyze_video_result", r, extra)
    lines = ["# Resumen de evidencias reales", "", "| Archivo | HTTP | modelVersions | inferenceMs (servidor) | Segundos totales (video) |", "|---|---|---|---|---|"]
    lines += [f"| {n}.json | {s} | `{m}` | {t if t is not None else '-'} | {w if w is not None else '-'} |" for n, s, m, t, w in rows]
    lines += ["", "La evidencia de cámara se genera con `scripts/camera_demo.py` (guarda evidence/06_camera_stream.json)."]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
