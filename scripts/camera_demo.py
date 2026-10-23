"""Cliente de demostración de cámara en vivo: captura fotogramas EN EL CLIENTE y los envía por WebSocket.

    python scripts/camera_demo.py --token $T --fps 2 --seconds 20             # webcam
    python scripts/camera_demo.py --token $T --source samples/clip.mp4        # usa un video como "cámara"

Muestra una ventana con la emoción detectada (tecla q para salir; --no-show la desactiva).
El contenedor nunca accede a la cámara. Requiere consentimiento de quien aparece en cuadro.
Guarda el JSON de la sesión y estadísticas de latencia en --out (no guarda imágenes).
"""
import argparse
import asyncio
import json
import os
import statistics
import time

import cv2
import httpx
import websockets


def open_source(source, camera_index):
    if source:
        return cv2.VideoCapture(source)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened() and hasattr(cv2, "CAP_DSHOW"):  # en Windows, DirectShow suele abrir cuando el backend por defecto falla
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cap


def draw_overlay(frame, msg):
    faces = msg.get("faces") or []
    if faces:
        f = faces[0]
        text = f"{f['emotion']} {f['confidence'] * 100:.0f}% [{f.get('mode', '')}]"
    else:
        text = "Sin rostro"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    t = msg.get("temporal")
    if t:
        info = f"ventana {t['windowFilled']}/{t['windowFrames']}  inferencia {msg.get('inferenceMs', 0)} ms"
        cv2.putText(frame, info, (8, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--token", required=True)
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--camera", type=int, default=0, help="índice de la webcam")
    ap.add_argument("--source", help="ruta de un video para usar en lugar de la webcam")
    ap.add_argument("--no-show", action="store_true", help="no abrir ventana de vista previa")
    ap.add_argument("--out", default="evidence/06_camera_stream.json")
    a = ap.parse_args()

    r = httpx.post(f"{a.base}/api/v1/emotions/streams", json={"sampleFps": a.fps},
                   headers={"Authorization": f"Bearer {a.token}"}, timeout=30)
    if r.status_code != 201:
        raise SystemExit(f"No se pudo crear la sesión: HTTP {r.status_code} {r.text}")
    s = r.json()

    cap = open_source(a.source, a.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/fuente. Pruebe --camera 1 o cierre otras apps que la usen.")
    for _ in range(5):  # descarta fotogramas iniciales (autoexposición)
        cap.read()

    show = not a.no_show
    ws_url = a.base.replace("http", "ws", 1) + s["websocketPath"]
    msgs, rtts, infs = [], [], []
    async with websockets.connect(ws_url, max_size=2 ** 22) as ws:
        await ws.send(json.dumps({"type": "auth", "token": s["streamToken"]}))  # token efímero; no va en la URL
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"Autenticación rechazada: {ready}")
        print(f"Conectado. Ventana temporal: {ready['windowFrames']} fotogramas, {ready['sampleFps']} fps. Ctrl+C para salir.")
        start = time.time()
        i = 0
        while time.time() - start < a.seconds:
            ok, frame = cap.read()
            if not ok:
                print("Fin de la fuente de video.")
                break
            frame = cv2.resize(frame, (640, 480))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            t0 = time.perf_counter()
            await ws.send(jpg.tobytes())
            m = json.loads(await ws.recv())
            rtt = int((time.perf_counter() - t0) * 1000)
            if m.get("error"):
                print("Error del servidor:", m["error"])
                continue
            if not m.get("dropped"):
                m["clientRoundTripMs"] = rtt
                msgs.append(m)
                rtts.append(rtt)
                infs.append(m.get("inferenceMs", 0))
                f = (m.get("faces") or [{}])[0]
                print(f"t={m.get('timestampMs')} ms | rostros={m.get('facesDetected')} | {f.get('emotion', '-')} "
                      f"{f.get('confidence', 0):.2f} [{f.get('mode', '-')}] | inferencia {m.get('inferenceMs')} ms | ida y vuelta {rtt} ms")
                if show:
                    try:
                        cv2.imshow("Demo emociones (q para salir)", draw_overlay(frame, m))
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except cv2.error:
                        print("Esta instalación de OpenCV no puede abrir ventanas; continúo sin vista previa.")
                        show = False
            i += 1
            await asyncio.sleep(max(0.0, start + i / a.fps - time.time()))
        await ws.send(json.dumps({"type": "close"}))
        try:
            closing = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
            print("Cierre ordenado:", closing)
        except Exception:
            pass
    cap.release()
    if show:
        cv2.destroyAllWindows()

    stats = {}
    if rtts:
        stats = {"framesProcessed": len(msgs), "framesWithFace": sum(1 for m in msgs if m.get("facesDetected")),
                 "inferenceMsMean": round(statistics.mean(infs), 1), "inferenceMsMax": max(infs),
                 "roundTripMsMean": round(statistics.mean(rtts), 1), "roundTripMsMax": max(rtts)}
    print("Resumen:", stats)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"session": {k: s[k] for k in ("sessionId", "sampleFps", "windowFrames", "modelVersions")},
                   "stats": stats, "messages": msgs}, f, indent=2, ensure_ascii=False)
    print("Guardado en", a.out)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
