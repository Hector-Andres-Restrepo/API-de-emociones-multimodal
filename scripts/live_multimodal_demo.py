"""Cliente de demostración MULTIMODAL EN VIVO: cámara + texto que se escribe, fusionados de forma continua.

    python scripts/live_multimodal_demo.py --token $T --typing "Estoy muy triste por la noticia"      # webcam
    python scripts/live_multimodal_demo.py --token $T --source samples/clip.mp4 --typing "Estoy feliz"  # video como "cámara"

El texto se "teclea" palabra por palabra (--typing) con una pausa tras cada palabra, igual que haría un usuario;
cada pausa se envía como mensaje de texto y el servidor responde con rostro + texto + fusión + incongruencia.
Los fotogramas se capturan EN EL CLIENTE; el contenedor nunca accede a la cámara. Requiere consentimiento de quien
aparece en cuadro. Guarda en --out el JSON de la sesión (probabilidades y métricas; NO imágenes ni el texto).
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
    if not cap.isOpened() and hasattr(cv2, "CAP_DSHOW"):
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    return cap


def line(m):
    v, t, f = m.get("visual"), m.get("text"), m.get("fusion")
    return (f"[{m['trigger']:<6}] rostros={m.get('facesDetected')} | "
            f"cara={v['emotion'] + ' %.2f' % v['confidence'] if v else '-'} | "
            f"texto={t['emotion'] + ' %.2f' % t['confidence'] if t else '-'} | "
            f"fusión={f['emotion'] + ' %.2f' % f['confidence'] if f else m['fusionStatus']} | "
            f"incongruencia={f['incongruence'] if f else '-'} | {m.get('inferenceMs')} ms")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--token", required=True)
    ap.add_argument("--fps", type=float, default=2)
    ap.add_argument("--weight", type=float, default=0.5, help="peso visual 0..1")
    ap.add_argument("--window", type=float, default=5, help="ventana visual en segundos")
    ap.add_argument("--typing", default="Estoy muy triste por la noticia", help="texto a teclear palabra por palabra")
    ap.add_argument("--word-delay", type=float, default=0.8, help="segundos entre palabras")
    ap.add_argument("--seconds", type=int, default=25)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--source", help="video para usar en lugar de la webcam")
    ap.add_argument("--out", default="evidence/07_live_multimodal.json")
    a = ap.parse_args()

    r = httpx.post(f"{a.base}/api/v1/emotions/live", headers={"Authorization": f"Bearer {a.token}"}, timeout=30,
                   json={"sampleFps": a.fps, "visualWeight": a.weight, "visualWindowSeconds": a.window, "language": "es"})
    if r.status_code != 201:
        raise SystemExit(f"No se pudo crear la sesión: HTTP {r.status_code} {r.text}")
    s = r.json()
    cap = open_source(a.source, a.camera)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/fuente. Pruebe --camera 1 o cierre otras apps que la usen.")
    for _ in range(5):
        cap.read()

    words = a.typing.split()
    updates, drops = [], {"FRAME": 0, "TEXT": 0}
    state = {"stop": False}
    ws_url = a.base.replace("http", "ws", 1) + s["websocketPath"]
    async with websockets.connect(ws_url, max_size=2 ** 22) as ws:
        await ws.send(json.dumps({"type": "auth", "token": s["streamToken"]}))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            raise SystemExit(f"Autenticación rechazada: {ready}")
        print(f"Conectado. {ready['sampleFps']} fps · ventana {ready['visualWindowSeconds']} s · umbral {ready['incongruenceThreshold']}")
        pending_text = {"v": 0, "last": None}

        async def reader():
            try:
                async for raw in ws:
                    m = json.loads(raw)
                    if m["type"] == "update":
                        updates.append(m)
                        print(line(m))
                    elif m["type"] == "dropped":
                        drops[m["kind"]] += 1
                        if m["kind"] == "TEXT":
                            pending_text["last"] = None  # se reenviará el último texto
                    elif m["type"] == "error":
                        print("Error del servidor:", m["error"])
                    elif m["type"] == "closing":
                        print("Cierre ordenado:", m["summary"])
                        state["stop"] = True
                        return
            except websockets.ConnectionClosed:
                state["stop"] = True

        async def frames():
            start, i = time.time(), 0
            while not state["stop"] and time.time() - start < a.seconds:
                ok, frame = cap.read()
                if not ok:
                    print("Fin de la fuente de video.")
                    break
                ok, jpg = cv2.imencode(".jpg", cv2.resize(frame, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                await ws.send(jpg.tobytes())
                i += 1
                await asyncio.sleep(max(0.0, start + i / a.fps - time.time()))
            state["stop"] = True

        async def typist():
            await asyncio.sleep(2)  # unos segundos de cara sin texto
            for n in range(1, len(words) + 1):
                if state["stop"]:
                    return
                partial = " ".join(words[:n])
                pending_text["v"] += 1
                await ws.send(json.dumps({"type": "text", "text": partial, "version": pending_text["v"]}))
                await asyncio.sleep(a.word_delay)  # el mínimo del servidor es 300 ms

        tasks = [asyncio.create_task(reader()), asyncio.create_task(frames()), asyncio.create_task(typist())]
        await tasks[1]
        await asyncio.sleep(1)
        await ws.send(json.dumps({"type": "close"}))
        try:
            await asyncio.wait_for(tasks[0], timeout=5)
        except Exception:
            pass
        for t in tasks:
            t.cancel()
    cap.release()

    fused = [m for m in updates if m.get("fusion")]
    infs = [m["inferenceMs"] for m in updates]
    stats = {"updates": len(updates), "fusions": len(fused), "incongruentFusions": sum(1 for m in fused if m["fusion"]["incongruence"]),
             "dropped": drops, "inferenceMsMean": round(statistics.mean(infs), 1) if infs else None,
             "inferenceMsMax": max(infs) if infs else None}
    print("Resumen:", stats)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"session": {k: s[k] for k in ("sessionId", "mode", "sampleFps", "visualWeight", "visualWindowSeconds",
                                                "incongruenceThreshold", "modelVersions")},
                   "stats": stats, "updates": [{k: v for k, v in m.items() if k != "text"} | {"text": m["text"] and
                                    {k: v for k, v in m["text"].items()}} for m in updates]},
                  f, ensure_ascii=False, indent=2)
    print(f"Guardado en {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
