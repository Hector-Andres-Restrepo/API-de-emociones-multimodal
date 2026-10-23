# API de reconocimiento de emociones multimodal (texto · imagen · video · cámara · multimodal en vivo)

Actividad académica individual · contrato v1.0.0 · **uso académico, no comercial**.

> **Alcance y limitaciones.** El sistema estima *categorías de expresiones faciales* y *clasificaciones de texto*.
> **No** determina de forma fiable la emoción interna, la sinceridad ni el estado psicológico de una persona.
> Una sonrisa no prueba felicidad y la **incongruencia no indica engaño ni ocultamiento**. No usar para decisiones
> laborales, disciplinarias, médicas o de alto impacto. Pesos de fusión, umbral y valencias son una **propuesta
> experimental** del informe, no una regla validada.

## 1. Estado de verificación (léalo primero)

| Elemento | Estado |
|---|---|
| Contratos REST/WebSocket, validaciones, errores, fusión, incongruencia, jobs, límites | Implementado y cubierto por **127 pruebas automatizadas** (`docs/test-results.txt`; ejecútelas de nuevo con la guía de demostración). Las pruebas usan motores **simulados, declarados en `tests/fakes.py` y usados solo allí**. |
| Inferencia real (RoBERTuito y Emo-AffectNet) con Docker | **Verificado en el equipo del autor (19-oct-2026):** `docker compose up --build` levanta la API, `/health/ready` responde 200 con ambos modelos y `/emotions/image` devolvió `HAPPINESS` (96,3 %) para una foto sonriente en 1275 ms (CPU). **Pendiente de registrar:** evidencias JSON de texto, video, multimodal y cámara (Fase 3 de la guía), con sus latencias medidas. Los JSON de ejemplo del enunciado **no** son evidencia. |
| Compatibilidad de pesos Emo-AffectNet | LSTM verificado como compatible; backbone corregido tras comparar con el volcado real de claves (322/322 nombres coinciden). Confirme con `scripts/inspect_weights.py` (sección 6) en su equipo; el resultado esperado es `COMPATIBLE` en ambos. |

La aplicación **nunca** devuelve respuestas estáticas: si un modelo no carga, responde `503 MODEL_UNAVAILABLE` y
`/health/ready` devuelve `503`.

## 2. Arquitectura

```
Cliente ── texto ─────────> POST /emotions/text ───────> RoBERTuito ───────────┐
        ── imagen ────────> POST /emotions/image ──────> Emo-AffectNet (frame) ┤
        ── video ─────────> POST /emotions/video ──────> cola → muestreo → ResNet50 por frame
        │                                                 → ventana de W frames → LSTM ┤
        ── fotogramas ────> WS /emotions/streams/{id} ─> ventana por sesión → LSTM ────┤
        ── cámara + texto ─> WS /emotions/live/{id} ────> LSTM (ventana reciente) + RoBERTuito
           mientras escribe                                 → fusión continua + incongruencia ┘
                                                                                      v
                                              alineación a 7 clases canónicas (por nombre, no por índice)
                                                                                      v
                                       fusión tardía (texto + 1 rostro) + incongruencia → JSON + advertencias
```

```
app/
  main.py            fábrica FastAPI, health, ciclo de vida, carga de modelos en hilo
  config.py          variables de entorno          deps.py     auth Bearer, límite de tasa, concurrencia
  labels.py          clases, valencias, adaptadores de etiquetas
  fusion.py          fusión tardía e incongruencia errors.py   contrato de errores
  engines/           text_engine, visual_engine, emo_affectnet_arch (ResNet50 + LSTM)
  services/          media (validación/archivos), video (muestreo, LSTM, jobs), streams (sesiones WS),
                     live (LiveAnalyzer: lógica de la sesión multimodal en vivo, sin transporte)
  api/               emotions.py (REST), streams.py y live.py (REST + WebSocket)
  static/demo.html   página de demostración (sirve en /demo; sin secretos)
tests/               pytest (mocks solo aquí)      scripts/   evidencias, cámara, multimodal en vivo, pesos
openspec/            especificaciones y cambios (flujo spec-driven con OpenSpec)
docs/                openapi.json, asyncapi.yaml, informe-tecnico.md, test-results.txt
```

## 3. Puesta en marcha

Requisitos: Docker con Compose, ~4 GB de RAM libres y conexión a internet en el **primer** arranque.

```bash
cp .env.example .env        # edite API_TOKEN (cadena larga y aleatoria; nunca la suba al repositorio)
docker compose up --build
curl localhost:8000/health/live
curl localhost:8000/health/ready   # 503 mientras descarga/carga; 200 cuando ambos modelos están listos
```

- **Descarga inicial:** ~100 MB (backbone) + ~12 MB (LSTM) + RoBERTuito (~500 MB) hacia el volumen `models_data`
  (`/models`, caché de Hugging Face). Persiste entre reinicios; el proceso corre como usuario `appuser` (uid 1000)
  y el volumen se inicializa con esos permisos. Para trabajo sin internet, precargue el volumen y use `HF_HUB_OFFLINE=1`.
- **Temporales:** los videos se escriben en `/tmp/emotions` (tmpfs de 512 MB) y se borran al terminar cada job
  (éxito, fallo o rechazo); al arrancar se limpia cualquier resto. Imágenes y texto no se escriben en disco.
- **GPU:** no requerida. En CPU el rendimiento depende del equipo; mida y registre sus tiempos en `evidence/`.
- Si `mediapipe` instala `opencv-contrib-python` junto a `opencv-python-headless` y aparece un conflicto de `cv2`,
  fije una sola variante de OpenCV en `requirements.txt`.

### Variables de entorno (ver `.env.example`)

| Grupo | Variables |
|---|---|
| Acceso | `API_TOKEN` (obligatoria), `PORT` |
| Modelos | `MODELS_DIR`, `TEXT_MODEL_ID`, `VISUAL_REPO_ID`, `VISUAL_BACKBONE_FILE`, `VISUAL_LSTM_FILE`, `LSTM_WINDOW`, `ALLOW_START_WITHOUT_MODELS` |
| Fusión | `DEFAULT_VISUAL_WEIGHT` (0.5), `INCONGRUENCE_THRESHOLD` (0.70) |
| Límites | `MAX_IMAGE_MB` 5, `MAX_VIDEO_MB` 50, `MAX_VIDEO_SECONDS` 60, `MAX_TEXT_CHARS` 5000, `MAX_FRAME_BYTES`, `MAX_FACES`, `MAX_CONCURRENT_INFERENCES` 2, `MAX_VIDEO_WORKERS` 1, `MAX_VIDEO_JOBS_QUEUE` 10, `RATE_LIMIT_PER_MINUTE` 60, `STREAM_MAX_SECONDS` 300, `STREAM_TOKEN_TTL_SECONDS` 60, `MAX_STREAM_SESSIONS` 5 (compartido con /live), `JOB_RETENTION_SECONDS` 3600 |
| Multimodal en vivo | `LIVE_VISUAL_WINDOW_SECONDS` 5 (ventana visual por defecto), `LIVE_TEXT_MIN_INTERVAL_MS` 300 (intervalo mínimo entre textos) |
| Recursos | `MEM_LIMIT`, `CPU_LIMIT` (compose) |

Sustituir el modelo visual (por ejemplo, si los pesos no cargan): cambie `VISUAL_REPO_ID`/archivos y adapte
`VISUAL_OUTPUT_ORDER` en `app/labels.py`; el adaptador exige que las etiquetas cubran exactamente las 7 clases.

## 4. Endpoints

Base `http://localhost:8000/api/v1`. Todos exigen `Authorization: Bearer $API_TOKEN`. Documentación interactiva:
`/docs`; contrato: `docs/openapi.json` (REST) y `docs/asyncapi.yaml` (WebSocket); solicitudes: `requests.http`.
Cada respuesta exitosa incluye `requestId` (UUID) y `modelVersions`; también `inferenceMs` en análisis síncronos.

```bash
T="Authorization: Bearer $API_TOKEN"; B=http://localhost:8000/api/v1

# Texto
curl -s $B/emotions/text -H "$T" -H 'Content-Type: application/json' \
  -d '{"text":"Hoy me siento muy feliz","language":"es"}'

# Imagen (JPEG/PNG ≤ 5 MB): varios rostros se reportan por separado; sin rostro → facesDetected 0 + NO_FACE_DETECTED
curl -s $B/emotions/image -H "$T" -F image=@rostro.jpg

# Video (MP4 ≤ 50 MB y ≤ 60 s): 202 + analysisId; luego consulta
curl -s $B/emotions/video -H "$T" -F video=@clip.mp4 -F sampleFps=2
curl -s $B/emotions/video/vid-xxxxxxxx -H "$T"

# Multimodal: texto + exactamente UNA fuente visual (image → 200; video → 202 + GET /emotions/analyze/{id})
curl -s $B/emotions/analyze -H "$T" -F text="Estoy muy triste" -F image=@rostro.jpg -F visualWeight=0.5
```

| Endpoint | Notas de comportamiento |
|---|---|
| `POST /emotions/text` | `text` 1–5000 caracteres; `language` solo `es` (otro → `422 UNSUPPORTED_LANGUAGE`). Avisa si el texto excede 128 tokens (se trunca). |
| `POST /emotions/image` | Etiquetas en inglés y mayúsculas; 7 probabilidades normalizadas por rostro (modelo estático ResNet50). |
| `POST /emotions/video` | Estados `PENDING→PROCESSING→COMPLETED/FAILED`. Usa **ResNet50 por frame + LSTM** sobre una ventana deslizante de `LSTM_WINDOW` (10) vectores de 512 características; la línea de tiempo agrega las **salidas del LSTM** por tramos de `TIMELINE_SEGMENT_SECONDS`. No se promedian imágenes estáticas. Se analiza el rostro de mayor área. Estados no terminados devuelven `dominantExpression` y `timeline` en `null`; `FAILED` incluye `error`. |
| `POST /emotions/analyze` | Fusión exige **un solo rostro**: 0 o más de uno → `422 FACE_COUNT_NOT_SUPPORTED_FOR_FUSION`. Con video, esta comprobación síncrona usa los primeros ~2 s; si luego aparecen varios rostros, el job termina en `FAILED` con ese mismo código. Se conservan por separado `text` y `visual`. |
| `POST /emotions/streams` | `201` con `sessionId`, `websocketPath`, `streamToken` efímero, `windowFrames`, límites. |
| `WS /emotions/streams/{sessionId}` | Ver abajo. |

**Errores:** `{"requestId", "error": {"code","message","details"}}` con los códigos de la tabla del enunciado
(400, 401, 404, 413, 415, 422, 429, 500, 503). Fallos posteriores a aceptar un job aparecen como `status: FAILED`.

### Fusión e incongruencia

`P_fusion(c) = visualWeight·P_visual(c) + (1 − visualWeight)·P_text(c)`; emoción global = argmax; confianza = su valor.
`incongruence = true` solo si las clases predominantes tienen valencias `POSITIVE` vs `NEGATIVE` **y** ambas
confianzas ≥ `INCONGRUENCE_THRESHOLD`. `NEUTRAL` y `SURPRISE` nunca activan la oposición. Sin rostro no hay fusión.

### Cámara en vivo (WebSocket)

1. `POST /emotions/streams {"sampleFps": 2}` → `sessionId`, `websocketPath`, `streamToken` (un solo uso, 60 s).
2. Conectar `ws://localhost:8000/api/v1/emotions/streams/{sessionId}` y enviar **primero** `{"type":"auth","token":"<streamToken>"}` (el token no viaja en la URL).
3. Enviar fotogramas binarios JPEG/PNG a 1–5 fps (≤ `MAX_FRAME_BYTES`). Se responde un JSON por fotograma con rostros, `mode` (`LSTM` para `face-1`), `temporal.windowFilled`, `inferenceMs`.
4. Cierre ordenado con `{"type":"close"}`; la sesión caduca a los `STREAM_MAX_SECONDS`. Cuadros más rápidos que `sampleFps` se descartan con aviso.

**Ventana y latencia:** ventana de 10 fotogramas por sesión (a 2 fps ≈ 5 s de contexto; se rellena repitiendo el primer
vector, como el pipeline oficial). Latencia = detección de rostro + ResNet50 + LSTM por fotograma: **mídala en su
equipo** y regístrela (cada mensaje trae `inferenceMs`). Demo: abra `http://localhost:8000/demo` o ejecute `python scripts/camera_demo.py --token $API_TOKEN`
(la cámara la gestiona el cliente; el contenedor no la necesita).

### Multimodal en vivo: cámara + texto mientras se escribe

Analiza la expresión facial **mientras el usuario escribe** y aplica el procedimiento establecido (fusión tardía de
probabilidades + incongruencia) de forma continua. Una sola conexión recibe fotogramas y texto.

1. `POST /emotions/live {"sampleFps":2,"visualWeight":0.5,"visualWindowSeconds":5}` → `sessionId`, `websocketPath`, `streamToken`
   (un solo uso; **no sirve** para `/streams` ni al revés). Comparte el límite `MAX_STREAM_SESSIONS`.
2. Conectar `ws://localhost:8000/api/v1/emotions/live/{sessionId}` y enviar **primero** `{"type":"auth","token":"<streamToken>"}`. Respuesta: `ready`.
3. Enviar fotogramas binarios (≤ `sampleFps`) y, cada vez que el usuario hace una pausa al teclear (~500 ms),
   `{"type":"text","text":"...","version":n}`. Opcional: `{"type":"config","visualWeight":0.7}` para cambiar el peso en vivo.
4. Tras **cada** evento el servidor devuelve un `update` completo: `text`, `visual` (media de las salidas del LSTM en los últimos
   `visualWindowSeconds`, nunca imágenes sueltas), `fusion` (`P = w·P_visual + (1−w)·P_text`, `incongruence`), `fusionStatus`
   (`OK`, `WAITING_FOR_TEXT`, `WAITING_FOR_FACE`, `FACE_COUNT_NOT_SUPPORTED_FOR_FUSION`) e `inferenceMs`.
   Con 0 o varios rostros **no se inventa fusión**: `fusion` es `null` y se informa el estado.
5. Un texto enviado antes de `LIVE_TEXT_MIN_INTERVAL_MS` se responde con `dropped` + `retryAfterMs` (reenvíe el último valor). Cierre con `{"type":"close"}` → `closing` con resumen de contadores.

Reglas: mismas siete clases, valencias y umbral que el resto de la API. Un texto a medio escribir puede clasificarse distinto del texto
final (se avisa en `warnings`); la incongruencia es solo oposición de valencias, **no indica engaño**. El texto no se guarda ni se registra
(solo probabilidades, longitud y un resumen con clave aleatoria por sesión para no reclasificar el mismo texto).
Demo: tarjeta «6 · Multimodal en vivo» de `/demo`, o `python scripts/live_multimodal_demo.py --token $API_TOKEN --typing "Estoy muy triste"`
(evidencia en `evidence/07_live_multimodal.json`). Contrato: `docs/asyncapi.yaml`.

## 5. Pruebas

```bash
pip install -r requirements.txt   # en un entorno virtual local (las pruebas no se copian a la imagen)
pytest -q
```

Cubren: validación de esquemas, mapeo por nombre de las 7 clases y de los órdenes reales de cada modelo, normalización,
fórmula de fusión, incongruencia (positivo/negativo, umbral, neutral/sorpresa), ausencia/múltiples rostros, formatos
inválidos, transiciones de jobs, limpieza de temporales, auth/límite de tasa, WebSocket (frame, desconexión, token
de un solo uso, tamaño, timeout), multimodal en vivo (estados de fusión, ventana temporal con reloj simulado, incongruencia,
cambio de peso, cadencia, tokens no intercambiables, privacidad en logs) y salud. **No** cubren la calidad del modelo real.

## 5b. Página de demostración

`http://localhost:8000/demo` (HTML estático, sin secretos) permite probar texto, imagen, video, multimodal, cámara en vivo y multimodal en vivo (cámara + texto) desde el navegador:
pegue su `API_TOKEN` (solo se conserva en memoria de la pestaña) y use cada tarjeta. La cámara la captura el navegador y envía fotogramas por el
WebSocket documentado; se muestran emoción, modo (`LSTM`), ventana temporal y latencias. Las llamadas siguen exigiendo Bearer: la página no se salta la seguridad.

## 5c. Desarrollo con OpenSpec (spec-driven)

El proyecto se trabaja con [OpenSpec](https://github.com/Fission-AI/OpenSpec): el comportamiento se describe primero en especificaciones
y cada cambio pasa por propuesta → deltas de especificación → diseño → tareas → implementación → archivo.

```
openspec/specs/                 capacidades vigentes (10 + live-multimodal-analysis tras archivar el cambio)
openspec/changes/archive/       cambios ya aplicados (baseline del contrato v1.0.0 y la función multimodal en vivo)
openspec/config.yaml            contexto del proyecto (artefactos en español; encabezados y SHALL/MUST en inglés)
```

```bash
npm install -g @fission-ai/openspec        # requiere Node.js 20.19+
openspec list --specs                       # capacidades documentadas
openspec validate --all --strict            # valida especificaciones y cambios
openspec new change <nombre>                # nuevo cambio (con el asistente: /opsx:propose)
openspec archive <nombre> --yes             # aplica los deltas a openspec/specs y archiva
```

El cambio `add-live-multimodal-session` (propuesta, diseño con decisiones D1–D8, 13 requisitos con escenarios y tareas) documenta esta función.

## 6. Evidencias reales (a completar en su equipo)

```bash
docker compose up --build -d
docker compose run --rm api python scripts/inspect_weights.py     # debe imprimir COMPATIBLE para backbone y LSTM
python scripts/collect_evidence.py --token $API_TOKEN --text "Hoy me siento muy feliz" --image samples/rostro.jpg --video samples/clip.mp4
python scripts/camera_demo.py --token $API_TOKEN --seconds 20      # evidence/06_camera_stream.json
python scripts/live_multimodal_demo.py --token $API_TOKEN --typing "Estoy muy triste por la noticia"   # evidence/07_live_multimodal.json (con su cara, con consentimiento)
docker compose ps                                                  # healthcheck "healthy"
```

Si `inspect_weights.py` reporta claves distintas, ajuste `KEY_REMAP` en `app/engines/visual_engine.py`
(no use `strict=False`). Use únicamente imágenes/videos propios o con consentimiento; `samples/` está en `.gitignore`.

## 7. Privacidad, licencias y uso responsable

- Obtenga consentimiento antes de procesar imágenes o video de personas (Ley 1581 de 2012, Colombia).
- No se persisten imágenes, video ni texto; los logs registran solo tipos de excepción y versiones, nunca contenido.
  El texto de `/analyze` con video solo se conserva como probabilidades mientras dura el job (≤ `JOB_RETENTION_SECONDS`).
- **Licencias:** RoBERTuito (`pysentimiento/robertuito-emotion-analysis`) y el código de Emo-AffectNet aparecen bajo MIT en sus repositorios, pero el repositorio de Emo-AffectNet indica que backbone y LSTM se ofrecen *"for scientific usage only"*. Esta actividad es académica y **no autoriza uso comercial**; cite el artículo (Ryumina et al., *Neurocomputing*, 2022) y los de RoBERTuito/pysentimiento.
- Sesgos: AffectNet/Aff-Wild2 y TASS 2020 (tuits) tienen sesgos demográficos y de dominio; las precisiones reportadas (≈66 % AffectNet, macro-F1 ≈0.56 en texto) indican error sustancial.

Más detalle en `docs/informe-tecnico.md`.
