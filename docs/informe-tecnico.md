# Informe breve de decisiones técnicas

## 1. Mapeo de etiquetas (verificado contra las fuentes)

**Texto.** `config.json` de `pysentimiento/robertuito-emotion-analysis`: `id2label = {0: others, 1: joy, 2: sadness, 3: anger, 4: surprise, 5: disgust, 6: fear}`. La ficha del modelo habla de "seis emociones de Ekman más una clase neutra" pero el modelo expone `others`, no `neutral`.
Decisión: `joy→HAPPINESS`, `others→NEUTRAL`, resto por nombre. *Supuesto:* `others` es la clase residual del corpus (sin emoción de Ekman) y se trata como neutral; no equivale necesariamente a "neutralidad emocional". El motor lee `id2label` real al cargar y aborta si las etiquetas no cubren exactamente las 7 canónicas.

**Visión.** El `run_webcam.ipynb` oficial define `DICT_EMO = {0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger}`. El adaptador reordena por nombre; una prueba demuestra que un mapeo índice-a-índice daría un resultado equivocado (índice 6 = `Anger`, pero `CANONICAL[6]` = `SURPRISE`).

## 2. Modelo visual y video temporal

Backbone ResNet50 (VGGFace2 → AffectNet, 7 clases) + `fc1` 2048→512; LSTM de dos capas (512→512→256) + lineal →7, que clasifica con el último paso de una ventana de 10 vectores ReLU(fc1). Preprocesado oficial: recorte facial (MediaPipe FaceMesh, caja de landmarks), 224×224 vecino más cercano, RGB→BGR y resta de media (91.4953, 103.8827, 131.0912) sin dividir.
Video y cámara reutilizan exactamente ese flujo: ventana deslizante que solo avanza cuando hay rostro, relleno inicial con el primer vector, y salida del LSTM como predicción. La línea de tiempo promedia **salidas del LSTM** por tramos (nunca imágenes estáticas). Con varios rostros solo el de mayor área tiene seguimiento temporal (el modelo no incluye asociación de identidades entre fotogramas); en imagen única todos se reportan.

**Verificación de pesos:** `scripts/inspect_weights.py` confirmó que el LSTM es compatible. Para el backbone, la primera arquitectura usaba nombres distintos a los del `.pt`; se corrigió con el volcado real de `scripts/dump_keys.py` (`conv_layer_s2_same`, `batch_norm1-3`, `i_downsample`, `fc1`, `fc2`; 322 parámetros). Los nombres y formas no pueden confirmar tres detalles heredados del ResNet50 de Keras con el que se entrenó el modelo: stride en la primera conv 1x1 de cada bloque, `eps=1e-3` en BatchNorm y max-pool sin relleno. Se documentan en `emo_affectnet_arch.py`; una comprobación práctica es que una foto claramente sonriente produzca `HAPPINESS` con alta probabilidad. Si los pesos resultaran inutilizables, la sustitución es configurable (README §3) y debe documentarse como limitación; nunca presentar simulaciones como inferencia.

## 3. Fusión

Fórmula y umbral tal como en el enunciado (`visualWeight=0.5`, umbral 0.70), todos configurables por entorno/solicitud. La "confianza" de cada modalidad para la incongruencia es su probabilidad máxima. Las probabilidades por modelo no están calibradas entre sí; promediarlas con pesos iguales es una heurística, no una regla validada. Para video, el componente visual es la media de las salidas del LSTM sobre los fotogramas con rostro.

### 3b. Multimodal en vivo (cámara + texto mientras se escribe)

Recurso propio `/emotions/live` (token no intercambiable con `/streams`; límite de sesiones compartido). Un único WebSocket recibe fotogramas
binarios y texto JSON. **Componente visual** = media de las salidas del LSTM de los últimos `visualWindowSeconds` (5 s por defecto): un solo
fotograma sería ruidoso y promediar salidas (no imágenes) refleja cómo se veía la cara mientras se escribía. **Texto**: el cliente espera ~500 ms tras la
última tecla y el servidor impone ≥ 300 ms entre textos (`dropped` + `retryAfterMs`). Cada evento devuelve una instantánea completa con la misma
fórmula, valencias y umbral 0.70. Con 0 o varios rostros no hay fusión (`fusionStatus`), en lugar de un error o un resultado inventado.
Limitaciones: un texto parcial se clasifica distinto del final; la incongruencia puede parpadear al cambiar texto o expresión (no se añadió histéresis para
no ocultar el comportamiento real del modelo); texto y fotogramas comparten la compuerta de concurrencia, por lo que en CPU se recomienda 1–2 fps.
Privacidad: no se guarda el texto (solo probabilidades, longitud y un resumen con clave aleatoria por sesión). Decisiones completas en
`openspec/changes/.../design.md`.

## 4. Latencia y recursos

**Medición preliminar (equipo del autor, CPU, Docker Compose):** una imagen con un rostro se analizó en 1275 ms de inferencia en el servidor (primera llamada tras el arranque; puede incluir calentamiento). Las latencias de video, multimodal y cámara se registran en `evidence/` con `scripts/collect_evidence.py` y `scripts/camera_demo.py` (que reporta media y máximo de inferencia y de ida y vuelta). Diseño: `MAX_CONCURRENT_INFERENCES=2` para REST/WS, un worker de video (`MAX_VIDEO_WORKERS=1`), cola de 10 jobs, tope de 4 GB/2 CPU en compose. Un video de 60 s a 2 fps son 120 inferencias de ResNet50 más 120 del LSTM, y la cámara a 5 fps puede saturar CPU modestas: reduzca `sampleFps` si la latencia por mensaje supera el intervalo.

## 5. Privacidad y seguridad

Sin persistencia de contenido; temporales en tmpfs con borrado garantizado (`finally`) y barrido al arrancar; logs sin contenido; Bearer comparado en tiempo constante; WebSocket con token efímero de un solo uso enviado por mensaje (no en URL); límite de tasa en memoria por IP (por proceso: con varias réplicas habría que moverlo a un almacén compartido); `torch.load(weights_only=True)` para evitar ejecución de código al deserializar pesos.

## 6. Limitaciones conocidas

Estado en memoria de jobs y sesiones (se pierde al reiniciar; una sola réplica). Solo español en texto. Verificación MP4 por cabecera `ftyp`. Sin pruebas de calidad del modelo real ni evaluación de equidad entre grupos demográficos. `others→NEUTRAL` es un supuesto.

## 7. Licencias

Ver README §7. Uso exclusivamente académico.
