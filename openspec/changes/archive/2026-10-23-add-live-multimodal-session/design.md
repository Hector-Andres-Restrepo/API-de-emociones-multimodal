## Context

El contrato v1.0.0 ya tiene dos piezas que esta función combina: la **sesión de cámara** (`/emotions/streams`, WebSocket, ventana temporal LSTM por sesión) y la **fusión tardía con incongruencia** (`/emotions/analyze`, solo con archivos). La fusión en vivo debe respetar las mismas reglas (siete clases, valencias, fórmula, umbral 0.70, un solo rostro) y las mismas normas de seguridad, límites y privacidad.

La diferencia clave es que las dos señales **cambian con el tiempo y llegan por separado**: los fotogramas a 1–5 por segundo y el texto cada vez que el usuario deja de teclear.

## Goals / Non-Goals

**Goals:**
- Una sola conexión que reciba fotogramas y texto, y devuelva el estado completo tras cada evento.
- Fusión continua con las reglas existentes, sin duplicar la lógica de `app/fusion.py`.
- Lógica de sesión **separada del transporte** para probarla con reloj simulado y sin WebSocket.
- Texto nunca persistido ni registrado.

**Non-Goals:**
- Audio, voz o dictado.
- Varias personas por sesión (la fusión sigue exigiendo un solo rostro).
- Histórico, estadísticas por usuario o cualquier almacenamiento.
- Interpretar engaño: la incongruencia sigue siendo solo oposición de valencias.

## Decisions

**D1. Recurso propio `/emotions/live` en lugar de un modo dentro de `/emotions/streams`.**
Contrato claro (parámetros distintos, mensajes distintos) y token no intercambiable entre tipos de sesión. Se reutiliza el almacén de sesiones y su límite, añadiéndole un campo de modo. Alternativa descartada: `mode` en `/streams`, que mezclaría dos esquemas de mensajes en un mismo canal.

**D2. Un único WebSocket con fotogramas binarios y texto JSON.**
Una conexión mantiene el orden de los eventos y un solo estado de sesión. Alternativa descartada: dos conexiones (cámara y texto) que habría que correlacionar en el servidor.

**D3. Componente visual = media de las salidas del LSTM en una ventana temporal reciente (5 s por defecto).**
Un único fotograma sería ruidoso y el modelo ya aporta contexto temporal; promediar sus salidas (nunca imágenes estáticas) suaviza parpadeos y refleja «cómo se veía la cara mientras escribía». Es parametrizable por sesión (1–30 s). Alternativas descartadas: usar solo el último fotograma (inestable) y acumular «desde que empezó a escribir» (exige definir episodios de escritura y complica el estado).

**D4. Cadencia del texto: el cliente espera ~500 ms tras la última tecla y el servidor impone un intervalo mínimo de 300 ms.**
Evita una inferencia por pulsación. Si llega un texto demasiado pronto, se responde `dropped` con `retryAfterMs` y el cliente reenvía el último valor. Alternativa descartada: fusionar y retrasar textos en el servidor, que exigiría temporizadores y tareas en segundo plano por sesión.

**D5. Cada actualización es una instantánea completa** (texto, visual, fusión, estado), no un diferencial.
Simplifica el cliente (renderiza lo último que recibe) y las pruebas; el costo extra es de unos cientos de bytes por mensaje.

**D6. Estado de fusión explícito en lugar de errores.**
En REST, cero o varios rostros devuelven 422. En un flujo continuo eso es un estado transitorio, no un fallo: se informa `fusionStatus` (`OK`, `WAITING_FOR_TEXT`, `WAITING_FOR_FACE`, `FACE_COUNT_NOT_SUPPORTED_FOR_FUSION`) con `fusion` nulo, de modo que nunca se produce una fusión ficticia. Con más de un rostro en el último fotograma se bloquea la fusión y la ventana temporal no se actualiza.

**D7. Privacidad por construcción.**
El analizador guarda solo probabilidades, contadores y un resumen con clave aleatoria por sesión (para no re-clasificar un texto idéntico). El texto no se escribe en disco ni en logs.

**D8. Separación `LiveAnalyzer` (lógica) y manejador WebSocket (transporte).**
El analizador recibe motores y reloj inyectables, así que las pruebas cubren ventana, estados, umbrales y límites de cadencia de forma determinista. El manejador solo autentica, valida tamaños, aplica la compuerta de concurrencia y serializa.

## Risks / Trade-offs

- [Un texto a medio escribir se clasifica distinto del texto final] → advertencia en cada actualización; el cliente espera una pausa antes de enviar.
- [La incongruencia puede parpadear al cambiar la expresión o el texto] → la ventana temporal amortigua la señal visual; no se añade histéresis para no ocultar el comportamiento real del modelo (puede evaluarse más adelante).
- [Cada fotograma y cada texto compiten por la misma compuerta de inferencia] → misma limitación global de concurrencia y de sesiones; en CPU se recomienda `sampleFps` 1–2.
- [Latencia por mensaje en CPU] → `inferenceMs` en cada actualización; se mide en las evidencias.
- [Una persona distinta de la que escribe puede aparecer en cámara] → se exige un solo rostro; con más, no hay fusión.
