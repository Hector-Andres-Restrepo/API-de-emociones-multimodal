# Proposal

## Why

Hoy el análisis multimodal (`/emotions/analyze`) solo funciona con **archivos ya capturados**: un texto más una imagen o un video. Se pidió una segunda modalidad combinada en **tiempo real**: que el sistema analice la expresión facial por cámara **mientras la persona escribe** un texto, y aplique el procedimiento ya establecido (fusión tardía de probabilidades e indicador de incongruencia) de forma continua, sin esperar a que termine de escribir.

## What Changes

- Nuevo recurso `POST /api/v1/emotions/live` que crea una **sesión multimodal en vivo** y devuelve un token efímero, igual que las sesiones de cámara existentes.
- Nuevo WebSocket `/api/v1/emotions/live/{sessionId}` por el que el cliente envía **fotogramas binarios** (cámara) y **mensajes JSON de texto** (lo que va escribiendo), y recibe **actualizaciones completas** con el estado facial, el estado textual, la fusión y la incongruencia.
- El componente visual es el promedio de las salidas del modelo temporal LSTM en una **ventana reciente** (5 s por defecto, configurable), para que la fusión refleje la expresión durante la escritura y no un único fotograma.
- Se reutilizan sin cambios las reglas de fusión e incongruencia de `multimodal-fusion` y las normas de autenticación, límites y privacidad vigentes.
- Se amplía la página `/demo` con una tarjeta de **multimodal en vivo** (cuadro de texto y cámara a la vez) y se añade un cliente de línea de comandos para generar evidencias.
- Nuevas variables de entorno: `LIVE_VISUAL_WINDOW_SECONDS` y `LIVE_TEXT_MIN_INTERVAL_MS`.
- Sin cambios **BREAKING**: los endpoints y mensajes existentes no cambian.

## Capabilities

### New Capabilities
- `live-multimodal-analysis`: sesión en vivo que fusiona continuamente la expresión facial de la cámara con el texto que se está escribiendo.

### Modified Capabilities
- `camera-stream`: el límite de sesiones simultáneas pasa a contar juntas las sesiones de cámara y las multimodales en vivo.
- `demo-page`: la cobertura de modalidades incluye el análisis multimodal en vivo.

## Impact

- **Código**: nuevo `app/services/live.py` (lógica de sesión sin transporte), `app/api/live.py` (REST y WebSocket), cambios menores en `app/services/streams.py` (modo de sesión), `app/config.py`, `app/main.py` y `app/static/demo.html`.
- **API**: una ruta REST y un canal WebSocket nuevos; se actualizan `docs/openapi.json` y `docs/asyncapi.yaml`.
- **Pruebas**: nuevas pruebas unitarias del analizador (con reloj simulado) y del WebSocket; los motores simulados siguen viviendo solo en `tests/fakes.py`.
- **Recursos**: cada evento de fotograma o de texto pasa por la misma compuerta de concurrencia de inferencia; el texto largo o muy frecuente se limita.
- **Fuera de alcance**: audio o voz, varias personas por sesión, historial persistente, detección de engaño y cualquier interpretación del estado interno de la persona.
