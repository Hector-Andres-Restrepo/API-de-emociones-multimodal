## 1. Configuración y sesiones

- [x] 1.1 Añadir `LIVE_VISUAL_WINDOW_SECONDS` y `LIVE_TEXT_MIN_INTERVAL_MS` a `app/config.py` y `.env.example`
- [x] 1.2 Añadir modo y parámetros a las sesiones de `app/services/streams.py`; la autenticación exige el modo correcto y el límite de sesiones es compartido

## 2. Lógica de la sesión en vivo

- [x] 2.1 Crear `app/services/live.py` con `LiveAnalyzer` (reloj inyectable): ventana visual reciente, estado del texto, fusión y estados de fusión
- [x] 2.2 Control de cadencia de fotogramas y de texto, con `retryAfterMs`
- [x] 2.3 Validación de texto, peso visual y privacidad (sin retener el texto)

## 3. API

- [x] 3.1 `POST /api/v1/emotions/live` con validaciones y 503 si falta algún modelo
- [x] 3.2 `WS /api/v1/emotions/live/{sessionId}` con autenticación, mensajes `text`, `config`, `close`, fotogramas, límites y cierre ordenado
- [x] 3.3 Registrar los routers en `app/main.py`

## 4. Pruebas

- [x] 4.1 Motor de texto simulado por palabras clave en `tests/fakes.py` (solo pruebas)
- [x] 4.2 Pruebas unitarias del analizador con reloj simulado
- [x] 4.3 Pruebas del WebSocket: contrato, autenticación cruzada, flujo texto más fotogramas, límites, cierre y privacidad en logs

## 5. Demostración

- [x] 5.1 Tarjeta «Multimodal en vivo» en `/demo` con cámara y cuadro de texto a la vez
- [x] 5.2 Cliente `scripts/live_multimodal_demo.py` con escritura simulada para generar evidencias

## 6. Documentación

- [x] 6.1 Regenerar `docs/openapi.json` y ampliar `docs/asyncapi.yaml`
- [x] 6.2 Actualizar README, informe técnico, `requests.http` y la guía de demostración

## 7. Verificación

- [x] 7.1 Suite completa en verde y prueba de punta a punta en navegador real con cámara simulada
- [x] 7.2 `openspec validate --strict` sin errores y archivar el cambio
