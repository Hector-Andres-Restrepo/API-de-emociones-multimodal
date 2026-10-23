## 1. Núcleo de etiquetas y fusión

- [x] 1.1 Definir las siete clases canónicas, valencias y adaptadores por nombre (`app/labels.py`)
- [x] 1.2 Implementar fusión tardía e incongruencia parametrizables (`app/fusion.py`)
- [x] 1.3 Verificar al cargar que las etiquetas reales cubren las siete clases

## 2. Motores de inferencia

- [x] 2.1 Motor de texto con RoBERTuito y aviso de truncado
- [x] 2.2 Arquitectura y carga de pesos de Emo-AffectNet verificada contra los `.pt` (322 parámetros del backbone, 10 del LSTM)
- [x] 2.3 Detección de rostros y preprocesado del pipeline oficial

## 3. API REST y WebSocket

- [x] 3.1 Endpoints de texto, imagen, video, análisis combinado y consulta de jobs
- [x] 3.2 Sesión y WebSocket de cámara con token efímero y ventana temporal
- [x] 3.3 Autenticación Bearer, límites de tasa y concurrencia, contrato común de errores

## 4. Despliegue y privacidad

- [x] 4.1 Dockerfile, compose, `.env.example`, volumen de modelos y tmpfs de temporales
- [x] 4.2 Endpoints `/health/live` y `/health/ready`
- [x] 4.3 Borrado de temporales y logs sin contenido
- [x] 4.4 Página `/demo` sin secretos

## 5. Verificación y documentación

- [x] 5.1 Suite de 65 pruebas automatizadas (motores simulados solo en `tests/fakes.py`)
- [x] 5.2 README, informe técnico, OpenAPI, AsyncAPI y colección `.http`
- [x] 5.3 Verificación real en Docker: `/health/ready` 200 y análisis de imagen con inferencia real
