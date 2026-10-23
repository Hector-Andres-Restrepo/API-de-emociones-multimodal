# Proposal

## Why

La API de reconocimiento de emociones multimodal (contrato v1.0.0 de la actividad académica) ya está implementada y funcionando, pero su comportamiento solo estaba descrito en el enunciado, el README y las pruebas. Para trabajar el proyecto con OpenSpec hace falta una **línea base** de especificaciones: así los cambios futuros (como el modo multimodal en vivo) se expresan como deltas sobre contratos explícitos y verificables.

**Esta línea base es retroactiva**: documenta lo que el sistema ya hace. No se presenta como desarrollo spec-first; cada requisito fue contrastado con el código y con las pruebas existentes (65 pruebas automatizadas).

## What Changes

- Se registran como especificaciones OpenSpec las capacidades ya implementadas, sin modificar código.
- Se documentan las reglas de contrato del enunciado: siete clases canónicas, adaptadores de etiquetas, fusión tardía, incongruencia, jobs asíncronos, cámara por WebSocket, contrato de errores y requisitos Docker.
- Se fijan límites y valores por defecto (5 MB de imagen, 50 MB y 60 s de video, 5000 caracteres, umbral de incongruencia 0.70, peso visual 0.5) como requisitos comprobables.
- No hay cambios **BREAKING**: es solo documentación de comportamiento vigente.

## Capabilities

### New Capabilities
- `emotion-label-alignment`: siete clases canónicas, valencias y adaptadores explícitos de etiquetas por modelo.
- `text-emotion-analysis`: clasificación de emoción en texto en español (`POST /emotions/text`).
- `image-emotion-analysis`: expresión facial en imágenes, uno o varios rostros (`POST /emotions/image`).
- `video-emotion-analysis`: análisis asíncrono de video con modelo temporal LSTM.
- `multimodal-fusion`: texto más una fuente visual, fusión tardía e indicador de incongruencia (`/emotions/analyze`).
- `camera-stream`: cámara en vivo por WebSocket con ventana temporal por sesión.
- `api-access-and-errors`: autenticación Bearer, trazabilidad, límites de tasa y contrato común de errores.
- `container-deployment`: despliegue con Docker Compose, salud, volúmenes y configuración por entorno.
- `privacy-and-responsible-use`: no persistencia, logs sin contenido, advertencias de alcance e inferencia real.
- `demo-page`: página de demostración servida por la propia API.

### Modified Capabilities
<!-- Ninguna: es la línea base. -->

## Impact

- **Código**: ninguno (documenta lo existente).
- **Artefactos nuevos**: `openspec/specs/*` tras archivar este cambio.
- **Relación con otros documentos**: complementa `README.md`, `docs/informe-tecnico.md`, `docs/openapi.json` y `docs/asyncapi.yaml`; no los reemplaza.
- **Fuera de alcance**: calidad predictiva de los modelos (no se especifica exactitud, solo comportamiento del contrato).
