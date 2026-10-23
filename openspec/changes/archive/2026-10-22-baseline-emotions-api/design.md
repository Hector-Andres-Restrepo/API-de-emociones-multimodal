## Context

Este diseño resume decisiones **ya tomadas** durante la implementación del contrato v1.0.0. El detalle técnico completo, con fuentes verificadas, está en `docs/informe-tecnico.md`; aquí se recogen las decisiones que sostienen las especificaciones.

## Goals / Non-Goals

**Goals:**
- Inferencia real de texto, imagen, video y cámara con dos modelos preentrenados.
- Contrato HTTP/WebSocket estable, testeable y documentado (OpenAPI y AsyncAPI).
- Despliegue reproducible con Docker Compose en CPU.

**Non-Goals:**
- Evaluar o mejorar la exactitud de los modelos.
- Persistir datos, cuentas de usuario o historial.
- Escalar horizontalmente (el estado de jobs y sesiones vive en memoria de un solo proceso).

## Decisions

- **Modelos**: RoBERTuito (`pysentimiento/robertuito-emotion-analysis`) para texto en español y Emo-AffectNet (`ElenaRyumina/face_emotion_recognition`, ResNet50 más LSTM) para visión. Alternativa descartada: un único modelo multimodal, porque el enunciado exige clasificación independiente por modalidad y fusión tardía.
- **Etiquetas por nombre**: cada modelo se adapta a las siete clases canónicas por nombre de etiqueta leída del modelo real (`others` pasa a NEUTRAL). Una prueba demuestra que un mapeo por índice daría resultados incorrectos.
- **Video temporal**: ResNet50 por fotograma y un LSTM sobre una ventana deslizante de 10 vectores de 512 características, con el mismo preprocesado del pipeline oficial. La línea de tiempo promedia salidas del LSTM por tramos, nunca imágenes estáticas.
- **Fusión**: media ponderada de probabilidades; incongruencia por valencias opuestas con confianza mínima por modalidad. Todo parametrizable y declarado como propuesta experimental.
- **Cámara**: la captura la hace el cliente; la sesión se crea por REST y el WebSocket se autentica con un token efímero de un solo uso enviado como primer mensaje, nunca en la URL.
- **Jobs en memoria**: un worker de video y una cola acotada; adecuado para el alcance académico.
- **Arranque sin modelos**: la API arranca y reporta `ready` 503 si los pesos no cargan, en lugar de simular respuestas.

## Risks / Trade-offs

- [Exactitud limitada de los modelos (≈66 % en AffectNet, macro-F1 ≈0.56 en texto)] → advertencias en cada respuesta y documentación de limitaciones.
- [Estado en memoria] → se pierde al reiniciar; documentado como limitación.
- [Pesos de Emo-AffectNet solo para uso científico] → uso académico no comercial, documentado en README e informe.
- [Latencia en CPU] → límites de concurrencia y de `sampleFps`; medición real registrada en las evidencias.
