# camera-stream Specification

## Purpose
Analizar la expresión facial de una cámara en vivo, gestionada por el cliente, mediante una sesión WebSocket con ventana temporal por sesión.

## Requirements

### Requirement: Stream session creation
The system SHALL expose `POST /api/v1/emotions/streams` accepting `sampleFps` between 1 and 5 (default 2) and SHALL answer 201 with `sessionId`, `websocketPath`, an ephemeral `streamToken`, the temporal window size and the applicable limits.

#### Scenario: Valid session
- **WHEN** an authenticated client creates a session with `sampleFps` 2
- **THEN** the response is 201 and `websocketPath` does not contain the token

#### Scenario: Invalid rate
- **WHEN** `sampleFps` is 0, 6 or negative
- **THEN** the response is 400 with code VALIDATION_ERROR

### Requirement: WebSocket authentication without URL tokens
The system SHALL authenticate a WebSocket connection with a first text message `{"type":"auth","token":"<streamToken>"}` received within a short timeout, SHALL accept each token only once, and SHALL reject expired, unknown or reused tokens with an UNAUTHORIZED error and close code 1008.

#### Scenario: Token reuse
- **WHEN** a second connection presents a token already used
- **THEN** the server answers an error with code UNAUTHORIZED and closes the connection

### Requirement: Per-frame result
The system SHALL accept binary JPEG or PNG messages and SHALL answer one JSON message per processed frame with the number of faces, each face's id, emotion and confidence, the analysis mode, the temporal window state, `inferenceMs` and `modelVersions`.

#### Scenario: Frame with one face
- **WHEN** a valid frame with one face is sent
- **THEN** the answer has `facesDetected` 1 and `faces[0].faceId` equal to `face-1`

#### Scenario: Frame without a face
- **WHEN** a frame has no face
- **THEN** the answer has `facesDetected` 0, an empty `faces` list and the warning NO_FACE_DETECTED, and the session stays open

### Requirement: Temporal window per session
The system SHALL keep a window of recent frame features per session and SHALL classify the main face with the temporal LSTM model, marking its mode as LSTM and any other face as STATIC_FRAME.

#### Scenario: Window fills up
- **WHEN** consecutive frames with a face are processed
- **THEN** `temporal.windowFilled` grows up to `temporal.windowFrames` and the main face mode is LSTM

### Requirement: Frame size and format limits
The system SHALL answer PAYLOAD_TOO_LARGE and close with code 1009 when a frame exceeds the configured size, and SHALL answer INVALID_FILE_FORMAT without closing the session when a frame is not a valid image.

#### Scenario: Corrupt frame
- **WHEN** a binary message is not an image
- **THEN** the server answers INVALID_FILE_FORMAT and later valid frames are still processed

### Requirement: Respect of the configured rate
The system SHALL discard frames that arrive faster than the configured `sampleFps` (with a small tolerance) and SHALL answer a message with `dropped` true and the warning FRAME_DROPPED_RATE_LIMIT.

#### Scenario: Burst of frames
- **WHEN** two frames arrive back to back at 1 fps
- **THEN** the second is reported as dropped

### Requirement: Orderly close, timeout and disconnection
The system SHALL close a session in an orderly way when the client sends `{"type":"close"}`, SHALL end it with reason SESSION_TIMEOUT after the maximum duration, and SHALL release the session resources when the client disconnects.

#### Scenario: Client disconnects
- **WHEN** the client drops the connection
- **THEN** the session is removed from the server

### Requirement: Simultaneous session limit
The system SHALL limit the number of simultaneous sessions, counting camera stream sessions and live multimodal sessions together, to a configured maximum and SHALL answer 429 RATE_LIMIT_EXCEEDED when the limit is reached.

#### Scenario: Limit reached
- **WHEN** the maximum number of sessions is already open
- **THEN** creating another session returns 429 with code RATE_LIMIT_EXCEEDED

#### Scenario: Mixed session types
- **WHEN** the open sessions are a mix of camera stream and live multimodal sessions that together reach the maximum
- **THEN** creating a session of either type returns 429 with code RATE_LIMIT_EXCEEDED
