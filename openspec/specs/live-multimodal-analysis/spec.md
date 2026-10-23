# live-multimodal-analysis Specification

## Purpose
Analizar en tiempo real la expresión facial de una cámara gestionada por el cliente mientras la persona escribe un texto, y fusionar de forma continua ambas señales con las reglas de fusión tardía e incongruencia ya establecidas.

## Requirements

### Requirement: Live session creation
The system SHALL expose `POST /api/v1/emotions/live` accepting optional `sampleFps` (1 to 5, default 2), `visualWeight` (0 to 1, default the configured weight), `visualWindowSeconds` (1 to 30, default the configured window) and `language` (`es`), and SHALL answer 201 with `sessionId`, `websocketPath`, an ephemeral `streamToken` and the effective parameters and limits.

#### Scenario: Valid session
- **WHEN** an authenticated client creates a live session with default parameters
- **THEN** the response is 201 with the effective `sampleFps`, `visualWeight`, `visualWindowSeconds`, `incongruenceThreshold` and `modelVersions` for both models, and `websocketPath` does not contain the token

#### Scenario: Invalid parameters
- **WHEN** `sampleFps` is outside 1 to 5, `visualWeight` outside 0 to 1, or `visualWindowSeconds` outside 1 to 30
- **THEN** the response is 400 with code VALIDATION_ERROR

#### Scenario: Unsupported language
- **WHEN** `language` is not `es`
- **THEN** the response is 422 with code UNSUPPORTED_LANGUAGE

### Requirement: Both models required
The system SHALL answer 503 MODEL_UNAVAILABLE when creating a live session while the text model or the visual model is not loaded.

#### Scenario: Visual model not loaded
- **WHEN** the visual model has not finished loading
- **THEN** creating a live session returns 503 with code MODEL_UNAVAILABLE

### Requirement: Live WebSocket authentication
The system SHALL authenticate the live WebSocket with a first text message `{"type":"auth","token":"<streamToken>"}`, SHALL accept each token once, and SHALL NOT accept a token issued for a camera stream session on a live session or the reverse.

#### Scenario: Token of another session type
- **WHEN** a client presents on the live channel a token issued by `POST /emotions/streams`
- **THEN** the server answers an error with code UNAUTHORIZED and closes the connection

### Requirement: Text updates while typing
The system SHALL accept JSON messages `{"type":"text","text":"..."}` at any time during the session, SHALL classify the current text with the text model, and SHALL treat an empty or whitespace-only text as the removal of the current text.

#### Scenario: Text arrives
- **WHEN** the client sends the text "Estoy muy triste"
- **THEN** the next update contains `text.emotion`, `text.valence`, `text.confidence` and the seven `text.probabilities`

#### Scenario: Text cleared
- **WHEN** the client sends an empty text
- **THEN** the next update has `text` null and `fusionStatus` WAITING_FOR_TEXT

### Requirement: Frames feed the temporal model
The system SHALL process each binary JPEG or PNG frame by detecting faces and, only when exactly one face is present, adding its features to the session's temporal window and classifying it with the temporal LSTM model.

#### Scenario: One face
- **WHEN** a frame with exactly one face arrives
- **THEN** the update reports `visual.mode` LSTM and the temporal window state

#### Scenario: Several faces
- **WHEN** a frame with two faces arrives
- **THEN** the temporal window is not updated with that frame

### Requirement: Recent facial expression window
The system SHALL compute the visual component of each update as the mean of the temporal model outputs obtained from single-face frames within the last `visualWindowSeconds`, and SHALL report the number of frames used, the window length and the age of the latest frame.

#### Scenario: Expression changes
- **WHEN** the facial expression changes from happy to neutral over several frames
- **THEN** the visual probabilities move gradually as older frames leave the window

#### Scenario: Window expires
- **WHEN** no single-face frame arrives within the window
- **THEN** the update has no usable visual component and `fusionStatus` WAITING_FOR_FACE

### Requirement: Continuous late fusion
The system SHALL emit an update after every processed frame and every processed text, containing the current text result, the visual result and, when both exist, the fusion computed with the weighted late fusion and incongruence rules of the multimodal-fusion capability.

#### Scenario: Both signals present
- **WHEN** a text with SADNESS (confidence 0.80) is current and the visual window shows HAPPINESS (confidence 0.85)
- **THEN** the update has `fusionStatus` OK, a `fusion` object with method WEIGHTED_LATE_FUSION and `incongruence` true

#### Scenario: Text edited without new frames
- **WHEN** the text changes while the visual window still holds recent frames
- **THEN** a new update recomputes the fusion with the new text

### Requirement: Fusion status without fictitious fusion
The system SHALL report `fusionStatus` as OK, WAITING_FOR_TEXT, WAITING_FOR_FACE or FACE_COUNT_NOT_SUPPORTED_FOR_FUSION, and SHALL set `fusion` to null unless the status is OK.

#### Scenario: No text yet
- **WHEN** frames arrive before any text
- **THEN** the updates show the visual result, `fusion` is null and `fusionStatus` is WAITING_FOR_TEXT

#### Scenario: More than one face
- **WHEN** the latest frame shows more than one face
- **THEN** `fusion` is null, `fusionStatus` is FACE_COUNT_NOT_SUPPORTED_FOR_FUSION and the warnings mention the multiple faces

#### Scenario: No face in the latest frame
- **WHEN** the latest frame has no face and older single-face frames remain within the window
- **THEN** the fusion uses those frames and the warnings include NO_FACE_DETECTED

### Requirement: Adjustable visual weight
The system SHALL accept `{"type":"config","visualWeight":<0..1>}` during the session, SHALL apply it from the next fusion, and SHALL answer VALIDATION_ERROR without closing the session when the value is outside the range.

#### Scenario: Weight changed
- **WHEN** the client sends `visualWeight` 1
- **THEN** the next fusion emotion equals the visual emotion and the update reports `visualWeight` 1 and `textWeight` 0

#### Scenario: Invalid weight
- **WHEN** the client sends `visualWeight` 1.5
- **THEN** the server answers VALIDATION_ERROR and the session continues with the previous weight

### Requirement: Input limits and cadence
The system SHALL answer PAYLOAD_TOO_LARGE and close with code 1009 when a frame exceeds the configured size, SHALL answer VALIDATION_ERROR without closing when a text exceeds the maximum length or a message is malformed, and SHALL discard with a `dropped` message any frame or text that arrives faster than the configured cadence.

#### Scenario: Text too frequent
- **WHEN** two texts arrive within the minimum text interval
- **THEN** the second is answered with a `dropped` message of kind TEXT that includes `retryAfterMs`

#### Scenario: Invalid frame
- **WHEN** a binary message is not a valid JPEG or PNG
- **THEN** the server answers INVALID_FILE_FORMAT and later valid frames are still processed

### Requirement: Typed text is not retained
The system SHALL keep from the text only the derived probabilities and counters during the session, SHALL NOT store the text on disk, and SHALL NOT write the text or the frames to logs.

#### Scenario: Session activity
- **WHEN** a session processes texts and frames
- **THEN** no text content appears in logs or in any server-side session record

### Requirement: Orderly end of the live session
The system SHALL end the session with a `closing` message containing the reason (CLIENT_CLOSE or SESSION_TIMEOUT) and counters of frames, texts and fusions without any content, and SHALL release the session when the client disconnects.

#### Scenario: Client closes
- **WHEN** the client sends `{"type":"close"}`
- **THEN** the server answers `closing` with reason CLIENT_CLOSE and the summary counters

### Requirement: Scope warnings in live updates
The system SHALL include in live updates the warnings that facial expressions and text classifications do not determine the inner emotional state, that incongruence does not imply deception, and that a partially typed text can classify differently from the finished text.

#### Scenario: Update with fusion
- **WHEN** an update carries a fusion
- **THEN** its warnings include the facial-expression limitation, the incongruence limitation and the partial-text note
