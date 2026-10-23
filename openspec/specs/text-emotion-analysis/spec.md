# text-emotion-analysis Specification

## Purpose
Permitir clasificar la emoción de un texto en español de forma independiente, sin requerir ninguna entrada visual.

## Requirements

### Requirement: Text analysis endpoint
The system SHALL expose `POST /api/v1/emotions/text` accepting JSON with `text` and `language` and returning modality TEXT, the predominant emotion, its confidence, the seven probabilities, model versions and warnings.

#### Scenario: Valid Spanish text
- **WHEN** a client sends `{"text": "Hoy me siento muy feliz", "language": "es"}` with a valid token
- **THEN** the response is 200 with `modality` TEXT, `emotion`, `confidence`, `probabilities`, `modelVersions.text`, `warnings` and a `requestId`

### Requirement: Text validation
The system SHALL require `text` to be present, not empty or whitespace-only, and at most 5000 characters, and SHALL answer 400 VALIDATION_ERROR otherwise.

#### Scenario: Length boundary
- **WHEN** the text has exactly 5000 characters
- **THEN** the request is accepted

#### Scenario: Over the limit or empty
- **WHEN** the text has 5001 characters, is empty, or is only spaces
- **THEN** the response is 400 with code VALIDATION_ERROR

### Requirement: Spanish only for now
The system SHALL accept only `es` as language and SHALL answer 422 UNSUPPORTED_LANGUAGE for any other value.

#### Scenario: English text
- **WHEN** `language` is `en`
- **THEN** the response is 422 with code UNSUPPORTED_LANGUAGE

### Requirement: Truncation warning
The system SHALL analyze at most the number of tokens the text model supports and SHALL add a TEXT_TRUNCATED warning when the text exceeds that limit.

#### Scenario: Long text
- **WHEN** the text exceeds the model token limit
- **THEN** the analysis uses the first tokens only and the warnings include TEXT_TRUNCATED

### Requirement: Text model availability
The system SHALL answer 503 MODEL_UNAVAILABLE when the text model is not loaded.

#### Scenario: Model not loaded
- **WHEN** a text request arrives before the text model finished loading
- **THEN** the response is 503 with code MODEL_UNAVAILABLE
