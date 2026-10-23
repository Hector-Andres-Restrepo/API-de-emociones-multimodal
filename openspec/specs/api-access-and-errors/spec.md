# api-access-and-errors Specification

## Purpose
Proteger los endpoints de análisis con autenticación y límites, y dar a todas las respuestas un formato trazable y un contrato común de errores.

## Requirements

### Requirement: Bearer authentication
The system SHALL require a Bearer token, configured through an environment variable, on every analysis endpoint and SHALL answer 401 UNAUTHORIZED when it is missing or invalid.

#### Scenario: Missing or wrong token
- **WHEN** a request has no Authorization header or a wrong token
- **THEN** the response is 401 with code UNAUTHORIZED and a `requestId`

### Requirement: Traceability of successful responses
The system SHALL include a UUID `requestId` and the `modelVersions` used in every successful analysis response.

#### Scenario: Successful analysis
- **WHEN** an analysis succeeds
- **THEN** the body contains a UUID `requestId` and `modelVersions`

### Requirement: Common error format
The system SHALL answer every HTTP failure with `{requestId, error: {code, message, details}}` using the codes VALIDATION_ERROR (400), UNAUTHORIZED (401), ANALYSIS_NOT_FOUND (404), PAYLOAD_TOO_LARGE (413), INVALID_FILE_FORMAT (415), UNSUPPORTED_LANGUAGE and FACE_COUNT_NOT_SUPPORTED_FOR_FUSION (422), RATE_LIMIT_EXCEEDED (429), INFERENCE_ERROR (500) and MODEL_UNAVAILABLE (503).

#### Scenario: Validation failure
- **WHEN** a required parameter is missing
- **THEN** the body has `error.code` VALIDATION_ERROR and `error.details.fields` listing the field

### Requirement: Request rate limit
The system SHALL limit the number of requests per client per minute to a configured value and SHALL answer 429 RATE_LIMIT_EXCEEDED beyond it.

#### Scenario: Exceeding the limit
- **WHEN** a client sends more requests per minute than allowed
- **THEN** the extra requests return 429 with code RATE_LIMIT_EXCEEDED

### Requirement: Bounded concurrent inference
The system SHALL limit the number of simultaneous inferences and SHALL answer 429 RATE_LIMIT_EXCEEDED when it cannot obtain a free slot within a bounded wait.

#### Scenario: Server busy
- **WHEN** all inference slots stay busy beyond the wait
- **THEN** the request returns 429 instead of waiting indefinitely

### Requirement: Failures after acceptance
The system SHALL report inference failures after a job was accepted through `status` FAILED and an `error` object, and SHALL answer 500 INFERENCE_ERROR for synchronous failures without exposing request content.

#### Scenario: Synchronous inference failure
- **WHEN** inference raises an unexpected error
- **THEN** the response is 500 with code INFERENCE_ERROR and a generic message
