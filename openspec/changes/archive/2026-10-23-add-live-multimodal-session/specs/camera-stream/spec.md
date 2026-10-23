# Spec Delta

## MODIFIED Requirements

### Requirement: Simultaneous session limit
The system SHALL limit the number of simultaneous sessions, counting camera stream sessions and live multimodal sessions together, to a configured maximum and SHALL answer 429 RATE_LIMIT_EXCEEDED when the limit is reached.

#### Scenario: Limit reached
- **WHEN** the maximum number of sessions is already open
- **THEN** creating another session returns 429 with code RATE_LIMIT_EXCEEDED

#### Scenario: Mixed session types
- **WHEN** the open sessions are a mix of camera stream and live multimodal sessions that together reach the maximum
- **THEN** creating a session of either type returns 429 with code RATE_LIMIT_EXCEEDED
