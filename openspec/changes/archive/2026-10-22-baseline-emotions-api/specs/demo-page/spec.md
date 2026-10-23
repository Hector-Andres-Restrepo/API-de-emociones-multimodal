# Spec Delta

## Purpose

Ofrecer una página de demostración servida por la propia API para probar cada modalidad desde el navegador sin herramientas adicionales.

## ADDED Requirements

### Requirement: Demo page without secrets
The system SHALL serve a static HTML page at `GET /demo` that contains no secrets, and every API call made from the page SHALL still require the Bearer token typed by the user.

#### Scenario: Page content
- **WHEN** a browser requests `/demo`
- **THEN** the response is 200 HTML that includes the scope limitation notice and no access token

### Requirement: Token kept in memory only
The demo page SHALL keep the access token only in the memory of the open tab and SHALL NOT store it in persistent browser storage.

#### Scenario: Reload
- **WHEN** the tab is reloaded
- **THEN** the token field is empty again

### Requirement: Coverage of every modality
The demo page SHALL let the user try text, image, video, combined text plus image or video, and live camera analysis, showing the probabilities, warnings, model versions and inference time returned by the API.

#### Scenario: Text analysis
- **WHEN** the user submits a text
- **THEN** the page shows the emotion, the seven probabilities, the warnings and the server inference time
