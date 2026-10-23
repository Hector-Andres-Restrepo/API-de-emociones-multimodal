# Spec Delta

## MODIFIED Requirements

### Requirement: Coverage of every modality
The demo page SHALL let the user try text, image, video, combined text plus image or video, live camera analysis, and live multimodal analysis while typing, showing the probabilities, warnings, model versions and inference time returned by the API.

#### Scenario: Text analysis
- **WHEN** the user submits a text
- **THEN** the page shows the emotion, the seven probabilities, the warnings and the server inference time

#### Scenario: Live multimodal analysis
- **WHEN** the user starts the camera and types in the live text box
- **THEN** the page shows the facial emotion, the text emotion, the fusion with its incongruence indicator and the fusion status, updating while the user types
