# Spec Delta

## Purpose

Combinar la clasificación de texto con la clasificación visual de un único rostro mediante fusión tardía de probabilidades, e indicar cuándo las valencias de ambas modalidades se oponen.

## ADDED Requirements

### Requirement: Combined analysis input
The system SHALL expose `POST /api/v1/emotions/analyze` accepting `text`, an optional `language`, an optional `visualWeight` between 0 and 1 (default 0.5) and exactly one of `image` or `video`, and SHALL answer 400 VALIDATION_ERROR when neither or both are attached.

#### Scenario: Both visual sources
- **WHEN** a request attaches an image and a video
- **THEN** the response is 400 with code VALIDATION_ERROR

### Requirement: Synchronous image and asynchronous video variants
The system SHALL answer 200 with the final result when an image is attached, and SHALL answer 202 with an `analysisId` queryable at `GET /api/v1/emotions/analyze/{analysisId}` with the same final schema when a video is attached.

#### Scenario: Video variant
- **WHEN** a text and a video with one face are submitted
- **THEN** the response is 202 and the later query returns a COMPLETED result with modality MULTIMODAL

### Requirement: Weighted late fusion
The system SHALL compute, for each class c, `P_fusion(c) = visualWeight * P_visual(c) + (1 - visualWeight) * P_text(c)`, SHALL report the class with the highest fused probability and that value as confidence, and SHALL keep the text and visual results separate in the response.

#### Scenario: Weight extremes
- **WHEN** `visualWeight` is 0
- **THEN** the fused emotion equals the text emotion
- **WHEN** `visualWeight` is 1
- **THEN** the fused emotion equals the visual emotion

#### Scenario: Weight out of range
- **WHEN** `visualWeight` is 1.5
- **THEN** the response is 400 with code VALIDATION_ERROR

### Requirement: Exactly one face for fusion
The system SHALL answer 422 FACE_COUNT_NOT_SUPPORTED_FOR_FUSION when the image or video shows zero or more than one face, and SHALL NOT produce a fusion in that case.

#### Scenario: Two faces in an image
- **WHEN** the image has two faces
- **THEN** the response is 422 with code FACE_COUNT_NOT_SUPPORTED_FOR_FUSION

#### Scenario: Multiple faces appear later in a video
- **WHEN** the initial check passes but later frames show several faces
- **THEN** the job ends FAILED with the same error code

### Requirement: Incongruence indicator
The system SHALL report `incongruence` true only when the predominant text class and the predominant visual class have opposite valences (POSITIVE versus NEGATIVE) and both confidences reach the configured threshold (0.70 by default).

#### Scenario: Opposed valences with high confidence
- **WHEN** the text is SADNESS with confidence 0.80 and the visual is HAPPINESS with confidence 0.85
- **THEN** `incongruence` is true

#### Scenario: Neutral or surprise
- **WHEN** one modality predominates with NEUTRAL or SURPRISE
- **THEN** `incongruence` is false whatever the other modality shows

#### Scenario: Confidence at the threshold
- **WHEN** both confidences equal 0.70 and valences are opposed
- **THEN** `incongruence` is true
- **WHEN** one confidence is 0.69
- **THEN** `incongruence` is false

### Requirement: Incongruence is not deception
The system SHALL include a warning stating that incongruence does not allow inferring deception or emotional concealment, and that weights, threshold and valences are an experimental proposal.

#### Scenario: Warning present
- **WHEN** any fusion result is returned
- **THEN** its `warnings` contain the incongruence limitation text

### Requirement: Configurable fusion parameters
The system SHALL allow the default visual weight and the incongruence threshold to be configured through environment variables, and the visual weight to be overridden per request.

#### Scenario: Custom threshold
- **WHEN** the threshold is set to 0.60 and both confidences are 0.65 with opposed valences
- **THEN** `incongruence` is true
