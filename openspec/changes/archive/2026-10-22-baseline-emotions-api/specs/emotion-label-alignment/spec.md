# Spec Delta

## Purpose

Garantizar que las predicciones de todos los modelos se expresen en las mismas siete clases canónicas, con etiquetas en inglés y mayúsculas, sin asumir que los índices de salida de los modelos coinciden entre sí.

## ADDED Requirements

### Requirement: Canonical emotion classes
The system SHALL express every emotion label in English uppercase using exactly seven canonical classes: ANGER, DISGUST, FEAR, HAPPINESS, NEUTRAL, SADNESS and SURPRISE.

#### Scenario: Labels in a text response
- **WHEN** a text analysis returns probabilities
- **THEN** the probability keys are exactly the seven canonical classes and the reported emotion is one of them

### Requirement: Valence of each class
The system SHALL assign positive valence to HAPPINESS, negative valence to ANGER, DISGUST, FEAR and SADNESS, and neutral valence to NEUTRAL and SURPRISE.

#### Scenario: Valence lookup
- **WHEN** the valence of SURPRISE is requested
- **THEN** the result is NEUTRAL and not POSITIVE or NEGATIVE

### Requirement: Explicit label adapters
The system SHALL map each model's real output labels to the canonical classes by name through an explicit adapter, and SHALL NOT assume that output indices coincide between models.

#### Scenario: Text model residual class
- **WHEN** the text model reports `joy` and `others`
- **THEN** `joy` is reported as HAPPINESS and `others` is reported as NEUTRAL

#### Scenario: Visual model output order
- **WHEN** the visual model outputs probabilities in the order Neutral, Happiness, Sadness, Surprise, Fear, Disgust, Anger
- **THEN** the highest value at the last index is reported as ANGER and not as SURPRISE

### Requirement: Label coverage verified at load time
The system SHALL refuse to load a model whose real labels do not cover exactly the seven canonical classes after mapping.

#### Scenario: Unknown label
- **WHEN** a model exposes a label that has no canonical mapping
- **THEN** the model is not marked ready and requests that need it fail with MODEL_UNAVAILABLE

### Requirement: Normalized probabilities
The system SHALL return seven probabilities per modality that are non-negative and sum to approximately 1.

#### Scenario: Normalization
- **WHEN** a raw probability vector does not sum to 1
- **THEN** the reported probabilities are rescaled so that their sum is 1 within rounding error
