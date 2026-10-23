# Spec Delta

## Purpose

Limitar el tratamiento de datos personales y comunicar con claridad el alcance y las limitaciones de un sistema que estima expresiones y clasificaciones, no estados internos.

## ADDED Requirements

### Requirement: No persistence of content
The system SHALL NOT persist images, video frames or text submitted for analysis by default, and SHALL keep uploaded video only in temporary storage for the duration of its job.

#### Scenario: After an image analysis
- **WHEN** an image analysis finishes
- **THEN** no copy of the image exists on disk

### Requirement: Logs without content
The system SHALL NOT write biometric content, images or full text to logs, and SHALL log only exception types and model versions.

#### Scenario: Inference error
- **WHEN** an inference error is logged
- **THEN** the log line contains the exception type and not the request content

### Requirement: Scope and limitation warnings
The system SHALL include in its responses and documentation warnings that facial expressions and text classifications do not determine the inner emotional state or sincerity of a person, and SHALL NOT present incongruence as deception.

#### Scenario: Image response
- **WHEN** an image analysis returns
- **THEN** `warnings` states that facial expressions do not determine the internal emotional state

### Requirement: Real inference only
The system SHALL produce every analysis result with the real models and SHALL NOT return static or simulated results, which may exist only inside automated tests and SHALL be declared there.

#### Scenario: Model failure
- **WHEN** a model is not available
- **THEN** the system answers MODEL_UNAVAILABLE instead of a placeholder prediction

### Requirement: Academic use and licenses
The system SHALL document that it is for academic, non-commercial use, the license of each model and the restriction of the visual weights to scientific use, and SHALL advise obtaining consent before processing images or video of people.

#### Scenario: README
- **WHEN** a reader opens the README
- **THEN** it states the academic scope, the model licenses and the consent requirement
