# Spec Delta

## Purpose

Analizar videos grabados de forma asíncrona aprovechando el componente temporal LSTM, entregando la expresión dominante y una línea de tiempo.

## ADDED Requirements

### Requirement: Asynchronous video submission
The system SHALL expose `POST /api/v1/emotions/video` accepting an MP4 file in the multipart field `video` and an optional `sampleFps` between 1 and 5 (default 2), and SHALL answer 202 with `analysisId`, `status` and `resultUrl`.

#### Scenario: Accepted video
- **WHEN** a valid MP4 of at most 50 MB and 60 seconds is submitted
- **THEN** the response is 202 with an `analysisId` starting with `vid-` and a `resultUrl` ending in that id

### Requirement: Video input validation
The system SHALL answer 415 INVALID_FILE_FORMAT for non-MP4 content, 413 PAYLOAD_TOO_LARGE above the size limit, and 400 VALIDATION_ERROR when the duration exceeds 60 seconds or `sampleFps` is outside 1 to 5.

#### Scenario: Invalid parameters
- **WHEN** `sampleFps` is 0, 6 or not a number
- **THEN** the response is 400 with code VALIDATION_ERROR and no job is created

### Requirement: Job status lifecycle
The system SHALL report job status as PENDING, PROCESSING, COMPLETED or FAILED through `GET /api/v1/emotions/video/{analysisId}`, with `dominantExpression` and `timeline` null while not finished and an `error` object when FAILED.

#### Scenario: Failed job
- **WHEN** inference fails after the job was accepted
- **THEN** the query returns `status` FAILED with `error.code` INFERENCE_ERROR

#### Scenario: Unknown identifier
- **WHEN** the analysis id does not exist or expired
- **THEN** the response is 404 with code ANALYSIS_NOT_FOUND

### Requirement: Temporal model for video
The system SHALL process sampled frames as sequences through the temporal LSTM model using a sliding window of frames, and SHALL NOT replace this behavior with averaging of independent static-image predictions.

#### Scenario: Completed analysis
- **WHEN** a job completes
- **THEN** the result states `temporalModel.type` LSTM with its window size and the timeline segments are computed from the temporal model outputs

### Requirement: Video temporary files removed
The system SHALL delete the stored video when the job ends in any state and when a submission is rejected.

#### Scenario: After completion or failure
- **WHEN** a job reaches COMPLETED or FAILED
- **THEN** no job directory remains in the temporary storage

### Requirement: Principal face in video
The system SHALL analyze the face with the largest area when several faces appear and SHALL report the maximum number of faces seen and the number of frames with more than one face.

#### Scenario: Two people in view
- **WHEN** some frames contain two faces
- **THEN** the result reports `multiFaceFrames` greater than 0 and analyzes the largest face
