# Spec Delta

## Purpose

Estimar la expresión facial de uno o varios rostros presentes en una imagen enviada por el cliente.

## ADDED Requirements

### Requirement: Image analysis endpoint
The system SHALL expose `POST /api/v1/emotions/image` accepting a multipart field `image` (JPEG or PNG, at most 5 MB) and returning, for each detected face, a face id, the predominant emotion, its confidence and the seven probabilities.

#### Scenario: One face
- **WHEN** a valid JPEG with one face is sent
- **THEN** the response is 200 with `modality` IMAGE, `facesDetected` 1, `faces[0].faceId` equal to `face-1`, and `modelVersions.visual`

### Requirement: No face detected
The system SHALL answer 200 with `facesDetected` 0, an empty `faces` list and the warning NO_FACE_DETECTED when the image has no detectable face.

#### Scenario: Image without a face
- **WHEN** an image with no face is sent
- **THEN** the response is 200, `faces` is empty and `warnings` contains NO_FACE_DETECTED

### Requirement: Multiple faces reported separately
The system SHALL report every detected face separately with ids `face-1`, `face-2`, and so on, and SHALL NOT merge faces with each other.

#### Scenario: Three faces
- **WHEN** an image with three faces is sent
- **THEN** `facesDetected` is 3 and each face has its own emotion and probabilities

### Requirement: Format checked by content
The system SHALL accept only real JPEG or PNG content, decided by the file bytes and not by the file name, and SHALL answer 415 INVALID_FILE_FORMAT with `details.field` equal to `image` otherwise.

#### Scenario: Unsupported format
- **WHEN** a GIF, WebP or corrupt file is sent in `image`
- **THEN** the response is 415 with code INVALID_FILE_FORMAT

### Requirement: Image size limit
The system SHALL answer 413 PAYLOAD_TOO_LARGE when the image exceeds the configured limit (5 MB by default).

#### Scenario: Oversized image
- **WHEN** an image larger than the limit is sent
- **THEN** the response is 413 with code PAYLOAD_TOO_LARGE

### Requirement: Image field required
The system SHALL answer 400 VALIDATION_ERROR when the multipart field `image` is missing.

#### Scenario: Missing field
- **WHEN** the request has a file under another field name
- **THEN** the response is 400 with code VALIDATION_ERROR
