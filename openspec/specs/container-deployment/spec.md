# container-deployment Specification

## Purpose
Poder construir, arrancar y operar la API en un contenedor Linux reproducible, con salud observable, modelos persistentes y configuración por variables de entorno.

## Requirements

### Requirement: Start with Docker Compose
The system SHALL build and start with `docker compose up --build`, expose the API on the configured port (8000 by default) and run as a non-privileged user.

#### Scenario: First start
- **WHEN** the project is started on a clean Linux host
- **THEN** the API listens on the configured port and the container becomes healthy once the process is alive

### Requirement: Health endpoints
The system SHALL expose `GET /health/live`, answering 200 while the process is alive, and `GET /health/ready`, answering 200 only when both the text and visual models are loaded and 503 with the load errors otherwise, both outside the `/api/v1` prefix and without authentication.

#### Scenario: Models still loading
- **WHEN** a model has not finished loading
- **THEN** `/health/ready` answers 503 and `/health/live` answers 200

### Requirement: Persistent model storage
The system SHALL keep model weights and download cache in a persistent volume so they survive restarts, and SHALL download them on first start.

#### Scenario: Restart
- **WHEN** the container restarts
- **THEN** the models load from the volume without downloading again

### Requirement: Configuration by environment
The system SHALL read the access key, model identifiers and paths, fusion weight, incongruence threshold and operating limits from environment variables, and the repository SHALL contain only an example file without secrets.

#### Scenario: Missing access key
- **WHEN** the access key variable is not defined
- **THEN** the application refuses to start with an explicit message

### Requirement: Documented resource limits
The system SHALL bound memory and CPU through the Compose file and SHALL bound concurrency, session duration, queue length and file sizes through configuration, with the defaults documented.

#### Scenario: Defaults
- **WHEN** no limit variable is set
- **THEN** the documented defaults apply (4 GB memory, 2 CPUs, 2 concurrent inferences, 300 s sessions)

### Requirement: Temporary storage cleanup
The system SHALL write temporary files only to a dedicated temporary location and SHALL remove leftovers when it starts.

#### Scenario: Restart after interruption
- **WHEN** the service starts and old job directories exist
- **THEN** they are deleted before serving requests

### Requirement: Start without models
The system SHALL be able to start without loaded models when so configured, reporting not ready and answering MODEL_UNAVAILABLE instead of fabricating results.

#### Scenario: Weights unavailable
- **WHEN** weights cannot be downloaded
- **THEN** `/health/ready` reports the load error and analysis endpoints answer 503
