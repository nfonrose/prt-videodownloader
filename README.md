
# Project prt-videoDownloader

Python based server that allows downloading a video using the `yt-dlp` CLI,
and expose the downloaded result via a public HTTPS URL.

[https://github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp)

A first iteration of the projet requires several calls to the `prt-videoDownloader` API
to initiate the download and get the download URL.

A second iteration of the project should make it possible to initiate the download
and access the downloaded result via a single HTTP GET call (for more convenience when
using it in video workflows).

## Development quickstart (Phase 1)

Prereqs: Python 3.11+ recommended.

1. Create and activate a virtualenv (optional but recommended).
2. Install dependencies:
   - pip install -r requirements.txt
3. Run the API server:
   - python entrypoint.py

The server listens on 0.0.0.0:8080 by default (override with env PORT).

OpenAPI docs are available at:
- Swagger UI: http://localhost:8080/doc
- OpenAPI JSON: http://localhost:8080/openapi

Hello endpoint:
- GET http://localhost:8080/hello → {"message":"hello world"}

SQLite database configuration:
- Env var: PRT_VIDEODOWNLOADER_SQLITEFILEPATH
- Default: /opt/prt/prt-videodownloader/db/prt-videodownloader.sqlite
