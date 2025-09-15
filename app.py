import os
import uuid
import subprocess
import shlex
import threading
import logging
import json
from enum import Enum
from typing import List, Optional
from datetime import datetime, timedelta
from flask_openapi3 import OpenAPI, Info, Tag
from flask import jsonify, request
from pydantic import BaseModel, Field
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Enum as SAEnum,
    BigInteger,
    Float,
    DateTime,
    func,
    text,
)
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuration for SQLAlchemy with SQLite
PATHPREFIX_FOR_DEVLOCAL_ENV = "/Users/teevity/Dev/misc/1.prtVideoDownloader"
DEFAULT_DB_PATH = (PATHPREFIX_FOR_DEVLOCAL_ENV + "/opt/prt/prt-videodownloader/db/prt-videodownloader.sqlite")
DB_ENV_VAR = "PRT_VIDEODOWNLOADER_SQLITEFILEPATH"

# Video data path configuration
DEFAULT_DATA_PATH = PATHPREFIX_FOR_DEVLOCAL_ENV + "/opt/prt/prt-videodownloader/data"
DATA_ENV_VAR = "PRT_VIDEODOWNLOADER_VIDEODATAPATH"


def ensure_dir(path: str) -> str:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def get_db_path() -> str:
    path = os.getenv(DB_ENV_VAR, DEFAULT_DB_PATH)
    # Ensure directory exists
    dir_path = os.path.dirname(path)
    ensure_dir(dir_path)
    return path


def get_data_path() -> str:
    path = os.getenv(DATA_ENV_VAR, DEFAULT_DATA_PATH)
    return ensure_dir(path)


# Initialize SQLAlchemy engine and session factory
DATABASE_PATH = get_db_path()
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # allow usage in Flask dev server threads
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy Declarative Base
Base = declarative_base()

# Download status enum
class DownloadStatusEnum(Enum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    COMPLETED = "COMPLETED"
    COMPLETED_ALREADYDOWNLOADED = "COMPLETED_ALREADYDOWNLOADED"
    FAILED = "FAILED"


# Download model
class Download(Base):
    __tablename__ = "downloads"

    # Using string UUID for portability with SQLite
    uuid = Column(String(36), primary_key=True, nullable=False)
    videoURL = Column(String(2048), nullable=False)
    status = Column(SAEnum(DownloadStatusEnum), nullable=False, default=DownloadStatusEnum.PENDING)
    sizeInBytes = Column(BigInteger, nullable=True)
    progress = Column(Float, nullable=True)  # 0.0 to 100.0 or 0.0 to 1.0 depending on later choice
    requestDataEpochMs = Column(BigInteger, nullable=True)
    createdAt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updatedAt = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Initialize Flask OpenAPI app
info = Info(title="PRT VideoDownloader API", version="0.1.0")
app = OpenAPI(__name__, info=info)
logger = logging.getLogger(__name__)

hello_tag = Tag(name="hello", description="Hello world operations")
initiate_tag = Tag(name="downloads", description="Download operations")


class InitiateDownloadRequest(BaseModel):
    videoURL: str = Field(..., description="URL of the video to download")
    ytdlpCLIParameters: Optional[List[str]] = Field(default=None, description="Additional yt-dlp CLI parameters")


class InitiateDownloadResponse(BaseModel):
    uuid: str


class ErrorResponse(BaseModel):
    errorCode: str
    errorMessage: str


class ListDownloadsQuery(BaseModel):
    overLastPeriodDurationInMinutes: int = Field(60, ge=1, description="Time window in minutes to include downloads from now back to this many minutes")
    onlyShowOngoingDownloads: bool = Field(False, description="If true, only include downloads that are currently ongoing (PENDING or DOWNLOADING)")


class DownloadListItem(BaseModel):
    downloadUUID: str
    downloadStatus: str
    downloadSizeInBytes: Optional[int] = None
    progress: Optional[float] = None
    requestDataEpochMs: Optional[int] = None


class ListDownloadsResponse(BaseModel):
    downloads: List[DownloadListItem]


def error_response(error_code: str, error_message: str):
    return jsonify({"errorCode": error_code, "errorMessage": error_message}), 418


@app.get("/hello", tags=[hello_tag], summary="Hello World")
def hello_world():
    return jsonify({"message": "hello world"})


# Health endpoint for basic checks (not required but handy)
@app.get("/healthz", summary="Health check")
def healthz():
    return jsonify({"status": "ok"})


# Middleware to log requests as cURL commands in multi-line format
@app.before_request
def log_request_as_curl():
    # Skip logging for certain endpoints if needed (e.g., health checks)
    if request.path == '/favicon.ico':  # Example: Skip favicon requests
        return

    # Start building the cURL command
    curl_cmd = ["curl"]

    # Add the HTTP method and URL on the first line
    curl_cmd.append(f"-X {request.method} '{request.url}'")

    # Add headers on separate lines
    for header_name, header_value in request.headers.items():
        # Skip headers that are not typically needed in cURL
        if header_name.lower() in ('content-length', 'host'):
            continue
        curl_cmd.append(f"-H '{header_name}: {header_value}'")

    # Add request body for POST, PUT, etc. on separate lines
    if request.method in ('POST', 'PUT', 'PATCH') and request.get_data():
        # Handle JSON data
        if request.is_json:
            data = json.dumps(request.get_json(), indent=None)
            curl_cmd.append(f"--data-raw '{data}'")
        # Handle form data
        elif request.form:
            for key, value in request.form.items():
                curl_cmd.append(f"-d '{key}={value}'")
        # Handle raw data
        else:
            data = request.get_data(as_text=True)
            curl_cmd.append(f"--data-raw '{data}'")

    # Join the command parts with backslash and newline for multi-line format
    # First line (curl -X METHOD URL) has no backslash
    curl_command = f"{curl_cmd[0]} {curl_cmd[1]}"  # Combine 'curl' and '-X METHOD URL' on first line
    # Add remaining arguments, each on a new line with backslash
    for part in curl_cmd[2:]:
        curl_command += f" \\\n  {part}"

    # Log the multi-line cURL command
    logger.info(f" -------------------------------------------------------------------------------------------------------------------------------------\n{curl_command}")



@app.post(
    "/downloads/initiate",
    tags=[initiate_tag],
    summary="Initiate a video download",
    responses={
        200: InitiateDownloadResponse,
        418: ErrorResponse,
    },
)
def initiate_download(body: InitiateDownloadRequest):
    # Validate input
    video_url = body.videoURL.strip() if body.videoURL else ""
    if not video_url:
        return error_response("invalidInput", "videoURL is required")

    # Prepare DB entry
    download_uuid = str(uuid.uuid4())
    session = SessionLocal()
    try:
        dl = Download(
            uuid=download_uuid,
            videoURL=video_url,
            status=DownloadStatusEnum.PENDING,
            requestDataEpochMs=int(datetime.utcnow().timestamp() * 1000),
        )
        session.add(dl)
        session.commit()
    except Exception as e:
        session.rollback()
        return error_response("dbError", f"Failed to create download record: {e}")

    # Build yt-dlp command
    data_path = get_data_path()
    mandatory_params = [
        "-f",
        "bestvideo[height<=1080]+bestaudio/best",     # This was downloading all videos streams "bv*+ba/best",
        "-o",
        os.path.join(data_path, "%(title)s-%(id)s.%(ext)s"),
        "--progress",
        "--progress-template",
        "download:%(progress._percent_str)s:%(progress.downloaded_bytes)s:%(progress.total_bytes)s",
    ]

    user_params: List[str] = []
    if body.ytdlpCLIParameters:
        # Filter out attempts to override mandatory params '-f' or '-o'
        skip_next = False
        for p in body.ytdlpCLIParameters:
            if skip_next:
                skip_next = False
                continue
            if p in ("-f", "--format", "-o", "--output"):
                skip_next = True  # skip its value
                continue
            user_params.append(p)

    cmd = [
        "yt-dlp",
        *mandatory_params,
        *user_params,
        video_url,
    ]

    # Start process in background
    app.logger.info("Executing yt-dlp command [%s]: [%s]", download_uuid, shlex.join(cmd))
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        # Shared flag to detect if yt-dlp reported the file was already downloaded
        already_downloaded_flag = {"val": False}

        # Async log subprocess output
        def _log_stream(stream, is_err=False):
            log_fn = app.logger.error if is_err else app.logger.info
            for line in iter(stream.readline, ''):
                line = line.rstrip('\n')
                if not line:
                    continue
                # Always log raw line
                log_fn("DOWNLOAD[%s] %s: %s", download_uuid, "stderr" if is_err else "stdout", line)

                # Detect "already downloaded" message from yt-dlp on either stream
                try:
                    low = line.lower()
                    if "has already been downloaded" in low or "[download]" in low and "already" in low and "downloaded" in low:
                        if not already_downloaded_flag["val"]:
                            already_downloaded_flag["val"] = True
                            app.logger.info("DOWNLOAD[%s] detected already-downloaded message", download_uuid)
                except Exception:
                    pass

                # Parse progress lines emitted via --progress-template
                try:
                    if not is_err and line.startswith("download:"):
                        # Expected format: download:<percent%>:<downloaded_bytes>:<total_bytes>
                        parts = line.split(":", 4)
                        if len(parts) >= 4:
                            percent_raw = parts[1].strip().rstrip('%')
                            downloaded_raw = parts[2].strip()
                            total_raw = parts[3].strip()
                            progress_val = None
                            size_total = None
                            try:
                                progress_val = float(percent_raw)
                            except Exception:
                                progress_val = None
                            try:
                                size_total = int(total_raw)
                            except Exception:
                                size_total = None
                            # Update DB with progress and size
                            if progress_val is not None or size_total is not None:
                                s = SessionLocal()
                                try:
                                    d = s.get(Download, download_uuid)
                                    if d:
                                        if progress_val is not None:
                                            d.progress = progress_val
                                        if size_total is not None:
                                            d.sizeInBytes = size_total
                                        s.add(d)
                                        s.commit()
                                except Exception as e:
                                    s.rollback()
                                    app.logger.debug("DOWNLOAD[%s] progress update failed: %s", download_uuid, e)
                                finally:
                                    s.close()
                except Exception as e:
                    app.logger.debug("DOWNLOAD[%s] progress parse error: %s", download_uuid, e)
            stream.close()
        threading.Thread(target=_log_stream, args=(proc.stdout, False), daemon=True).start()
        threading.Thread(target=_log_stream, args=(proc.stderr, True), daemon=True).start()

        # Watcher thread to capture process completion and update final status
        def _wait_and_finalize():
            rc = proc.wait()
            try:
                app.logger.info("DOWNLOAD[%s] process completed with return code: %s", download_uuid, rc)
            except Exception:
                pass
            s = SessionLocal()
            try:
                d = s.get(Download, download_uuid)
                if d:
                    if rc == 0:
                        if already_downloaded_flag.get("val"):
                            d.status = DownloadStatusEnum.COMPLETED_ALREADYDOWNLOADED
                        else:
                            d.status = DownloadStatusEnum.COMPLETED
                        # Ensure progress shows complete
                        if d.progress is None or d.progress < 100.0:
                            d.progress = 100.0
                    else:
                        d.status = DownloadStatusEnum.FAILED
                    s.add(d)
                    s.commit()
            except Exception as e:
                s.rollback()
                app.logger.error("DOWNLOAD[%s] failed to update final status: %s", download_uuid, e)
            finally:
                s.close()
        threading.Thread(target=_wait_and_finalize, daemon=True).start()

        # Update status to DOWNLOADING
        try:
            dl.status = DownloadStatusEnum.DOWNLOADING
            session.add(dl)
            session.commit()
        except Exception:
            session.rollback()
        return jsonify({"uuid": download_uuid})
    except FileNotFoundError:
        # yt-dlp not found
        try:
            dl.status = DownloadStatusEnum.FAILED
            session.add(dl)
            session.commit()
        except Exception:
            session.rollback()
        return error_response("ytDlpNotFound", "yt-dlp executable not found on PATH")
    except Exception as e:
        try:
            dl.status = DownloadStatusEnum.FAILED
            session.add(dl)
            session.commit()
        except Exception:
            session.rollback()
        return error_response("processStartError", f"Failed to start yt-dlp: {e}")
    finally:
        session.close()


@app.get(
    "/downloads/list",
    tags=[initiate_tag],
    summary="List downloads",
    responses={
        200: ListDownloadsResponse,
        418: ErrorResponse,
    },
)
def list_downloads(query: ListDownloadsQuery):
    session = SessionLocal()
    try:
        minutes = query.overLastPeriodDurationInMinutes if query.overLastPeriodDurationInMinutes is not None else 60
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        q = session.query(Download).filter(Download.createdAt >= cutoff)
        if query.onlyShowOngoingDownloads:
            q = q.filter(Download.status.in_([DownloadStatusEnum.PENDING, DownloadStatusEnum.DOWNLOADING]))
        rows = q.order_by(Download.createdAt.desc()).all()
        items: List[DownloadListItem] = []
        for r in rows:
            items.append(
                DownloadListItem(
                    downloadUUID=r.uuid,
                    downloadStatus=r.status.value if isinstance(r.status, DownloadStatusEnum) else str(r.status),
                    downloadSizeInBytes=r.sizeInBytes,
                    progress=r.progress,
                    requestDataEpochMs=r.requestDataEpochMs,
                )
            )
        resp = ListDownloadsResponse(downloads=items)
        return jsonify(resp.model_dump())
    except Exception as e:
        return error_response("listError", f"Failed to list downloads: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    # Allow external connections (not limited to 127.0.0.1)
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)