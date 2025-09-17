import os
import uuid
import subprocess
import shlex
import threading
import logging
import json
import time
from enum import Enum
from typing import List, Optional
from datetime import datetime, timedelta, timezone
from flask_openapi3 import OpenAPI, Info, Tag
from flask import jsonify, request, redirect
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
from sqlalchemy import Text
from urllib.parse import quote

# Configuration for SQLAlchemy with SQLite
PATHPREFIX_FOR_DEVLOCAL_ENV = "/Users/teevity/Dev/misc/1.prtVideoDownloader"
DEFAULT_DB_PATH = (PATHPREFIX_FOR_DEVLOCAL_ENV + "/opt/prt/prt-videodownloader/db/prt-videodownloader.sqlite")
DB_ENV_VAR = "PRT_VIDEODOWNLOADER_SQLITEFILEPATH"

# Video data path configuration
DEFAULT_DATA_PATH = PATHPREFIX_FOR_DEVLOCAL_ENV + "/opt/prt/prt-videodownloader/data"
DATA_ENV_VAR = "PRT_VIDEODOWNLOADER_VIDEODATAPATH"

# Public URL base configuration
DEFAULT_HTTPS_BASE_URL = "https://127.0.0.1:9443/videodownloader"   # Production URL https://prt.teevity.com:9443/videodownloader
HTTPS_BASEURL_ENV_VAR = "PRT_VIDEODOWNLOADER_HTTPSBASEURL"
# For S3, we rely on a base URL that points to the RustFS S3-compatible endpoint.
# This can be set to something like: s3://prt-videodownloader/data or http(s) URL served by rustfs.
DEFAULT_S3_BASE_URL = "s3://prt-videodownloader/data"
S3_BASEURL_ENV_VAR = "PRT_VIDEODOWNLOADER_S3BASEURL"

# Redirect polling configuration
MAX_RETRIES = int(os.getenv("PRT_VIDEODOWNLOADER_MAX_RETRIES", "5"))
RETRY_SLEEP_SECONDS = float(os.getenv("PRT_VIDEODOWNLOADER_RETRY_SLEEP_SECONDS", "3"))


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


def get_https_base_url() -> str:
    return os.getenv(HTTPS_BASEURL_ENV_VAR, DEFAULT_HTTPS_BASE_URL)


def get_s3_base_url() -> str:
    return os.getenv(S3_BASEURL_ENV_VAR, DEFAULT_S3_BASE_URL)


def build_public_url(file_name: str, url_type: "URLTypeEnum") -> str:
    # Ensure file name is URL-encoded
    encoded_name = quote(file_name)
    if url_type == URLTypeEnum.HTTPS:
        base = get_https_base_url().rstrip("/")
        return f"{base}/data/{encoded_name}"
    elif url_type == URLTypeEnum.S3:
        base = get_s3_base_url().rstrip("/")
        return f"{base}/{encoded_name}"
    else:
        # Fallback; should not happen
        base = get_https_base_url().rstrip("/")
        return f"{base}/data/{encoded_name}"


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
    requestCreationDateEpochMs = Column(BigInteger, nullable=True)
    # Name of the output file stored under the data directory; basename only
    fileName = Column(String(4096), nullable=True)
    createdAt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updatedAt = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class DownloadError(Base):
    __tablename__ = "download_errors"

    # Reuse UUID as primary key; one error per download (latest overwrites)
    uuid = Column(String(36), primary_key=True, nullable=False)
    errorMessage = Column(Text, nullable=False)
    createdAt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Initialize Flask OpenAPI app
info = Info(title="PRT VideoDownloader API", version="0.1.0")
app = OpenAPI(__name__, info=info)
logger = logging.getLogger(__name__)

hello_tag = Tag(name="hello", description="Hello world operations")
initiate_tag = Tag(name="downloads", description="Download operations")


class URLTypeEnum(str, Enum):
    HTTPS = "HTTPS"
    S3 = "S3"


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
    requestCreationDateEpochMs: Optional[str] = None
    fileName: Optional[str] = None
    errorMessage: Optional[str] = None


class ListDownloadsResponse(BaseModel):
    downloads: List[DownloadListItem]


class GetPublicURLQuery(BaseModel):
    videoDownloadUUID: str = Field(..., description="UUID of the download request")
    urlType: URLTypeEnum = Field(..., description="Type of URL to generate: HTTPS or S3")


class GetPublicURLResponse(BaseModel):
    publicURL: str


class RedirectToHTTPSQuery(BaseModel):
    video_id: str = Field(..., description="The ID (UUID) of the video download task")
    retryCounter: int = Field(0, ge=0, description="Number of times this endpoint has redirected so far")


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
            requestCreationDateEpochMs=int(datetime.now(timezone.utc).timestamp() * 1000),
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
        # Collect stderr lines to determine error cause on failure (bounded buffer)
        stderr_lines_buffer = {"lines": []}
        stderr_max_lines = 200
        # Track output filenames captured from yt-dlp logs
        file_capture = {"merge": None, "already": None, "dest": []}

        # Async log subprocess output
        def _log_stream(stream, is_err=False):
            log_fn = app.logger.error if is_err else app.logger.info
            for line in iter(stream.readline, ''):
                line = line.rstrip('\n')
                if not line:
                    continue
                # Always log raw line
                log_fn("DOWNLOAD[%s] %s: %s", download_uuid, "stderr" if is_err else "stdout", line)

                # If stderr, keep a copy to extract cause later
                if is_err:
                    try:
                        stderr_lines_buffer["lines"].append(line)
                        if len(stderr_lines_buffer["lines"]) > stderr_max_lines:
                            # Drop oldest to keep memory bounded
                            del stderr_lines_buffer["lines"][0:len(stderr_lines_buffer["lines"]) - stderr_max_lines]
                    except Exception:
                        pass

                # Capture output file paths and detect "already downloaded" messages
                try:
                    low = line.lower()
                    # Merger final output filename
                    if "merging formats into" in low:
                        try:
                            fn = None
                            if '"' in line:
                                sidx = line.find('"')
                                eidx = line.rfind('"')
                                if eidx > sidx >= 0:
                                    fn = line[sidx + 1:eidx]
                            if not fn:
                                k = low.rfind("into ")
                                if k != -1:
                                    fn = line[k + 5:].strip().strip('"')
                            if fn:
                                base = os.path.basename(fn)
                                file_capture["merge"] = base
                                app.logger.info("DOWNLOAD[%s] detected merged output file: %s", download_uuid, base)
                        except Exception:
                            pass
                    # Destination lines for downloaded streams (video/audio)
                    if "[download]" in low and "destination:" in low:
                        try:
                            k = low.find("destination:")
                            fn = line[k + len("destination:"):].strip().strip('"') if k != -1 else None
                            if fn:
                                base = os.path.basename(fn)
                                file_capture["dest"].append(base)
                        except Exception:
                            pass
                    # Already downloaded line, also contains filename
                    if "has already been downloaded" in low:
                        try:
                            endk = low.find("has already been downloaded")
                            prefix = line[:endk].strip()
                            rb = prefix.rfind("] ")
                            if rb != -1:
                                prefix = prefix[rb + 2:]
                            fn = prefix.strip().strip('"') if prefix else None
                            if fn:
                                base = os.path.basename(fn)
                                file_capture["already"] = base
                        except Exception:
                            pass
                    # Maintain boolean flag for status
                    if "has already been downloaded" in low or ("[download]" in low and "already" in low and "downloaded" in low):
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
            # If failed, try to extract an error cause from captured stderr
            if rc != 0:
                cause = None
                try:
                    lines = stderr_lines_buffer.get("lines") or []
                    # Prefer the last line containing 'ERROR:'
                    for l in reversed(lines):
                        if "ERROR:" in l or l.strip().startswith("ERROR"):
                            cause = l.strip()
                            break
                    if cause is None and lines:
                        # Fallback to the last non-empty stderr line
                        for l in reversed(lines):
                            if l and l.strip():
                                cause = l.strip()
                                break
                except Exception:
                    pass
                if cause:
                    app.logger.error("DOWNLOAD[%s] failure cause detected from stderr: %s", download_uuid, cause)
                else:
                    app.logger.error("DOWNLOAD[%s] failure cause: unknown (no stderr output)", download_uuid)

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
                        # Determine and persist final output filename (no UUID in name)
                        try:
                            data_path_loc = get_data_path()
                            best_name = None
                            # Prefer merged output filename
                            if file_capture.get("merge"):
                                best_name = file_capture.get("merge")
                            # Fallbacks
                            if not best_name:
                                # Prefer last destination that doesn't look like an intermediate '.f###.' file
                                dests = file_capture.get("dest") or []
                                non_inter = [n for n in dests if ".f" not in n]
                                if non_inter:
                                    best_name = non_inter[-1]
                                elif dests:
                                    best_name = dests[-1]
                            if not best_name and file_capture.get("already"):
                                best_name = file_capture.get("already")
                            if best_name:
                                d.fileName = best_name
                                # Persist size
                                try:
                                    fullp = os.path.join(data_path_loc, best_name)
                                    stat = os.stat(fullp)
                                    if stat.st_size is not None and (d.sizeInBytes is None or d.sizeInBytes <= 0):
                                        d.sizeInBytes = int(stat.st_size)
                                except Exception:
                                    pass
                            else:
                                app.logger.debug("DOWNLOAD[%s] could not detect output file name from logs", download_uuid)
                        except Exception as e2:
                            app.logger.debug("DOWNLOAD[%s] failed while setting output file name: %s", download_uuid, e2)
                    else:
                        d.status = DownloadStatusEnum.FAILED
                        # Upsert error message into DownloadError table
                        try:
                            # Prefer detected cause; otherwise, set a generic message
                            err_msg = cause if 'cause' in locals() and cause else "Unknown error"
                            existing = s.get(DownloadError, download_uuid)
                            if existing:
                                existing.errorMessage = err_msg
                                s.add(existing)
                            else:
                                s.add(DownloadError(uuid=download_uuid, errorMessage=err_msg))
                        except Exception as e2:
                            app.logger.debug("DOWNLOAD[%s] failed to upsert error message: %s", download_uuid, e2)
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
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        q = session.query(Download).filter(Download.createdAt >= cutoff)
        if query.onlyShowOngoingDownloads:
            q = q.filter(Download.status.in_([DownloadStatusEnum.PENDING, DownloadStatusEnum.DOWNLOADING]))
        rows = q.order_by(Download.createdAt.desc()).all()

        # Fetch error messages for these downloads in one query
        uuids = [r.uuid for r in rows]
        error_by_uuid = {}
        if uuids:
            try:
                err_rows = session.query(DownloadError).filter(DownloadError.uuid.in_(uuids)).all()
                error_by_uuid = {er.uuid: er.errorMessage for er in err_rows}
            except Exception as e2:
                app.logger.debug("Failed to fetch error messages for list_downloads: %s", e2)

        items: List[DownloadListItem] = []
        for r in rows:
            # Transform epoch ms to human-readable UTC string if present
            req_created_human = None
            try:
                if r.requestCreationDateEpochMs is not None:
                    req_created_human = datetime.fromtimestamp(r.requestCreationDateEpochMs / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            except Exception:
                req_created_human = None
            items.append(
                DownloadListItem(
                    downloadUUID=r.uuid,
                    downloadStatus=r.status.value if isinstance(r.status, DownloadStatusEnum) else str(r.status),
                    downloadSizeInBytes=r.sizeInBytes,
                    progress=r.progress,
                    requestCreationDateEpochMs=req_created_human,
                    fileName=r.fileName,
                    errorMessage=error_by_uuid.get(r.uuid),
                )
            )
        resp = ListDownloadsResponse(downloads=items)
        return jsonify(resp.model_dump())
    except Exception as e:
        return error_response("listError", f"Failed to list downloads: {e}")
    finally:
        session.close()


@app.get(
    "/downloads/publicURL",
    tags=[initiate_tag],
    summary="Get public URL for a downloaded video",
    responses={
        200: GetPublicURLResponse,
        418: ErrorResponse,
    },
)
def get_public_url(query: GetPublicURLQuery):
    session = SessionLocal()
    try:
        d = session.get(Download, query.videoDownloadUUID)
        if not d:
            return error_response("invalidDownloadUUID", "download UUID not found")

        # Check status
        st = d.status.value if isinstance(d.status, DownloadStatusEnum) else str(d.status)
        if st == DownloadStatusEnum.FAILED.value:
            # Retrieve error message if any
            err_msg = None
            try:
                derr = session.get(DownloadError, query.videoDownloadUUID)
                if derr:
                    err_msg = derr.errorMessage
            except Exception:
                pass
            return error_response("downloadFailed", err_msg or "The download failed")
        if st not in (DownloadStatusEnum.COMPLETED.value, DownloadStatusEnum.COMPLETED_ALREADYDOWNLOADED.value):
            return error_response("downloadNotReady", "The download is not completed yet")

        # Determine file name
        file_name = d.fileName
        data_path = get_data_path()
        if not file_name:
            # Try to locate by UUID prefix (new naming scheme)
            prefix = f"{d.uuid}-"
            try:
                candidates = []
                for name in os.listdir(data_path):
                    if name.startswith(prefix):
                        try:
                            fullp = os.path.join(data_path, name)
                            stat = os.stat(fullp)
                            candidates.append((stat.st_mtime, name))
                        except Exception:
                            pass
                if candidates:
                    candidates.sort(key=lambda t: t[0], reverse=True)
                    file_name = candidates[0][1]
                    # persist on record
                    try:
                        d.fileName = file_name
                        session.add(d)
                        session.commit()
                    except Exception:
                        session.rollback()
                else:
                    return error_response("fileNotFound", "Video file not found on disk yet")
            except Exception as e2:
                return error_response("fileLookupError", f"Failed to locate video file: {e2}")
        else:
            # Verify it still exists
            try:
                if not os.path.exists(os.path.join(data_path, file_name)):
                    return error_response("fileNotFound", "Video file not found on disk")
            except Exception:
                pass

        # Build URL
        url = build_public_url(file_name, query.urlType)
        return jsonify(GetPublicURLResponse(publicURL=url).model_dump())
    except Exception as e:
        return error_response("publicUrlError", f"Failed to compute public URL: {e}")
    finally:
        session.close()



@app.get(
    "/redirectToHTTPSVideoDownloadPublicURL",
    tags=[initiate_tag],
    summary="Redirect to HTTPS public URL of a video when ready",
    responses={
        418: ErrorResponse,
    },
)
def redirect_to_https_video_download_public_url(query: RedirectToHTTPSQuery):
    current_retry = query.retryCounter if query.retryCounter is not None else 0

    # Enforce retry limit
    if current_retry >= MAX_RETRIES:
        return error_response("downloadNotReadyInTime", "Download not ready in time")

    session = SessionLocal()

    def _redirect_if_ready(d_obj: Download):
        # Determine file name and return redirect to HTTPS public URL if possible
        st_local = d_obj.status.value if isinstance(d_obj.status, DownloadStatusEnum) else str(d_obj.status)
        if st_local in (DownloadStatusEnum.COMPLETED.value, DownloadStatusEnum.COMPLETED_ALREADYDOWNLOADED.value):
            file_name = d_obj.fileName
            data_path_local = get_data_path()
            if not file_name:
                # Try to locate by UUID prefix
                prefix = f"{d_obj.uuid}-"
                try:
                    candidates = []
                    for name in os.listdir(data_path_local):
                        if name.startswith(prefix):
                            try:
                                fullp = os.path.join(data_path_local, name)
                                stat = os.stat(fullp)
                                candidates.append((stat.st_mtime, name))
                            except Exception:
                                pass
                    if candidates:
                        candidates.sort(key=lambda t: t[0], reverse=True)
                        file_name_found = candidates[0][1]
                        try:
                            d_obj.fileName = file_name_found
                            session.add(d_obj)
                            session.commit()
                        except Exception:
                            session.rollback()
                        url_local = build_public_url(file_name_found, URLTypeEnum.HTTPS)
                        return redirect(url_local, code=302)
                    else:
                        return None
                except Exception:
                    return None
            else:
                try:
                    if not os.path.exists(os.path.join(data_path_local, file_name)):
                        return None
                except Exception:
                    pass
                url_local = build_public_url(file_name, URLTypeEnum.HTTPS)
                return redirect(url_local, code=302)
        return None

    try:
        d = session.get(Download, query.video_id)
        if not d:
            return error_response("invalidDownloadUUID", "download UUID not found")

        # Handle failed state explicitly
        st = d.status.value if isinstance(d.status, DownloadStatusEnum) else str(d.status)
        if st == DownloadStatusEnum.FAILED.value:
            # Retrieve error message if any
            err_msg = None
            try:
                derr = session.get(DownloadError, query.video_id)
                if derr:
                    err_msg = derr.errorMessage
            except Exception:
                pass
            return error_response("downloadFailed", err_msg or "The download failed")

        # If ready now, redirect to public HTTPS URL
        resp = _redirect_if_ready(d)
        if resp is not None:
            return resp

        # Not ready: sleep, then re-check once
        try:
            time.sleep(RETRY_SLEEP_SECONDS)
        except Exception:
            pass

        # Refresh and re-check
        try:
            session.expire(d)
        except Exception:
            pass
        d2 = session.get(Download, query.video_id)
        if d2:
            resp2 = _redirect_if_ready(d2)
            if resp2 is not None:
                return resp2

        # Still not ready → redirect back to this endpoint with incremented retry counter
        next_retry = current_retry + 1
        next_url = f"/redirectToHTTPSVideoDownloadPublicURL?video_id={quote(query.video_id)}&retryCounter={next_retry}"
        return redirect(next_url, code=302)
    except Exception as e:
        return error_response("redirectError", f"Failed to process redirect: {e}")
    finally:
        session.close()
