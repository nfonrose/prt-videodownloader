import os
import uuid
import subprocess
import shlex
import threading
from enum import Enum
from typing import List, Optional
from flask_openapi3 import OpenAPI, Info, Tag
from flask import jsonify
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
)
from sqlalchemy.orm import sessionmaker, declarative_base

# Configuration for SQLAlchemy with SQLite
PATHPREFIX_FOR_DEVLOCAL_ENV = "/Users/teevity/Dev/misc/1.prtVideoDownloader/"
DEFAULT_DB_PATH = (PATHPREFIX_FOR_DEVLOCAL_ENV + "/opt/prt/prt-videodownloader/db/prt-videodownloader.sqlite")
DB_ENV_VAR = "PRT_VIDEODOWNLOADER_SQLITEFILEPATH"

# Video data path configuration
DEFAULT_DATA_PATH = PATHPREFIX_FOR_DEVLOCAL_ENV + "/opt/prt/prt-videodownloader/data/"
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
    createdAt = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    updatedAt = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


# Create tables if they don't exist
Base.metadata.create_all(bind=engine)

# Initialize Flask OpenAPI app
info = Info(title="PRT VideoDownloader API", version="0.1.0")
app = OpenAPI(__name__, info=info)

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


def error_response(error_code: str, error_message: str):
    return jsonify({"errorCode": error_code, "errorMessage": error_message}), 418


@app.get("/hello", tags=[hello_tag], summary="Hello World")
def hello_world():
    return jsonify({"message": "hello world"})


# Health endpoint for basic checks (not required but handy)
@app.get("/healthz", summary="Health check")
def healthz():
    return jsonify({"status": "ok"})


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
        "bv*+ba/best",
        "-o",
        os.path.join(data_path, "%(title)s-%(id)s.%(ext)s"),
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
        # Async log subprocess output
        def _log_stream(stream, is_err=False):
            logger = app.logger.error if is_err else app.logger.info
            for line in iter(stream.readline, ''):
                line = line.rstrip('\n')
                if line:
                    logger("DOWNLOAD[%s] %s: %s", download_uuid, "stderr" if is_err else "stdout", line)
            stream.close()
        threading.Thread(target=_log_stream, args=(proc.stdout, False), daemon=True).start()
        threading.Thread(target=_log_stream, args=(proc.stderr, True), daemon=True).start()
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


if __name__ == "__main__":
    # Allow external connections (not limited to 127.0.0.1)
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)