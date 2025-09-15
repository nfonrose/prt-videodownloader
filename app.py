import os
from enum import Enum
from flask_openapi3 import OpenAPI, Info, Tag
from flask import jsonify
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

def get_db_path() -> str:
    path = os.getenv(DB_ENV_VAR, DEFAULT_DB_PATH)
    # Ensure directory exists
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return path

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

@app.get("/hello", tags=[hello_tag], summary="Hello World")
def hello_world():
    return jsonify({"message": "hello world"})

# Health endpoint for basic checks (not required but handy)
@app.get("/healthz", summary="Health check")
def healthz():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    # Allow external connections (not limited to 127.0.0.1)
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)