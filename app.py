import os
from flask_openapi3 import OpenAPI, Info, Tag
from flask import jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configuration for SQLAlchemy with SQLite
DEFAULT_DB_PATH = "/opt/prt/prt-videodownloader/db/prt-videodownloader.sqlite"
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