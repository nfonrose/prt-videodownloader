import os
import logging
from app import app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", "9080"))
    app.run(host="0.0.0.0", port=port)
