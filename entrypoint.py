import os
import logging
from app import app

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    port = int(os.getenv("PORT", "9380"))
    app.json.compact = False                   # Make the OpenAPI generated pretty-printed
    app.run(host="0.0.0.0", port=port)
