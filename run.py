import os
from app import create_app
import logging

app = create_app()

# Sembunyikan error traceback dari werkzeug
# logging.getLogger('werkzeug').setLevel(logging.WARNING)

if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    app.run(debug=True, host=host, port=port, threaded=True)