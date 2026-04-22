
from app import create_app
import logging

app = create_app()

# Sembunyikan error traceback dari werkzeug
# logging.getLogger('werkzeug').setLevel(logging.WARNING)

if __name__ == "__main__":
    app.run(debug=True)