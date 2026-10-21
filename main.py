import os
import sys

sys.path.insert(0, os.path.abspath("."))
from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)