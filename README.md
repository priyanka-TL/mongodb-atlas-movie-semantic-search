# Movie Recommendations Script

This project provides a Python script for movie recommendations using MongoDB.

## Requirements

- Python 3.12+
- All dependencies are installed in a local virtual environment (`.venv`).
- Main dependency: `pymongo`

## Setup Instructions

1. **Clone or download this repository.**

2. **Create and activate the virtual environment (if not already present):**

   The environment is auto-configured, but if you need to recreate it:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   .venv/bin/pip install -r requirements.txt
   ```
   Or, if you only need `pymongo`:
   ```bash
   .venv/bin/pip install pymongo
   ```

4. **Run the script:**

   ```bash
   .venv/bin/python movie_recs.py
   ```

## Notes
- Do not use system Python for installing packages; always use the `.venv`.
- If you need more dependencies, add them to `requirements.txt` and reinstall.

## License
MIT
