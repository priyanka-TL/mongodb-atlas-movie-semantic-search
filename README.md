# Movie Recommendations Script

This project provides a Python script for movie recommendations using MongoDB Atlas and Hugging Face embeddings.

## Requirements

- Python 3.12+
- All dependencies are installed in a local virtual environment (`.venv`).
- MongoDB Atlas account
- Hugging Face account

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

4. **Set up MongoDB Atlas sample data (`sample_mflix`):**

   - Create/login to MongoDB Atlas Cloud.
   - Open your cluster and click **Browse Collections**.
   - Choose **Load Sample Dataset** and load **`sample_mflix`**.

5. **Create MongoDB Atlas Search index on `sample_mflix.movies`:**

    - Go to your cluster in Atlas.
    - Open **Atlas Search** (or **Search and Vector Search**).
    - Create an index named `plotSemanticSearch` for database **`sample_mflix`** and collection **`movies`**.
    - Use the JSON editor and paste this definition:

    ```json
    {
       "mappings": {
          "dynamic": true,
          "fields": {
             "plot_embedding_hf": {
                "dimensions": 384,
                "similarity": "dotProduct",
                "type": "knnVector"
             }
          }
       }
    }
    ```

6. **Set up Hugging Face API access:**

   - Create/login to your Hugging Face account.
   - Generate an access token from **Settings > Access Tokens**.
   - Use this model name in the project:
     - `sentence-transformers/all-MiniLM-L6-v2`

7. **Create `.env` file in project root:**

   ```env
   MONGO_URI=mongodb+srv://<username>:<password>@<your-cluster-url>/?retryWrites=true&w=majority&appName=<your-app-name>
   HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
   ```

   Replace `<your-cluster-url>` with your MongoDB Atlas cloud cluster URL.

8. **Run the script:**

   ```bash
   .venv/bin/python movie_recs.py
   ```

## Notes
- Do not use system Python for installing packages; always use the `.venv`.
- If you need more dependencies, add them to `requirements.txt` and reinstall.
- Ensure `.env` is present before running the script.

## License
MIT
