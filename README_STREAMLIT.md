# Movie Semantic Search Interface (Streamlit)

This README is for the new Streamlit interface app.

## What was added

- New file: `movie_search_streamlit.py`
- Existing file `movie_recs.py` is unchanged.
- UI lets users enter a query and returns semantically similar movies from MongoDB Atlas.

## Prerequisites

- Python 3.12+
- MongoDB Atlas cluster
- Hugging Face account and API token
- `sample_mflix` dataset loaded in Atlas

## Atlas setup

1. Load sample data:
   - Atlas -> **Browse Collections** -> **Load Sample Dataset** -> `sample_mflix`

2. Create vector search index on `sample_mflix.movies` named `plotSemanticSearch`:

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

## Environment variables

Copy `.env.sample` to `.env` and fill values:

```env
MONGO_URI=mongodb+srv://<username>:<password>@<cluster-url>/?retryWrites=true&w=majority&appName=<app-name>
HUGGING_FACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
HUGGING_FACE_MODEL=sentence-transformers/all-MiniLM-L6-v2
ATLAS_VECTOR_INDEX=plotSemanticSearch
```

## Install dependencies

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit interface

```bash
streamlit run movie_search_streamlit.py
```

Then open the local URL printed in terminal (usually `http://localhost:8501`).

## End-to-end test checklist

1. App starts without import/config errors.
2. Enter query such as: `imaginary characters from outer space at war`.
3. Click **Search**.
4. Results list movie title, plot, year, and similarity score.

## Notes

- The app searches the `plot_embedding_hf` vector field.
- If no results appear, verify:
  - documents contain `plot_embedding_hf`
  - index name matches `ATLAS_VECTOR_INDEX`
  - token and Mongo URI are valid
