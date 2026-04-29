import pymongo
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables from .env
load_dotenv()

# Get MongoDB connection string from environment
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
	raise ValueError("MONGO_URI not set in environment. Please set it in your .env file.")

client = pymongo.MongoClient(mongo_uri)
db = client.sample_mflix
collection = db.movies

print("Connected to MongoDB successfully!")
print("Sample movie document:", collection.find_one())

# Get Hugging Face token from environment
hf_token = os.getenv("HUGGING_FACE_TOKEN")
if not hf_token:
    raise ValueError("HUGGING_FACE_TOKEN not set in environment. Please set it in your .env file.")

# Create Hugging Face inference client
hf_client = InferenceClient(token=hf_token)

def generate_embedding(text: str) -> list[float]:
    embedding = hf_client.feature_extraction(
        text,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
        # print("Embedding converted to list using tolist() method.", embedding)

    if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
        return embedding[0]

    if isinstance(embedding, list):
        return embedding

    raise ValueError("Unexpected embedding response format from Hugging Face API")
    
generate_embedding('freeCodeCamp is awesome!')

for doc in collection.find({'plot': {'$exists': True}}).limit(50):
    doc['plot_embedding_hf'] = generate_embedding(doc['plot'])
    collection.replace_one({'_id': doc['_id']}, doc)

query = "Imaginary characters from outer space at war"

results = collection.aggregate([
    {
        "$vectorSearch": {
            "queryVector": generate_embedding(query),
            "path": "plot_embedding_hf",
            "numCandidates": 100,
            "limit": 4,
            "index": "plotSemanticSearch"
        }
    }])

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')