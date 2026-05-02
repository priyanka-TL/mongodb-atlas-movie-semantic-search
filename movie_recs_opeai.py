import pymongo
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not set in environment. Please set it in your .env file.")

# Get MongoDB connection string from environment
mongo_uri = os.getenv("MONGO_URI")
if not mongo_uri:
	raise ValueError("MONGO_URI not set in environment. Please set it in your .env file.")

client = pymongo.MongoClient(mongo_uri)
db = client.sample_mflix
collection = db.embedded_movies

def generate_embedding(text: str) -> list[float]:

    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=text
    )
    return response['data'][0]['embedding']

query = "imaginary characters from outer space at war"

results = collection.aggregate([
  {"$vectorSearch": {
    "queryVector": generate_embedding(query),
    "path": "plot_embedding",
    "numCandidates": 100,
    "limit": 4,
    "index": "PlotSemanticSearch",
      }}
])

for document in results:
    print(f'Movie Name: {document["title"]},\nMovie Plot: {document["plot"]}\n')