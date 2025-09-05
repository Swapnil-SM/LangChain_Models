# to use model from local system
# from langchain_huggingface import HuggingFaceEmbeddings

# embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# documents = [
#     "Delhi is the capital of India",
#     "Kolkata is the capital of West Bengal",
#     "Paris is the capital of France"
# ]

# vector = embedding.embed_documents(documents)

# print(str(vector))


# code to using models from HF Inference API
from langchain_huggingface import HuggingFaceInferenceEmbeddings
from dotenv import load_dotenv
import os

# Load HF token from .env
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Use the Inference API for embeddings
embedding = HuggingFaceInferenceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    api_key=hf_token
)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(vector)


