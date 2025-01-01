# import pinecone
# import re
# from typing import List

# # Initialize the Pinecone client with the API key
# try:
#     pinecone.init(api_key=pinecone_api_key)
#     print("Pinecone client initialized successfully.")
# except Exception as e:
#     print(f"Error initializing Pinecone client: {e}")

# # Create a new index in Pinecone
# index_name = "qa_bot_index"
# try:
#     pinecone.create_index(name=index_name, dimension=512, metric="cosine")
#     print(f"Index '{index_name}' created successfully.")
# except Exception as e:
#     print(f"Error creating index '{index_name}': {e}")

# # Connect to the created index
# try:
#     index = pinecone.Index(index_name)
#     print(f"Connected to index '{index_name}' successfully.")
# except Exception as e:
#     print(f"Error connecting to index '{index_name}': {e}")

# # Print the index description to verify the setup
# try:
#     print(index.describe_index())
# except Exception as e:
#     print(f"Error describing index '{index_name}': {e}")

# # Define Helper Functions
# def preprocess_text(text: str) -> str:
#     """
#     Preprocess the input text by removing special characters and extra spaces.
#     """
#     text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
#     text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
#     return text.strip()

# # Define your OpenAI API key
# openai_api_key = "sk-...QqAA"

# # Configure the OpenAI client with the API key
# openai.api_key = openai_api_key
# # Test the OpenAI API setup by making a simple request
# response = openai.Completion.create(
#     engine="davinci",
#     prompt="Hello, world!",
#     max_tokens=5
# )

# # Print the response to verify the setup
# print(response.choices[0].text.strip())
# # Set Up Pinecone DB

# # Define your Pinecone API key
# pinecone_api_key = "pcsk_3mvD2b_***********"
# # Initialize the Pinecone client with the API key
# pinecone.init(api_key=pinecone_api_key)

# # Create a new index in Pinecone
# index_name = "qa_bot_index"
# pinecone.create_index(name=index_name, dimension=512, metric="cosine")

# # Connect to the created index
# index = pinecone.Index(index_name)

# # Print the index description to verify the setup
# print(index.describe_index())
# # Define Helper Functions

# import re
# from typing import List

# def preprocess_text(text: str) -> str:
#     """
#     Preprocess the input text by removing special characters and extra spaces.
#     """
#     # Remove special characters and extra spaces
#     text = re.sub(r'\s+', ' ', text)
#     text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
#     return text.strip()

# def generate_embeddings(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
#     """
#     Generate embeddings for a list of texts using the OpenAI API.
#     """
#     # Generate embeddings using OpenAI API
#     response = openai.Embedding.create(input=texts, model=model)
#     embeddings = [embedding['embedding'] for embedding in response['data']]
#     return embeddings

# def upsert_embeddings(index, texts: List[str], ids: List[str]):
#     """
#     Upsert embeddings into the Pinecone index.
#     """
#     # Preprocess texts
#     preprocessed_texts = [preprocess_text(text) for text in texts]

#     # Generate embeddings
#     embeddings = generate_embeddings(preprocessed_texts)

#     # Create a list of (id, embedding) tuples
#     vectors = [(id, embedding) for id, embedding in zip(ids, embeddings)]
#     # Upsert vectors into the Pinecone index
#     index.upsert(vectors)

# def query_index(index, query_text: str, top_k: int = 5) -> List[str]:
#     """
#     Query the Pinecone index with a text and return the top_k most similar texts.
#     """
# # Preprocess the query text
#     preprocessed_query = preprocess_text(query_text)
# # Generate embedding for the query text
#     query_embedding = generate_embeddings([preprocessed_query])[0]
# # Query the Pinecone index
#     query_response = index.query(queries=[query_embedding], top_k=top_k)
# # Extract the IDs of the top_k most similar texts
#     top_ids = [match['id'] for match in query_response['matches']]
#     return top_ids
# # Create Vector Database

# # Sample documents to be embedded and stored in Pinecone DB
# documents = [
#     "What are the business hours?",
#     "How can I contact customer support?",
#     "What is the return policy?",
#     "Where is the company located?",
#     "What services do you offer?"
# ]

# # Generate unique IDs for each document
# document_ids = [f"doc_{i}" for i in range(len(documents))]

# # Upsert the documents into the Pinecone index
# upsert_embeddings(index, documents, document_ids)

# # Verify that the documents have been upserted by querying the index
# query_result = query_index(index, "What are your business hours?")
# print("Query Result:", query_result)
# # Retrieve Relevant Documents

# # Define the query text
# query_text = "How do I contact support?"

# # Query the Pinecone index to retrieve relevant documents
# retrieved_document_ids = query_index(index, query_text)

# # Print the retrieved document IDs
# print("Retrieved Document IDs:", retrieved_document_ids)

# # Retrieve the actual documents based on the IDs
# retrieved_documents = [documents[int(id.split('_')[1])] for id in retrieved_document_ids]

# # Print the retrieved documents
# print("Retrieved Documents:", retrieved_documents)
# # Generate Answers Using OpenAI API

# # Define a function to generate answers using the OpenAI API
# def generate_answer(query: str, documents: List[str]) -> str:
#     """
#     Generate an answer using the OpenAI API based on the query and retrieved documents.
#     """
#     # Combine the query and documents into a single prompt
#     prompt = f"Query: {query}\n\nDocuments:\n" + "\n".join(documents) + "\n\nAnswer:"

#     # Generate the answer using the OpenAI API
#     response = openai.Completion.create(
#         engine="davinci",
#         prompt=prompt,
#         max_tokens=150,
#         n=1,
#         stop=None,
#         temperature=0.7
#     )

#     # Extract the generated answer from the response
#     answer = response.choices[0].text.strip()
#     return answer

# # Define the query text
# query_text = "How do I contact support?"

# # Generate the answer using the OpenAI API
# answer = generate_answer(query_text, retrieved_documents)

# # Print the generated answer
# print("Generated Answer:", answer)
# # Integrate Retrieval and Generation

# # Define a function to integrate retrieval and generation steps
# def rag_model(query: str) -> str:
#     """
#     Integrate retrieval and generation steps to create a complete RAG model for the QA bot.
#     """
#     # Retrieve relevant documents from Pinecone index
#     retrieved_document_ids = query_index(index, query)
#     retrieved_documents = [documents[int(id.split('_')[1])] for id in retrieved_document_ids]

#     # Generate an answer using the OpenAI API based on the query and retrieved documents
#     answer = generate_answer(query, retrieved_documents)
#     return answer

# # Define the query text
# query_text = "What is the return policy?"

# # Use the RAG model to get the answer
# rag_answer = rag_model(query_text)

# # Print the answer generated by the RAG model
# print("RAG Model Answer:", rag_answer)
import pinecone
import openai
import re
from typing import List

# Initialize Pinecone client with API key
pinecone_api_key = "pcsk_3mvD2b_***********"
pinecone.init(api_key=pinecone_api_key, environment="us-west1-gcp")

# Create a new index if it doesn't already exist
index_name = "qa_bot_index"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=512, metric="cosine")

# Connect to the created index
index = pinecone.GRPCIndex(index_name)

# Define OpenAI API key
openai.api_key = "sk-...QqAA"

# Define helper functions
def preprocess_text(text: str) -> str:
    """Preprocess the input text by removing special characters and extra spaces."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.strip()

def generate_embeddings(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    """Generate embeddings for a list of texts using OpenAI API."""
    try:
        response = openai.Embedding.create(input=texts, model=model)
        embeddings = [embedding['embedding'] for embedding in response['data']]
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        embeddings = []
    return embeddings

def upsert_embeddings(index, texts: List[str], ids: List[str]):
    """Upsert embeddings into the Pinecone index."""
    preprocessed_texts = [preprocess_text(text) for text in texts]
    embeddings = generate_embeddings(preprocessed_texts)
    vectors = [(id, embedding) for id, embedding in zip(ids, embeddings)]
    index.upsert(vectors)

def query_index(index, query_text: str, top_k: int = 5) -> List[str]:
    """Query the Pinecone index with a text and return the top_k most similar texts."""
    preprocessed_query = preprocess_text(query_text)
    query_embedding = generate_embeddings([preprocessed_query])[0]
    query_response = index.query(queries=[query_embedding], top_k=top_k)
    if not query_response or 'matches' not in query_response:
        return []
    return [match['id'] for match in query_response['matches']]

def generate_answer(query: str, documents: List[str]) -> str:
    """Generate an answer using the OpenAI API based on the query and retrieved documents."""
    combined_docs = "\n".join(documents[:5])  # Limit to the first 5 documents
    prompt = f"Query: {query}\n\nDocuments:\n{combined_docs}\n\nAnswer:"
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=150,
            n=1,
            stop=None,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating answer: {e}")
        return "An error occurred while generating the answer."

# Example documents and queries
documents = [
    "What are the business hours?",
    "How can I contact customer support?",
    "What is the return policy?",
    "Where is the company located?",
    "What services do you offer?"
]

document_ids = [f"doc_{i}" for i in range(len(documents))]
upsert_embeddings(index, documents, document_ids)

query_text = "What is the return policy?"
retrieved_document_ids = query_index(index, query_text)
retrieved_documents = [documents[int(id.split('_')[1])] for id in retrieved_document_ids]
rag_answer = generate_answer(query_text, retrieved_documents)

print("RAG Model Answer:", rag_answer)

