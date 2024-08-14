#WITH RAG FOR 7.8KB FILE



import re
from transformers import AutoTokenizer, AutoModel
import torch
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai
import os
from sklearn.metrics.pairwise import cosine_similarity
from odf.opendocument import load
from odf.text import P

huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="hugging_face_api_key",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Set up the persistent directory for ChromaDB
persist_directory = "./chroma_db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

chroma_client = chromadb.PersistentClient(path=persist_directory)
collection = chroma_client.get_or_create_collection(name="my_collection")

GOOGLE_API_KEY = "google_ai_studio_api_key"
genai.configure(api_key=GOOGLE_API_KEY)

file_path = 'Path_to_actual_file'  
def create_chunks(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += (chunk_size - overlap)
    return chunks

def process_document(file_path):
    print("Processing document...")
    doc = load(file_path)
    content = []
    for paragraph in doc.getElementsByType(P):
        paragraph_text = ''.join(node.data for node in paragraph.childNodes if node.nodeType == 3)
        content.append(paragraph_text.strip())
    document_text = ' '.join(content)
    
    chunks = create_chunks(document_text)
    print(f"Number of chunks created: {len(chunks)}")
    embeddings = huggingface_ef(chunks)
    print("Generated embeddings for all chunks.")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"text": chunk} for chunk in chunks]
    collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
    print("Chunks and embeddings added to Chroma DB.")

def query_collection(query_text, n_results=10):
    query_embedding = huggingface_ef([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["metadatas", "embeddings"]
    )
    relevant_chunks = [metadata['text'] for metadata in results['metadatas'][0]]
    chunk_embeddings = results['embeddings'][0]
    similarity_scores = cosine_similarity([query_embedding], chunk_embeddings)[0]
    return relevant_chunks, similarity_scores.tolist()

def generate_answer(query, contexts, similarity_scores):
    if not contexts or max(similarity_scores) <= 0.1:
        return "Information not found in the document."
    context = "\n\n".join(contexts)
    prompt = f"""Based on the following context, answer the question. If the information is not explicitly mentioned, make reasonable inferences and
clearly state that you are doing so. If the answer cannot be derived from the given context,
state that the information is not found in the document.

Context:
{context}

Question: {query}
Answer:"""

    try:
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=250
        ))
        if response.parts:
            return response.text
        else:
            return "Error: The model did not generate any content."
    except Exception as e:
        return f"Error: Unable to generate response. {str(e)}"

def expand_query(query):
    prompt = f"Given the question: '{query}', generate 2-3 related questions that might help provide a more comprehensive answer. Format the output as a comma-separated list."
    try:
        response = genai.GenerativeModel('gemini-pro').generate_content(prompt)
        if response.parts:
            expanded_queries = [query.strip() for query in response.text.split(',')]
            return [query] + expanded_queries
        else:
            return [query]
    except Exception as e:
        print(f"Error in query expansion: {str(e)}")
        return [query]

def main():
    if collection.count() == 0:
        process_document(file_path)
    else:
        print("Document already processed. Skipping embedding creation.")

    while True:
        query = input("\nEnter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        expanded_queries = expand_query(query)
        all_chunks = []
        all_scores = []
        for eq in expanded_queries:
            chunks, scores = query_collection(eq, n_results=5)
            all_chunks.extend(chunks)
            all_scores.extend(scores)
        answer = generate_answer(query, all_chunks, all_scores)
        print("\nGenerated Answer:")
        print(answer)

if __name__ == "__main__":
    main()