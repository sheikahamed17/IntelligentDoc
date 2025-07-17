import google.generativeai as genai
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid
import os

# --- IMPORTANT ---
# Set your Google API key here.
# It is strongly recommended to use environment variables for security.
# For example: os.environ["GOOGLE_API_KEY"]
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Ensure the API key is set
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it before running the application.")
genai.configure(api_key=GOOGLE_API_KEY)


class GoogleRAGPipeline:
    def __init__(self, db_path="chroma_db", collection_name="rag_collection_google"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = "models/embedding-001"
        self.generation_model = genai.GenerativeModel('gemma-3-27b-it')


    def _process_and_chunk(self, text, source):
        chunks = self.text_splitter.split_text(text)
        return chunks

    def add_document(self, text, source_name):
        """Adds a document's text to the ChromaDB collection using Google's embedding model."""
        chunks = self._process_and_chunk(text, source_name)
        
        for i, chunk in enumerate(chunks):
            embedding_result = genai.embed_content(model=self.embedding_model, content=chunk)
            embedding = embedding_result['embedding']
            self.collection.add(
                ids=[f"{source_name}-{uuid.uuid4()}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": source_name}]
            )
        return f"Successfully added {len(chunks)} chunks from {source_name} using Google AI."

    def query_rag(self, query):
        """Queries the RAG pipeline to get an answer using Google's models."""
        embedding_result = genai.embed_content(model=self.embedding_model, content=query)
        query_embedding = embedding_result['embedding']
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5 
        )
        
        context = "\n".join([doc for sublist in results["documents"] for doc in sublist])
        
        prompt = f"""Use the following context to answer the question. If the answer is not in the context, just say that you do not know.

Context: {context}

Question: {query}

Answer:"""
        
        response = self.generation_model.generate_content(prompt)
        return response.text

    def get_sources(self):
        """Retrieves a list of unique source names from the ChromaDB collection."""
        all_items = self.collection.get(include=['metadatas'])
        metadatas = all_items.get('metadatas', [])
        if not metadatas:
            return []
        
        sources = set(md['source'] for md in metadatas if md and 'source' in md)
        return list(sources)

    def clear_knowledge_base(self):
        """Clears all documents from the ChromaDB collection."""
        self.collection.delete()
        self.collection = self.client.get_or_create_collection(name=self.collection.name)
        return "Knowledge base cleared."


