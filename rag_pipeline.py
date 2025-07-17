import ollama
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

class RAGPipeline:
    def __init__(self, db_path="chroma_db", collection_name="rag_collection"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def _process_and_chunk(self, text, source):
        chunks = self.text_splitter.split_text(text)
        return chunks

    def add_document(self, text, source_name):
        """Adds a document's text to the ChromaDB collection."""
        chunks = self._process_and_chunk(text, source_name)
        
        for i, chunk in enumerate(chunks):
            embedding = ollama.embeddings(model="nomic-embed-text", prompt=chunk)["embedding"]
            self.collection.add(
                ids=[f"{source_name}-{uuid.uuid4()}"],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": source_name}]
            )
        return f"Successfully added {len(chunks)} chunks from {source_name}."

    def query_rag(self, query):
        """Queries the RAG pipeline to get an answer."""
        query_embedding = ollama.embeddings(model="nomic-embed-text", prompt=query)["embedding"]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5 
        )
        
        context = "\n".join(results["documents"][0])
        
        sources = []
        if results.get("metadatas"):
            sources = sorted(list(set(md['source'] for md_list in results["metadatas"] for md in md_list if md and 'source' in md)))

        prompt = f"""Use the following context to answer the question. If the answer is not in the context, just say that you do not know.

Context: {context}

Question: {query}

Answer:"""
        
        response = ollama.generate(
            model="gemma3:4b",
            prompt=prompt,
        )
        return response["response"], sources

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

