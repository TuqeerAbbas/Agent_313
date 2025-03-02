import chromadb
import json
from langchain_openai import OpenAIEmbeddings
from config import Config

def load_products_to_chroma():
    # Initialize ChromaDB
    config = Config()
    client = chromadb.PersistentClient(path=config.CHROMA_BASE_PATH)
    collection = client.get_or_create_collection(
        name="product_collection",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Load your product data
    with open("/Users/ranatuqeerabbas/Desktop/Dataset/products.json", "r") as f:
        products = json.load(f)
    
    # Prepare data for ChromaDB
    documents = []  # Product content
    metadatas = []  # Metadata like age_group, category
    ids = []        # Unique IDs
    
    for i, product in enumerate(products):
        # Convert product to string for embedding
        product_text = json.dumps(product)
        documents.append(product_text)
        
        # Prepare metadata
        metadata = {
            "age_group": product.get("age_group"),
            "category": "product"
        }
        metadatas.append(metadata)
        
        # Create unique ID
        ids.append(f"product_{i}")
        
    # Get embeddings
    embeddings_list = [embeddings.embed_query(doc) for doc in documents]
    
    # Add to ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Added {len(documents)} products to ChromaDB")

if __name__ == "__main__":
    load_products_to_chroma()