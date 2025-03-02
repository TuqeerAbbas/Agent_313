import chromadb
import json
from langchain_openai import OpenAIEmbeddings
from config.config import Config

def load_data_to_chroma(category: str, data_path: str):
    """
    Load data into ChromaDB for a specific category
    
    Args:
        category: 'product', 'recipe', 'parenting_advice', or 'health_concern'
        data_path: Path to the JSON data file
    """
    # Initialize ChromaDB
    config = Config()
    client = chromadb.PersistentClient(path=config.CHROMA_BASE_PATH)
    collection = client.get_or_create_collection(
        name=f"{category}_collection",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Load data
    with open(data_path, "r") as f:
        items = json.load(f)
    
    # Prepare data for ChromaDB
    documents = []
    metadatas = []
    ids = []
    
    for i, item in enumerate(items):
        # Convert item to string for embedding
        item_text = json.dumps(item)
        documents.append(item_text)
        
        # Prepare metadata
        metadata = {
            "age_group": item.get("age_group"),
            "category": category
        }
        metadatas.append(metadata)
        
        # Create unique ID
        ids.append(f"{category}_{i}")
    
    # Get embeddings
    embeddings_list = [embeddings.embed_query(doc) for doc in documents]
    
    # Add to ChromaDB
    collection.add(
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"Added {len(documents)} {category} items to ChromaDB")

if __name__ == "__main__":
    # Load all datasets
    datasets = {
        "product": "/Users/ranatuqeerabbas/Desktop/Dataset/nestle baby&me/output/product.json",
        "recipe": "/Users/ranatuqeerabbas/Desktop/Dataset/nestle baby&me/output/recipe.json",
        "parenting_advice": "/Users/ranatuqeerabbas/Desktop/Dataset/nestle baby&me/output/parenting_advice.json",
        "health_concern": "/Users/ranatuqeerabbas/Desktop/Dataset/nestle baby&me/output/health_concern.json"
    }
    
    for category, data_path in datasets.items():
        print(f"\nProcessing {category} dataset...")
        load_data_to_chroma(category, data_path)