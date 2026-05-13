from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

def _initialize_vector_store():
    """
    Initializes a local Chroma vector store using HuggingFaceEmbeddings.
    """
    sample_text = """
    First-Aid Guidelines & Medical Protocols:
    - Mild Fever: Rest, hydrate, and take over-the-counter acetaminophen or ibuprofen.
    - Severe Chest Pain: Immediately route to the emergency room or call 911. Possible heart attack.
    - Minor Burns: Cool the burn under running water for 10-15 minutes, apply aloe vera, and cover loosely.
    - Choking: Perform the Heimlich maneuver immediately if the person cannot breathe or speak.
    - Allergic Reaction (Anaphylaxis): Use an epinephrine auto-injector (EpiPen) and seek immediate emergency care.
    """
    
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20, separator="\n")
    docs = [Document(page_content=line.strip()) for line in sample_text.strip().split("\n") if line.strip()]
    split_docs = text_splitter.split_documents(docs)
    
    # Initialize HuggingFace local embeddings
    hf_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        embedding=hf_embeddings,
        collection_name="medical_guidelines_local"
    )
    return vectorstore

# Initialize on load
_vectorstore = _initialize_vector_store()
_retriever = _vectorstore.as_retriever()

@tool
def retrieve_medical_guidelines(query: str) -> str:
    """Retrieve standard medical guidelines, first-aid, and protocols based on a query."""
    docs = _retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])
