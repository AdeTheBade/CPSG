import os
import argparse
import json
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from utils import load_pdfs

def extract_themes(pdf_folder_path, chroma_persist_dir, queries):
    """
    Extract key climate policy themes from documents using a RAG pipeline for multiple queries.
    
    Args:
        pdf_folder_path (str): Path to the folder containing subfolders with PDFs.
        chroma_persist_dir (str): Path to store the Chroma vector store.
        queries (list): List of queries to extract themes for.
    """
    # Load PDFs and setup Chroma
    documents, pdf_count = load_pdfs(pdf_folder_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f'Total number of document chunks: {len(chunks)}')

    ollama = OllamaLLM(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
        temperature=0,
    )
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text:latest")

    if not os.path.exists(chroma_persist_dir):
        os.makedirs(chroma_persist_dir)
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="rag-chroma",
        embedding=embeddings_model,
        persist_directory=chroma_persist_dir
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    rag_chain = RetrievalQA.from_chain_type(
        llm=ollama,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # Extract themes for each query and store results
    results = []
    for query in queries:
        print(f"\nProcessing query: {query}")
        response = rag_chain.invoke(query)
        
        # Extract themes and source documents
        extracted_themes = response["result"]
        source_docs = [
            {
                "source": doc.metadata['source'],
                "page": doc.metadata['page'],
                "content": doc.page_content[:200] + "..."
            }
            for doc in response["source_documents"]
        ]

        # Print results
        print("Extracted Themes:")
        print(extracted_themes)
        print("\nSource Documents:")
        for doc in source_docs:
            print(f"- {doc['source']} (Page {doc['page']}): {doc['content']}")

        # Append to results
        results.append({
            "query": query,
            "extracted_themes": extracted_themes,
            "source_documents": source_docs
        })

    # Save results to JSON file
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "extracted_themes.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract climate policy themes using RAG.")
    parser.add_argument("--pdf-folder", default="data/cop_documents", help="Path to the folder containing PDFs.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Path to store the Chroma vector store.")
    parser.add_argument("--queries", nargs='+', default=["Key themes within all the documents, focusing on energy policy, climate adaptation, and mitigation strategies.", "All Climate policies themes mentioned in the documents"], help="List of queries to extract themes for.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    pdf_path = os.path.join(base_dir, args.pdf_folder)
    chroma_path = os.path.join(base_dir, args.chroma_dir)

    extract_themes(pdf_path, chroma_path, args.queries)