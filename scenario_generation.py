import os
import argparse
import pandas as pd  # Add pandas import
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from datasets import Dataset
from utils import load_pdfs

def generate_scenarios(pdf_folder_path, chroma_persist_dir):
    """
    Generate climate policy scenarios for Sub-Saharan Africa using a RAG pipeline.
    
    Args:
        pdf_folder_path (str): Path to the folder containing subfolders with PDFs.
        chroma_persist_dir (str): Path to store the Chroma vector store.
    """
    # Load PDFs
    documents, pdf_count = load_pdfs(pdf_folder_path)
    print(f"Total number of PDFs ingested: {pdf_count}")

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f'Total number of document chunks: {len(chunks)}')

    # Setup Ollama and Chroma
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
        return_source_documents=False
    )

    # Load questions from input_prompts.csv
    prompts_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input_prompts.csv")
    if not os.path.exists(prompts_file):
        raise FileNotFoundError(f"The prompts file {prompts_file} does not exist.")
    prompts_df = pd.read_csv(prompts_file)
    if 'prompt' not in prompts_df.columns:
        raise ValueError("The input_prompts.csv file must contain a 'prompt' column.")
    questions = prompts_df['prompt'].tolist()

    # Generate responses
    answers = []
    contexts = []
    for question in questions:
        relevant_docs = retriever.get_relevant_documents(question)
        formatted_contexts = [doc.page_content for doc in relevant_docs]
        response = rag_chain.invoke(question)
        if isinstance(response, dict) and "result" in response:
            answers.append(response["result"])
        else:
            answers.append(response)
        contexts.append(formatted_contexts)

    # Prepare dataset
    data = {
        "Question": questions,
        "Answer": answers,
        "Contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    # Save the dataset to generated_responses.csv
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "generated_responses.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(data).to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate climate policy scenarios using LLMs.")
    parser.add_argument("--pdf-folder", default="data/cop_documents", help="Path to the folder containing PDFs.")
    parser.add_argument("--chroma-dir", default="chroma_db", help="Path to store the Chroma vector store.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    pdf_path = os.path.join(base_dir, args.pdf_folder)
    chroma_path = os.path.join(base_dir, args.chroma_dir)

    generate_scenarios(pdf_path, chroma_path)