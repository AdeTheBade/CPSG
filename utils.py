import os
import re
from langchain.document_loaders import PyPDFLoader

def load_pdfs(pdf_folder_path):
    """
    Load PDFs from subfolders and return a list of documents.
    
    Args:
        pdf_folder_path (str): Path to the folder containing subfolders with PDFs.
        
    Returns:
        list: List of loaded documents.
    """
    # Check if the folder exists
    if not os.path.exists(pdf_folder_path):
        raise FileNotFoundError(f"The folder {pdf_folder_path} does not exist. Please provide a valid path.")

    subfolders = [folder for folder in os.listdir(pdf_folder_path) if os.path.isdir(os.path.join(pdf_folder_path, folder))]

    # Custom sorting function for subfolders (COP, CMP, CMA in descending order)
    def custom_sort_key(folder_name):
        match = re.match(r"(COP|CMP|CMA)(\d+)", folder_name)
        if match:
            conference_type, number = match.group(1), int(match.group(2))
            if conference_type == "COP":
                return (0, -number)
            elif conference_type == "CMP":
                return (1, -number)
            elif conference_type == "CMA":
                return (2, -number)
        return (3, folder_name)

    # Sort subfolders by type
    cop_subfolders = [folder for folder in subfolders if "COP" in folder]
    cmp_subfolders = [folder for folder in subfolders if "CMP" in folder]
    cma_subfolders = [folder for folder in subfolders if "CMA" in folder]

    cop_subfolders.sort(key=custom_sort_key)
    cmp_subfolders.sort(key=custom_sort_key)
    cma_subfolders.sort(key=custom_sort_key)

    final_order = []
    max_length = max(len(cop_subfolders), len(cmp_subfolders), len(cma_subfolders))
    for i in range(max_length):
        if i < len(cop_subfolders):
            final_order.append(cop_subfolders[i])
        if i < len(cmp_subfolders):
            final_order.append(cmp_subfolders[i])
        if i < len(cma_subfolders):
            final_order.append(cma_subfolders[i])

    documents = []
    pdf_count = 0

    for subfolder in final_order:
        subfolder_path = os.path.join(pdf_folder_path, subfolder)
        files_in_subfolder = sorted([file for file in os.listdir(subfolder_path) if file.endswith('.pdf')])
        print(f"Processing folder: {subfolder}")

        for file_name in files_in_subfolder:
            pdf_path = os.path.join(subfolder_path, file_name)
            loader = PyPDFLoader(pdf_path)
            doc = loader.load()
            documents.extend(doc)
            pdf_count += 1
            print(f'Loaded and processed: {file_name}')

    print(f"Total number of PDFs ingested: {pdf_count}")
    return documents, pdf_count