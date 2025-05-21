# Climate-Policy-Scenario-Generation

This is the code that accompanies the paper titled **"AI-Driven Climate Policy Scenario Generation for Sub-Saharan Africa"** by *Rafiu Adekoya Badekale* and *Dr. Adewale Akinfaderin*. The project explores the use of large language models (LLMs) for generating and evaluating climate policy scenarios tailored to Sub-Saharan Africa (SSA). This research was fully funded by Hamoye Foundation, a 501(c)(3) organization.

## Repository Structure
- `data/`: Contains input prompts, the UN COP documents and generated outputs.
- `results/`: Compiled evaluation metrics and generated evaluation outputs.
- `theme_extraction.py`: Extracts key climate policy themes from the documents.
- `scenario_generation.py`: Generates climate policy scenarios for Sub-Saharan Africa using a RAG pipeline.
- `automated_evaluation.py`: Evaluates the generated scenarios using the RAGAS framework.
- `follow_up_generated_response.py`: Generates follow-up responses (measurable outcomes, key drivers, milestones) for the scenarios.
- `correlation.py`: Computes mean scores, standard deviations, and Spearman rank correlations for evaluation metrics.
- `utils.py`: Utility functions for loading PDFs.
- `requirements.txt`: List of Python dependencies.

## Dataset Access

All 94 UN COP policy documents used as source material are included in this repository under: "data/cop_documents"

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/AdeTheBade/CPSG.git

cd CPSG
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv

source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Install and Run Ollama
Install Ollama: Follow the instructions at Ollama's official website (https://ollama.com)

Pull the required models:
```bash
ollama pull llama3.2:3b

ollama pull llama3:latest

ollama pull mistral:7b

ollama pull gemma2:2b

ollama pull nomic-embed-text:latest
```
Start the Ollama server:
```bash
ollama serve
```
### 5. Prepare the Data
The required data is already included in the repository:

PDF Documents: Located in data/cop_documents/.

Input Prompts: Located in data/input_prompts.csv.

The follow_up_generated_response.py script expects a generated_responses.csv file, which should be the output of scenario_generation.py

The correlation.py script expects a Compiled evaluation result.csv file, which should contain evaluation metrics (e.g., from automated_evaluation.py).

### 6. Run the Scripts
Each script can be run with default paths or custom paths using command-line arguments. For example, to extract themes from the UN COP documents, you can run the command below.
```bash
python theme_extraction.py
```
Or with custom paths:
```bash
python theme_extraction.py --queries "Query 1" "Query 2" "Query 3"
```
### Project Workflow
Theme Extraction: Run theme_extraction.py to extract key themes from the documents.

Scenario Generation: Run scenario_generation.py to generate climate policy scenarios and save them to a CSV file.

Automated Evaluation: Run automated_evaluation_gemma2.py or automated_evaluation_mistral.py to evaluate the scenarios using RAGAS metrics and save the results.

Correlation Analysis: Run correlation.py to compute correlations between evaluation metrics.

Follow-Up Responses: Run follow_up_generated_response.py to generate follow-up responses for the scenarios.