# Climate-Policy-Scenario-Generation

This is the code that accompanies the paper titled **"AI-Driven Climate Policy Scenario Generation for Sub-Saharan Africa"** by *Rafiu Adekoya Badekale* and *Dr. Adewale Akinfaderin*. The project explores the use of large language models (LLMs) for generating and evaluating climate policy scenarios tailored to Sub-Saharan Africa (SSA). This research was fully funded by Hamoye Foundation, a 501(c)(3) organization.

## Folder Structure
- `data/`: Contains input prompts and the UN COP documents.
- `notebooks/`: Jupyter notebooks used for theme extraction, scenario generation, evaluation and analysis.
- `results/`: Generated outputs and compiled evaluation metrics results.

## Requirements
Install all required Python packages:

```bash
pip install -r requirements.txt

## Reproduce
1. Run `notebooks/ThemeExtraction_RAG.ipynb` to extract themes from the UN COP documents.
2. Run `notebooks/Scenario_Generation.ipynb` to generate policy scenarios.
3. Run `notebooks/Automated Evaluation.ipynb` for automated evaluation.
5. Run `notebooks/Correlation.ipynb` for correlation analysis.
4. Run `notebooks/follow_up_generated_response.ipynb` to generate follow-up policy details.

## 📂 Dataset Access

All 91 UN COP policy documents used as source material are included in this repository under: "data/cop_documents"

