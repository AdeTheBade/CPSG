import os
import argparse
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextUtilization, AnswerRelevancy
from ragas.run_config import RunConfig
import csv

def evaluate_scenarios(input_file, output_dir):
    """
    Evaluate pre-generated climate policy scenarios using the RAGAS framework.
    
    Args:
        input_file (str): Path to the input CSV file with pre-generated responses.
        output_dir (str): Path to save evaluation results.
    """
    # Load pre-generated responses from CSV
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")
    df = pd.read_csv(input_file)
    if not {'Question', 'Answer', 'Contexts'}.issubset(df.columns):
        raise ValueError("The input CSV must contain 'Question', 'Answer', and 'Contexts' columns.")
    
    questions = df['Question'].tolist()
    answers = df['Answer'].tolist()
    contexts = [context.split(" | ") if isinstance(context, str) else [] for context in df['Contexts']]

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    dataset = Dataset.from_dict(data)

    # Evaluate responses
    llm_mistral = OllamaLLM(
        model="mistral:7b",
        verbose=False,
        timeout=600,
        num_ctx=5000,
        disable_streaming=False,
        temperature=0
    )
    embeddings_nomic = OllamaEmbeddings(model="nomic-embed-text:latest")

    evaluation_results = []
    for i, sample in enumerate(dataset):
        print(f"Evaluating sample {i+1}/{len(dataset)}...")
        single_sample_dataset = Dataset.from_dict({
            "question": [sample["question"]],
            "answer": [sample["answer"]],
            "contexts": [sample["contexts"]]
        })
        result_mistral = evaluate(
            dataset=single_sample_dataset,
            metrics=[Faithfulness(), ContextUtilization(), AnswerRelevancy()],
            llm=llm_mistral,
            embeddings=embeddings_nomic,
            run_config=RunConfig(max_workers=16, timeout=600, max_retries=5, max_wait=20, log_tenacity=True)
        )
        scores = {metric: result_mistral[metric][0] if result_mistral[metric] else "NaN" for metric in result_mistral._scores_dict.keys()}
        evaluation_results.append({
            "Question": sample["question"],
            "Answer": sample["answer"],
            "Contexts": " | ".join(sample["contexts"]),
            "Faithfulness": scores.get("faithfulness", "NaN"),
            "ContextUtilization": scores.get("context_utilization", "NaN"),
            "AnswerRelevancy": scores.get("answer_relevancy", "NaN")
        })

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_result_mistral.csv")
    header = ["Question", "Answer", "Contexts", "Faithfulness", "ContextUtilization", "AnswerRelevancy"]
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(evaluation_results)
    print(f"All evaluation results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pre-generated climate policy scenarios using mistral-7B.")
    parser.add_argument("--input-file", default="data/generated_responses.csv", help="Path to the input CSV file with pre-generated responses.")
    parser.add_argument("--output-dir", default="results", help="Path to save evaluation results.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    input_path = os.path.join(base_dir, args.input_file)
    output_path = os.path.join(base_dir, args.output_dir)

    evaluate_scenarios(input_path, output_path)