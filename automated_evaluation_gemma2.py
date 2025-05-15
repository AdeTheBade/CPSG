import os
import argparse
import pandas as pd
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, ContextUtilization, AnswerRelevancy
from ragas.run_config import RunConfig
import csv

def evaluate_scenarios_gemma2(input_file, output_dir):
    """
    Evaluate pre-generated climate policy scenarios using the RAGAS framework with the Gemma2 LLM.
    
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

    # Setup Gemma2 LLM and embeddings for evaluation
    llm_gemma2 = OllamaLLM(
        model="gemma2:2b",
        verbose=False,
        timeout=30,
        num_ctx=5000,
        disable_streaming=False,
        temperature=0
    )
    embeddings_nomic = OllamaEmbeddings(model="nomic-embed-text:latest")

    evaluation_results = []
    for i, sample in enumerate(dataset):
        print(f"Evaluating sample {i+1}/{len(dataset)}...")

        # Create a single-item dataset for this sample
        single_sample_dataset = Dataset.from_dict({
            "question": [sample["question"]],
            "answer": [sample["answer"]],
            "contexts": [sample["contexts"]]
        })

        # Evaluate with Gemma2
        result_gemma2 = evaluate(
            dataset=single_sample_dataset,
            metrics=[Faithfulness(), ContextUtilization(), AnswerRelevancy()],
            llm=llm_gemma2,
            embeddings=embeddings_nomic,
            run_config=RunConfig(max_workers=16, timeout=30, max_retries=5, max_wait=20, log_tenacity=True)
        )
        gemma2_scores = {metric: result_gemma2[metric][0] if result_gemma2[metric] else "NaN" for metric in result_gemma2._scores_dict.keys()}

        # Combine results into a single row
        evaluation_results.append({
            "Question": sample["question"],
            "Answer": sample["answer"],
            "Contexts": " | ".join(sample["contexts"]),
            "Faithfulness": gemma2_scores.get("faithfulness", "NaN"),
            "ContextUtilization": gemma2_scores.get("context_utilization", "NaN"),
            "AnswerRelevancy": gemma2_scores.get("answer_relevancy", "NaN")
        })

        print(f"Sample {i+1} evaluation completed.\n")

    print("Evaluation completed. Ready to save results.")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results_gemma2.csv")
    header = ["Question", "Answer", "Contexts", "Faithfulness", "ContextUtilization", "AnswerRelevancy"]
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(evaluation_results)
    print(f"All evaluation results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pre-generated climate policy scenarios using the Gemma2 LLM.")
    parser.add_argument("--input-file", default="notebooks/generated_responses_rea.csv", help="Path to the input CSV file with pre-generated responses.")
    parser.add_argument("--output-dir", default="results3", help="Path to save evaluation results.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    input_path = os.path.join(base_dir, args.input_file)
    output_path = os.path.join(base_dir, args.output_dir)

    evaluate_scenarios_gemma2(input_path, output_path)