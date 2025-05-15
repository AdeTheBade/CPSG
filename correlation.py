import pandas as pd
import argparse
import os

def compute_correlations(input_file):
    """
    Compute mean scores, standard deviations, and Spearman rank correlations for evaluation metrics.
    
    Args:
        input_file (str): Path to the input CSV file containing evaluation results.
    """
    # Load dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")
    df = pd.read_csv(input_file)

    # Define relevant columns
    relevant_columns = [
        'Gemma2_2b_faithfulness', 'Mistral_7b_faithfulness',
        'Gemma2_2b_answer_relevancy', 'Mistral_7b_answer_relevancy',
        'Gemma2_2b_context_utilization', 'Mistral_7b_context_utilization',
        'Human_faithfulness', 'Human_answer_relevancy', 'Human_context_utilization'
    ]

    # Compute column means and standard deviations, ignoring 0s
    column_means = df[relevant_columns].replace(0, pd.NA).mean()
    column_stds = df[relevant_columns].replace(0, pd.NA).std()

    # Display means and standard deviations
    print("\n**Mean Scores and Standard Deviations for Each Metric:**\n")
    for col, mean, std in zip(relevant_columns, column_means, column_stds):
        print(f"{col}: Mean = {mean:.3f}, Std = {std:.3f}")

    # Define columns for correlation
    corr_columns = [
        'Gemma2_2b_faithfulness', 
        'Mistral_7b_faithfulness',
        #'Gemma2_2b_answer_relevancy', 
        #'Mistral_7b_answer_relevancy',
        #'Gemma2_2b_context_utilization', 
        #'Mistral_7b_context_utilization',
        #'Human_faithfulness', 
        #'Human_answer_relevancy', 
        #'Human_context_utilization'
    ]

    # Identify rows that contain at least one NaN or zero
    mask = df[corr_columns].isna().any(axis=1) | (df[corr_columns] == 0).any(axis=1)
    print(f"\nNumber of rows identified with NaN or zero: {mask.sum()}")

    # Replace entire rows with zeros where any NaN or zero exists
    df.loc[mask, corr_columns] = 0

    # Case 1: No Forward-Fill
    df_no_fill = df[corr_columns].copy()

    # Compute Spearman Rank Correlation
    spearman_corr_no_fill = df_no_fill.corr(method='spearman')

    # Extract and print key correlations
    key_metrics = [
        ('Gemma2_2b_faithfulness', 'Mistral_7b_faithfulness'),
        #('Gemma2_2b_answer_relevancy', 'Mistral_7b_answer_relevancy'),
        #('Gemma2_2b_context_utilization', 'Mistral_7b_context_utilization'),
        #('Human_faithfulness', 'Gemma2_2b_faithfulness'),
        #('Human_answer_relevancy', 'Gemma2_2b_answer_relevancy'),
        #('Human_context_utilization', 'Gemma2_2b_context_utilization'),
        #('Human_faithfulness', 'Mistral_7b_faithfulness'),
        #('Human_answer_relevancy', 'Mistral_7b_answer_relevancy'),
        #('Human_context_utilization', 'Mistral_7b_context_utilization')
    ]

    def print_key_correlations(correlation_matrix, case_name):
        print(f"\n**Spearman Rank Correlations - {case_name}:**\n")
        for metric1, metric2 in key_metrics:
            print(f"{metric1} vs {metric2}: {correlation_matrix.loc[metric1, metric2]:.3f}")

    print_key_correlations(spearman_corr_no_fill, "No Forward Fill")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute correlations for evaluation metrics.")
    parser.add_argument("--input-file", default="results/Compiled evaluation result.csv", help="Path to the input CSV file containing evaluation results.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    input_path = os.path.join(base_dir, args.input_file)

    compute_correlations(input_path)