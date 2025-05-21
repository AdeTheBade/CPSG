import os
import argparse
import pandas as pd
from langchain_ollama import OllamaLLM

def generate_follow_up_responses(input_file, output_file):
    """
    Generate follow-up responses for climate policy scenarios.
    
    Args:
        input_file (str): Path to the input CSV file containing scenarios.
        output_file (str): Path to save the output CSV file with follow-up responses.
    """
    # Load CSV file containing scenarios
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"The input file {input_file} does not exist.")
    df = pd.read_csv(input_file)

    # Initialize the new LLM model
    ollama = OllamaLLM(
        base_url="http://localhost:11434",
        model="llama3:latest",
        temperature=0,
    )

    # Function to generate follow-up responses
    def process_scenario(scenario, index, total):
        # Define structured follow-up prompts
        follow_up_prompt_1 = f"""
        Considering the following climate policy scenarios:
        '{scenario}',
        what specific metrics can be used to measure the success of each scenario?
        Identify Key Performance Indicators (KPIs) that track progress and milestones that indicate meaningful advancements.
        Present the metrics in a structured format with justifications for each.
        """

        follow_up_prompt_2 = f"""
        For the given climate policy scenarios:
        '{scenario}',
        identify the key drivers that will influence the success or failure of each scenario.
        Consider factors such as economic conditions, policy frameworks, technology readiness, public perception, and environmental constraints.
        Additionally, analyze how these key drivers interact with each other, highlighting dependencies or feedback loops that could impact the scenario.
        """

        follow_up_prompt_3 = f"""
        For each of the climate policy scenarios presented:
        '{scenario}',
        outline a clear implementation roadmap with well-defined milestones.
        Define short-term (0-2 years), medium-term (3-7 years), and long-term (8+ years) milestones.
        Highlight critical decision points, dependencies, and risks that could impact progress.
        Provide specific action steps that should be taken at each milestone to ensure scenario success.
        """

        # Generate responses using the model
        measurable_outcomes_response = ollama.invoke(follow_up_prompt_1)
        key_drivers_response = ollama.invoke(follow_up_prompt_2)
        milestones_response = ollama.invoke(follow_up_prompt_3)

        # Extract text results while handling errors
        measurable_outcomes = measurable_outcomes_response if isinstance(measurable_outcomes_response, str) else measurable_outcomes_response.get("result", "")
        key_drivers = key_drivers_response if isinstance(key_drivers_response, str) else key_drivers_response.get("result", "")
        milestones = milestones_response if isinstance(milestones_response, str) else milestones_response.get("result", "")

        # Print progress update
        print(f"Processed prompt {index + 1}/{total}: '{df.loc[index, 'Question']}'")

        return measurable_outcomes, key_drivers, milestones

    # Lists to store results
    measurable_outcomes_list = []
    key_drivers_list = []
    milestones_list = []

    # Loop through each scenario and generate follow-up responses
    for index, scenario in enumerate(df["Answer"]):
        outcomes, drivers, milestones = process_scenario(scenario, index, len(df))
        measurable_outcomes_list.append(outcomes)
        key_drivers_list.append(drivers)
        milestones_list.append(milestones)

    # Add new columns to DataFrame
    df["Measurable Outcomes"] = measurable_outcomes_list
    df["Key Drivers"] = key_drivers_list
    df["Milestones"] = milestones_list

    # Save updated CSV file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Follow-up responses generated! Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate follow-up responses for climate policy scenarios.")
    parser.add_argument("--input-file", default="data/generated_responses.csv", help="Path to the input CSV file containing scenarios.")
    parser.add_argument("--output-file", default="results/follow_up_prompt_responses.csv", help="Path to save the output CSV file.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    input_path = os.path.join(base_dir, args.input_file)
    output_path = os.path.join(base_dir, args.output_file)

    generate_follow_up_responses(input_path, output_path)