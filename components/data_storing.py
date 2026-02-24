# Import necessary libraries
import os
import mlflow
import shutil
import sys
from datetime import datetime
import pickle
import joblib

# Define function to create a new experiments output folder for each run 
def make_experiments_folder(output_folder):
    """
    Create a new experiment folder for each run.

    Args:
        output_folder (str): Path to the output folder. 

    Returns:
        experiment_path(str): Path to the experiments folder.
    """

    # Create output folder in the mounted volume if it does not yet exist
    os.makedirs(output_folder, exist_ok=True)

    # Determine next experiment number for saving outputs in container
    existing = [d for d in os.listdir(output_folder) if d.startswith("experiment_") and os.path.isdir(os.path.join(output_folder, d))]
    existing_nums = [int(d.split("_")[1]) for d in existing if d.split("_")[1].isdigit()]
    next_num = max(existing_nums, default=0) + 1

    # Create new experiment output folder
    folder_name = f"experiment_{next_num:03d}"
    experiment_path = os.path.join(output_folder, folder_name)
    os.makedirs(experiment_path)

    return experiment_path

# Define function to log the versions of the role instruction and prompt used for this experiment
def log_versions(log_path, role_instruction_version, prompt_version):
    """
    Logs the versions of the role instruction and prompt used for this experiment.

    Args:
        log_path (str): Path to the logs folder.
        role_instruction_version (int/float): Role instruction version number.
        prompt_version (int/float): Prompt version number.
    """

    with open(log_path, "a") as log_file:
        log_file.write(f"Role instruction version: {role_instruction_version}\n")
    with open(log_path, "a") as log_file:
        log_file.write(f"Prompt version: {prompt_version}\n")

# Define function to format inference times
def format_inference_time(seconds):
    """
    Formats the inference time into hours, minutes and seconds where appropriate.

    Args:
        seconds (float): Inference time in seconds.

    Returns:
        inference_time (FString): Appropriately formatted inference time.
    """

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60

    
    if hours:
        inference_time = f"{hours}h {minutes}m {seconds:.5f}s\n"
        return inference_time
    elif minutes:
        inference_time = f"{minutes}m {seconds:.5f}s\n"
        return inference_time
    else:
        inference_time = f"{seconds:.5f}s\n"
        return inference_time

# Define function to log inference times for NLP
def log_inference_time_NLP(start_time, end_time, df, batch_size, log_path):
    """
    Logs the average inference time per row of data, per batch and the total inference time.

    Args:
        start_time (float): Time when inference starts.
        end_time (float): Time when inference ends.
        df (dataframe): Dataframe containing radiology report data.
        batch_size (int): Number of rows per batch.
        log_path (str): Path to the logs folder.
    """

    total_inference_time = end_time - start_time
    average_row_inference_time = total_inference_time / len(df)
    average_batch_inference_time = average_row_inference_time * batch_size

    total_inference_time_fmt = format_inference_time(total_inference_time)
    average_row_inference_time_fmt = format_inference_time(average_row_inference_time)
    average_batch_inference_time_fmt = format_inference_time(average_batch_inference_time)

    with open(log_path, "a") as log_file:
        log_file.write(f"\nAverage inference time per row: {average_row_inference_time_fmt}\n")
        log_file.write(f"Average inference time per batch: {average_batch_inference_time_fmt}\n")
        log_file.write(f"Total inference time: {total_inference_time_fmt}\n")


# Define function to log inference times for NLI
def log_inference_time_NLI(start_time_rf, end_time_rf, start_time_r, end_time_r, df, batch_size, log_path, max_NLI_rf_inference_time, max_NLI_r_inference_time):
    """
    Logs the average inference time per row of data, per batch and the total inference time for both the NLI of the 
    report findings and rationales.

    Args:
        start_time_rf (float): Time when report findings inference starts.
        end_time_rf (float): Time when report findings inference ends.
        start_time_r (float): Time when rationale inference starts.
        end_time_r (float): Time when rationale inference ends.
        df (dataframe): Dataframe containing radiology report data.
        batch_size (int): Number of rows per batch.
        log_path (str): Path to the logs folder.
        max_NLI_rf_inference_time (float): Maximum time for any one inference of the report findings.
        max_NLI_r_inference_time (float): Maximum time for any one inference of the rationales.
    """

    total_report_findings_inference_time = end_time_rf - start_time_rf
    average_row_report_findings_inference_time = total_report_findings_inference_time / len(df)
    average_batch_report_findings_inference_time = average_row_report_findings_inference_time * batch_size

    total_rationale_inference_time = end_time_r - start_time_r
    average_row_rationale_inference_time = total_rationale_inference_time / len(df)
    average_batch_rationale_inference_time = average_row_rationale_inference_time * batch_size

    total_NLI_inference_time = total_report_findings_inference_time + total_rationale_inference_time

    total_report_findings_inference_time_fmt = format_inference_time(total_report_findings_inference_time)
    average_row_report_findings_inference_time_fmt = format_inference_time(average_row_report_findings_inference_time)
    average_batch_report_findings_inference_time_fmt = format_inference_time(average_batch_report_findings_inference_time)

    total_rationale_inference_time_fmt = format_inference_time(total_rationale_inference_time)
    average_row_rationale_inference_time_fmt = format_inference_time(average_row_rationale_inference_time)
    average_batch_rationale_inference_time_fmt = format_inference_time(average_batch_rationale_inference_time)

    total_NLI_inference_time_fmt = format_inference_time(total_NLI_inference_time)

    with open(log_path, "a") as log_file:
        log_file.write(f"\nAverage report findings inference time per row: {average_row_report_findings_inference_time_fmt}\n")
        log_file.write(f"NLI Max report findings inference time: {max_NLI_rf_inference_time}\n")
        log_file.write(f"Average report findings inference time per batch: {average_batch_report_findings_inference_time_fmt}\n")
        log_file.write(f"Total report findings inference time: {total_report_findings_inference_time_fmt}\n")

        log_file.write(f"\nAverage rationale inference time per row: {average_row_rationale_inference_time_fmt}\n")
        log_file.write(f"NLI Max rationale inference time: {max_NLI_r_inference_time}\n")
        log_file.write(f"Average rationale inference time per batch: {average_batch_rationale_inference_time_fmt}\n")
        log_file.write(f"Total rationale inference time: {total_rationale_inference_time_fmt}\n")

        log_file.write(f"\nTotal NLI inference time: {total_NLI_inference_time_fmt}")


def log_token_lengths(generated_token_length_list, inference_time_list, log_path):
    '''
    Logs the Max inference time, average token length and max token length of a group of outputs.

    Args:
        generated_token_length_list (list): List of generated token lengths.
        inference_time_list (list): List of inference times.
        log_path (str): Path to the log file.
    '''

    try:
        avg_generated_token_length = sum(generated_token_length_list) / len(generated_token_length_list)
        max_generated_token_length = max(generated_token_length_list)

        max_inference_time = max(inference_time_list)

        with open(log_path, "a") as log_file:
            log_file.write(f"Max Inference time: {max_inference_time}\n")
            log_file.write(f"Average generated token length: {avg_generated_token_length}\n")
            log_file.write(f"Max generated token length: {max_generated_token_length}\n\n")

    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"Error with log_token_lengths: {e}\n") 


def log_performance_metrics(log_path, confidence_score, max_probability, perplexity, avg_entropy, energy, generated_scores, token_probabilities, scaled_probabilities, generated_token_length):
    """
    Logs all of the calculated perfomance metrics of the model

    Args:
        log_path (str): Path to the logs folder.
        confidence_score (float): Final confidence score of the model.
        max_probability (float): The maximum value of the probability tokens.
        perplexity (float): Perplexity value of the model.
        avg_entropy (float): Entropy value of the model.
        energy (float): Energy value of the model.
        generated_scores (tuple): All generated logits of the model.
        token_probabilites (tuple): All token probabilities of the model.
        scaled_probabilities (tuple): All temperature scaled probabilities of the model.
        generated_token_length (int): Token length of the generated output.
    """
    with open(log_path, "a") as log_file:
        log_file.write(f"Final confidence score: {confidence_score}\n")
        log_file.write(f"Maximum probability: {max_probability}\n")
        log_file.write(f"Perplexity: {perplexity}\n")
        log_file.write(f"Entropy: {avg_entropy}\n")
        log_file.write(f"Energy: {energy}\n")
        log_file.write(f"Token length: {generated_token_length}\n")


# Define function to save and copy outputs, logs and config file into the experiment folder
def save_to_volume(log_path, config_path, input_path, experiment_path):
    """
    Save and copy outputs, logs and config files into the experiments folder in the volume.

    Args:
        log_path (str): Path to the logs folder.
        config_path (str): Path to the config file.
        input_path (str): Path to the input data.
        experiment_path (str): Path to the experiment folder for this run.
    """

    for file_name in [log_path, config_path, input_path]:

        src_path = os.path.join(".", file_name)
        dest_path = os.path.join(experiment_path, file_name)
        
        # Ensure destination folder exists
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            with open(dest_path, "w") as f:
                f.write(f"Placeholder for {file_name}\n")


# Define function to save and copy outputs, logs and config file to MLflow
def save_to_mlflow(role_instruction_version, prompt_version, confidence_score, max_probability, perplexity, avg_entropy, energy, intermediate_results_path, final_results_path, NLI_combined_results_path, log_path, config_path, input_path):
    """
    Save and copy outputs, logs and config files into the experiments folder in the volume.

    Args:
        role_instruction_version (float): Version number of the role instruction used in this experiment.
        prompt_version (float): Version number of the prompt used in this experiment.
        confidence_score (float): Final confidence score of the model.
        max_probability (float): The maximum value of the probability tokens.
        perplexity (float): Perplexity value of the model.
        entropy (float): Entropy value of the model.
        energy (float): Energy value of the model.
        intermediate_results_path (str): Path to the saved intermediate results.
        final_results_path (str): Path to the saved final results.
        NLI_combined_results_path (str): Path to the combined NLI results.
        log_path (str): Path to the logs folder.
        config_path (str): Path to the config file.
        input_path (str): Path to where the input file is saved.
    """

    # Define MLflow metrics and parameters
    metrics = {
        "Role instruction version": role_instruction_version,
        "Prompt Version": prompt_version,
        "Final confidence score": confidence_score,
        "Maximum probability": max_probability,
        "Perplexity": perplexity,
        "Entropy": avg_entropy,
        "Energy": energy,
    }

    # Save intermediate_results.csv to MLflow
    with mlflow.start_run():
        # Log results, logs and config as artifacts
        mlflow.log_artifact(intermediate_results_path, artifact_path="results")
        mlflow.log_artifact(final_results_path, artifact_path="results")
        mlflow.log_artifact(NLI_combined_results_path, artifact_path="results")
        mlflow.log_artifact(log_path, artifact_path="logs")
        mlflow.log_artifact(config_path, artifact_path="config")
        mlflow.log_artifact(input_path, artifact_path="input")

        # Log prompt metrics
        mlflow.log_metrics(metrics)
