# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import pipeline, AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import mlflow
import yaml
import time
import shutil
import numpy as np
from components import data_ingestion, data_processing, data_storing, results_inferencing, api_component

# Load config file
config_path = "config/config.yaml"
config = data_ingestion.load_config(config_path)

# Define file paths from config
output_folder = config["output"]["output_path"]
role_instruction_path = config["prompts"]["role_instruction_path"]
prompt_path = config["prompts"]["prompt_path"]
mlflow_role_instruction_path = config["prompts"]["mlflow_role_instruction_path"]
mlflow_prompt_path = config["prompts"]["mlflow_prompt_path"]
NLI_prompt_path = config["prompts"]["NLI_prompt_path"]
mlflow_NLI_prompt_path = config["prompts"]["mlflow_NLI_prompt_path"]
log_path = config["logging"]["log_path"]
input_path = config["input"]["input_path"]
model_path = config["model"]["model_path"]
NLI_model_path = config["model"]["NLI_model_path"]
mlflow_path = config["mlflow"]["mlflow_server"]
batch_size = config["dataloader"]["batch_size"]

# Check if api is enabled
api_enabled = os.getenv("API_ENABLED", "false").lower() == "true"

# Connect API and retrieve input data if available
if api_enabled == True:
    api_component.connect_api(input_path, log_path)

# Check if mlflow is enabled
mlflow_enabled = os.getenv("MLFLOW_TRACKING_ENABLED", "false").lower() == "true"

# Set up MLflow tracking if enabled
if mlflow_enabled:
    mlflow.set_tracking_uri(mlflow_path)
    qi_project_experiment = mlflow.set_experiment("qi_fracture_project")

# If mlflow disabled, create new experiment output folder for each experiment
if mlflow_enabled == False:
    experiment_path = data_storing.make_experiments_folder(output_folder)

# Load multimodal processor and transformer model
tokenizer, model = data_processing.load_processor_and_model(model_path, log_path)

# Load NLI model 
medphi_tokenizer, medphi_model = results_inferencing.Load_NLI_model(NLI_model_path)

# Create preconfigured model, tokenizer and processor pipeline
pipe = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=tokenizer,
    trust_remote_code=True,
    max_new_tokens=1024,
    device=1
)

# Load CSV data with custom dataset and dataloader
dataset = data_ingestion.RadiologyDataset(input_path)
def collate_keep_list(batch):
    return batch
dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False, collate_fn=collate_keep_list)

# Load role instruction and prompt for MedGemma model from json (gitlab) if mlflow disabled and log them
if mlflow_enabled == False:
    # Load role instruction and prompt
    role_instruction_text, role_instruction_version, prompt_text, prompt_version = data_ingestion.load_prompt_json(role_instruction_path, prompt_path)

    # Log versions
    data_storing.log_versions(log_path, role_instruction_version, prompt_version)

# Load role instructions and prompt for MedGemma model from mlflow if enabled
if mlflow_enabled:
    # Load role instruction and prompt
    role_instruction_text, prompt_text, role_instruction_version, prompt_version = data_ingestion.load_prompt_mlflow(mlflow_role_instruction_path, mlflow_prompt_path)

    # Log versions
    data_storing.log_versions(log_path, role_instruction_version, prompt_version)

# Format model input
messages = data_processing.format_input(role_instruction_text, prompt_text)

# Run process_batches_with_model function on every batch in dataset, stores the output in a column in df called "gemmajson" and log inference times
start_time_NLP = time.time()
df, generated_token_length_list, inference_time_list, gen_out, enc = data_processing.process_batches_with_model(dataloader, role_instruction_text, prompt_text, pipe, tokenizer, messages, model, log_path, batch_size, dataset)
end_time_NLP = time.time()
data_storing.log_inference_time_NLP(start_time_NLP, end_time_NLP, df, batch_size, log_path)
data_storing.log_token_lengths(generated_token_length_list, inference_time_list, log_path)

# Create output folder in the mounted volume if it does not yet exist
os.makedirs(output_folder, exist_ok=True)

# Save df with output as intermediate results
intermediate_results_path = os.path.join(output_folder, "intermediate_results.csv")
df.to_csv(intermediate_results_path, index=False)

# If mlflow disabled, save versions to experiment folder in volume
if mlflow_enabled == False:
    volume_intermediate_results_path = os.path.join(experiment_path, "intermediate_results.csv")
    df.to_csv(volume_intermediate_results_path, index=False)
    
# Apply the orchastorator_fordataprocessing() function on the "gemmajson" column of the df to save the relevant data into new columns in the df
df[['fracture_mentioned','pathological_fracture','report_findings','rationale']] = df['gemmajson'].apply(lambda x: data_processing.orchastorator_fordataprocessing(x, log_path)).apply(pd.Series)

# Calculate and log overall perfomance metrics of the model
confidence_score, max_probability, perplexity, avg_entropy, energy, generated_scores, token_probabilities, scaled_probabilities, generated_token_length = data_processing.calculate_performance_metrics(gen_out, enc, 0) 
data_storing.log_performance_metrics(log_path, confidence_score, max_probability, perplexity, avg_entropy, energy, generated_scores, token_probabilities, scaled_probabilities, generated_token_length)

# Save the final results as a csv
final_results_path = os.path.join(output_folder, "final_results.csv")
df.to_csv(final_results_path, index=False)

# Load NLI prompt from JSON if mlflow disabled and log version
if mlflow_enabled == False:
    # Load NLI prompt
    NLI_prompt_text, NLI_prompt_version = data_ingestion.load_NLI_prompt_json(NLI_prompt_path)

    # Log version
    with open(log_path, "a") as log_file:
        log_file.write(f"NLI prompt version: {NLI_prompt_version}\n")
        log_file.write(f"NLI PROMPT FROM MLFLOW: {NLI_prompt_text}\n")

# Load NLI prompt from mlflow if enabled and log version
if mlflow_enabled:
    # Load NLI prompt
    NLI_prompt_text, NLI_prompt_version = data_ingestion.load_NLI_prompt_mlflow(mlflow_NLI_prompt_path)

    # Log version
    with open(log_path, "a") as log_file:
        log_file.write(f"NLI prompt version: {NLI_prompt_version}\n")
        log_file.write(f"NLI PROMPT FROM MLFLOW: {NLI_prompt_text}\n")

# Run NLI Inference
start_time_rf = time.time()
NLI_report_findings_df, max_NLI_rf_inference_time = results_inferencing.Run_NLI_inference_report_findings(df, NLI_prompt_text, medphi_model, medphi_tokenizer, output_folder, log_path)
end_time_rf = time.time()
start_time_r = time.time()
NLI_rationale_df, max_NLI_r_inference_time = results_inferencing.Run_NLI_inference_rationale(df, NLI_prompt_text, medphi_model, medphi_tokenizer, output_folder, log_path)
end_time_r = time.time()
data_storing.log_inference_time_NLI(start_time_rf, end_time_rf, start_time_r, end_time_r, df, batch_size, log_path, max_NLI_rf_inference_time, max_NLI_r_inference_time)
NLI_combined_results_path = results_inferencing.combine_NLI_dfs(NLI_report_findings_df, NLI_rationale_df, output_folder)

# If mlflow disabled, save versions to experiment folder in volume
if mlflow_enabled == False:
    volume_final_results_path = os.path.join(experiment_path, "final_results.csv")
    df.to_csv(volume_final_results_path, index=False)

# Save and copy inputs, logs and config files into the experiment folder if mlflow disabled
if mlflow_enabled == False:
    data_storing.save_to_volume(log_path, config_path, input_path, experiment_path)

# Save and copy outputs, logs and config files to MLflow if enabled
if mlflow_enabled:
    data_storing.save_to_mlflow(role_instruction_version, prompt_version, confidence_score, max_probability, perplexity, avg_entropy, energy, intermediate_results_path, final_results_path, NLI_combined_results_path, log_path, config_path, input_path)