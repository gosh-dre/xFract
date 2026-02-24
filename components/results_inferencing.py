import pandas as pd
import json
import mlflow
import glob
import os
import argparse
import logging
import time
import gc 
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, PretrainedConfig


def Load_NLI_model(NLI_model_path):
    """
    Loads the NLI model MediPhi-Clinical from the mounted volume.

    Args:
        NLI_model_path (string): Path to the NLI model in the named volume.

    Returns:
        medphi_tokenizer (AutoTokenizer): Handles text preprocessing for models.
        medphi_model (AutoModelForCausalLM): Causal language model for predicting next token in sequence.
    """

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
    medphi_tokenizer = AutoTokenizer.from_pretrained(NLI_model_path)
    medphi_model = AutoModelForCausalLM.from_pretrained(NLI_model_path, dtype=torch.bfloat16).to(device)
    print("Loaded NLI MODEL")
    return medphi_tokenizer, medphi_model


def table_extraction(prompt_update, medphi_model, medphi_tokenizer):
    """
    Formats the input for the NLI model, creates the pipeline using the MedPhi-Clinical model, defines 
    the arguments for generated outputs and generates the outputs from the pipeline.

    Args:
        prompt_update (string): Containing cleaned NLI_input.
        medphi_model (AutoModelForCausalLM): Causal language model for predicting next token in sequence.
        medphi_tokenizer (AutoTokenizer): Handles text preprocessing for models.
        
    Returns:
        output (string): Output from the NLI MediPhi-Clinical model.
    """

    messages = [
        {"role": "system", "content": prompt_update},
        {"role": "user", "content": "Output: "},
    ]
    
    pipe = pipeline(
        "text-generation",
        model=medphi_model,
        tokenizer=medphi_tokenizer,
        device=1
    )
    
    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }
    
    output = pipe(messages, **generation_args)
    return output

def Run_NLI_inference_report_findings(df, NLI_prompt_text, medphi_model, medphi_tokenizer, output_folder, log_path):
    """
    Runs the NLI inference on the MedGemma4b report findings to give an answer, score and rationale and stores them in 
    a pandas Dataframe.

    Args:
        df (Dataframe): Contains the output data from the NLP MedGemma4b model
        NLI_prompt_text (string): The prompt to be input into the NLI model.
        medphi_model (AutoModelForCausalLM): Causal language model for predicting next token in sequence.
        medphi_tokenizer (AutoTokenizer): Handles text preprocessing for models.
        output_folder (string): Path to the output folder in the container.
        log_path (string): Path to the log folder in the container.
        
    Returns:
        NLI_report_findings_df (Dataframe): Containing the reports, predictions, report findings, NLI answer, score and 
            rationale.
        max_NLI_rf_inference_time (float): The maximum inference time for all given NLI outputs on the report findings.
    """

    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())

    max_NLI_rf_inference_time = 0.0
    
    final_details = []

    narrative = df.narrative.to_list()
    report_findings = df.report_findings.to_list()
    pathological_fracture = df.pathological_fracture.to_list()
    
    for num, input in enumerate(narrative):

        time_start = time.time()

        details = {}
        details["Report"] = input
        details["prediction"] = pathological_fracture[num]
        details["report_findings"] = [s.replace('"', '\\"') for s in report_findings[num]]

        output = table_extraction(NLI_prompt_text.format(context=input, statement=details["report_findings"]), medphi_model, medphi_tokenizer)[0]["generated_text"]
        with open(log_path, "a") as log_file:
            log_file.write(f"Raw NLI_inference_report_findings: {output}\n")

        try:
            cleaned_output = str(output).replace("json","").replace("```","")
            with open(log_path, "a") as log_file:
                log_file.write(f"Cleaned NLI_inference_report_findings: {cleaned_output}\n")
            my_list = json.loads(cleaned_output)

            details.update(my_list)

        except json.JSONDecodeError as e:
            with open(log_path, "a") as log_file:
                log_file.write(f"Error decoding JSON: {e}\n")

        final_details.append(details)

        time_end = time.time()
        NLI_rf_inference_time = time_end - time_start

        with open(log_path, "a") as log_file:
            log_file.write(f"NLI report findings inference time: {NLI_rf_inference_time}\n")

        if NLI_rf_inference_time > max_NLI_rf_inference_time:
            max_NLI_rf_inference_time = NLI_rf_inference_time

    NLI_report_findings_df = pd.DataFrame(final_details)
  
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())

    return NLI_report_findings_df, max_NLI_rf_inference_time
        

def Run_NLI_inference_rationale(df, NLI_prompt_text, medphi_model, medphi_tokenizer, output_folder, log_path):
    """
    Runs the NLI inference on the MedGemma4b rationales to give an answer, score and rationale and stores them in 
    a pandas Dataframe.

    Args:
        df (Dataframe): Contains the output data from the NLP MedGemma4b model
        NLI_prompt_text (string): The prompt to be input into the NLI model.
        medphi_model (AutoModelForCausalLM): Causal language model for predicting next token in sequence.
        medphi_tokenizer (AutoTokenizer): Handles text preprocessing for models.
        output_folder (string): Path to the output folder in the container.
        log_path (string): Path to the log folder in the container.
        
    Returns:
        NLI_rationale_df (Dataframe): Containing the reports, predictions, NLP prediction rationale, NLI answer, 
            score and NLI rationale.
        max_NLI_r_inference_time (float): The maximum inference time for all given NLI outputs on the rationales.
    """

    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())

    max_NLI_r_inference_time = 0.0
    
    final_details = []

    narrative = df.narrative.to_list()
    rationale = df.rationale.to_list()
    pathological_fracture = df.pathological_fracture.to_list()
    
    for num, input in enumerate(narrative):
        
        time_start = time.time()

        details = {}
        details["Report"] = input
        details["prediction"] = pathological_fracture[num]
        details["prediction_rationale"] = rationale[num].replace('"', '\\"')
        
        output = table_extraction(NLI_prompt_text.format(context=input, statement=details["prediction_rationale"]), medphi_model, medphi_tokenizer)[0]["generated_text"]
        with open(log_path, "a") as log_file:
            log_file.write(f"NLI_inference_rationale: {output}\n")

        try:
            cleaned_output = str(output).replace("json","").replace("```","")
            with open(log_path, "a") as log_file:
                log_file.write(f"Cleaned NLI_inference_rationale: {cleaned_output}\n")
            my_list = json.loads(str(output).replace("json","").replace("```",""))

            details.update(my_list)

        except json.JSONDecodeError as e:
            with open(log_path, "a") as log_file:
                log_file.write(f"Error decoding JSON: {e}\n")

        final_details.append(details)

        time_end = time.time()
        NLI_r_inference_time = time_end - time_start

        with open(log_path, "a") as log_file:
            log_file.write(f"NLI rationale inference time: {NLI_r_inference_time}\n")

        if NLI_r_inference_time > max_NLI_r_inference_time:
            max_NLI_r_inference_time = NLI_r_inference_time

    NLI_rationale_df = pd.DataFrame(final_details)
  
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())

    return NLI_rationale_df, max_NLI_r_inference_time


def combine_NLI_dfs(NLI_report_findings_df, NLI_rationale_df, output_folder):
    """
    Combines the NLI_report_findings_df and NLI_rationale_df into one dataframe

    Args:
        NLI_report_findings_df (Dataframe): Containing the reports, predictions, report findings, NLI answer, score and 
            rationale.
        NLI_rationale_df (Dataframe): Containing the reports, predictions, NLP prediction rationale, NLI answer, 
            score and NLI rationale.
        output_folder (string): Path to the output folder in the container.
        
    Returns:
        NLI_combined_results_df (Dataframe): Containing the reports, predictions, NLP report findings, 
            NLI report findings answers, NLI report findings scores, NLI report findings rationale, NLP prediction rationale, 
            NLI prediction rationale answers, NLI prediction rationale scores and the NLI rationale of the NLP prediction rationale.
    """
    
    cols_from_report_findings = NLI_report_findings_df[["Report", "prediction", "report_findings", "Answer", "Score", "Rationale"]].rename(columns={"Answer": "report_findings Answer", "Score": "report_findings Score", "Rationale": "report_findings Rationale"})
    cols_from_rationale = NLI_rationale_df[["prediction_rationale", "Answer", "Score", "Rationale"]].rename(columns={"Answer": "prediction_rationale Answer", "Score": "prediction_rationale Score", "Rationale": "prediction_rationale Rationale"})

    # combined_NLI_df = pd.merge(NLI_report_findings_df[["Report", "prediction", "report_findings", "Answer", "Score", "Rationale"]], NLI_rationale_df[["prediction_rationale", "Answer", "Score", "Rationale"]], left_index=True, right_index=True)
    combined_NLI_df = pd.concat([cols_from_report_findings, cols_from_rationale], axis=1)

    # Save to CSV
    NLI_combined_results_path = os.path.join(output_folder, "radiology_qi_nli.csv")
    combined_NLI_df.to_csv(NLI_combined_results_path, index=False)

    return NLI_combined_results_path
