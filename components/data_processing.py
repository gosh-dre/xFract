# Import necessary libraries
from transformers import pipeline, AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F
import numpy as np
import os
import gc
import time
from typing import List, Dict
from transformers import StoppingCriteria, StoppingCriteriaList
from components import data_storing

# Define function to load multimodal processor and transformer model
def load_processor_and_model(model_path, log_path):
    """
    Loads the processor and model for MedGemma4b.

    Args:
        model_path (str): Path to the MedGemma4b model.
        log_path (str): Path to the logs folder

    Returns:
        tokenizer (AutoTokenizer): Tokenizer for inputs.
        model (AutoModelForImageToText): Transformer model capable of generating text from both image and text inputs.
    """

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)

    with open(log_path, "a") as log_file:
        log_file.write(f"Loaded the processor and model\n")

    return tokenizer, model

# Define function to format the input
def format_input(role_instruction_text, prompt_text):
    """
    Formats the role instruction and prompt into the correct chat format for the model input.

    Args:
        role_instruction_text (str): Text containing the role instruction.
        prompt_text (str): Text containing the prompt.

    Returns:
        messages (str): Formatted role instructions and prompt text.
    """

    messages = [
                        {
                            "role": "system",
                            "content": role_instruction_text
                        },
                        {
                            "role": "user",
                            "content": prompt_text
                        }
                    ]

    return messages


class FinishTimeRecorder(StoppingCriteria):
    """
    Records the wall-clock time (perf_counter) when each sequence in the batch first emits EOS. Items that never hit EOS remain 
    None. This means that it records and logs the inference time for each individual output in the batch.

    Parameters
    ----------
    eos_token_id: int
        Int containing the end of sequence token id.
    batch_size: int
        Int representing the size of the batches.
    start_time: float
        Float containing the time at which an inference begins.
    input_ids: torch.LongTensor
        torch.LongTensor containing the tokenized input text.
    scores: torch.FloatTensor
        torch.FloatTensor containing the logit scores produced by the model when decoding.
    """

    def __init__(self, eos_token_id: int, batch_size: int, start_time: float = None):
        """
        Inherits functions from the "StoppingCriteria" class. Extracts the data about the eos_tokens, batch_size, start_time and
        the finish time "finished_at".

        Parameters
        ----------
        eos_token_id: int
            Int containing the end of sequence token id.
        batch_size: int
            Int representing the size of the batches.
        start_time: float
        Float containing the time at which an inference begins.
        """

        super().__init__()
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self.start_time = start_time if start_time is not None else time.perf_counter()
        self.finished_at = [None] * batch_size 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        """
        Record and return the finished time when the last token of a sequence equals EOS meaning that the text generation has 
        finished.

        Parameters
        ----------
        input_ids: torch.LongTensor
            torch.LongTensor containing the tokenized input text.
        scores: torch.FloatTensor
            torch.FloatTensor containing the logit scores produced by the model when decoding.
        """

        now = time.perf_counter()
        for i in range(input_ids.size(0)):
            if self.finished_at[i] is None:
                if input_ids[i, -1].item() == self.eos_token_id:
                    self.finished_at[i] = now
        return False


def format_and_run_batch(role_instruction_text, prompt_text, batch, pipe, tokenizer, messages, model, log_path, batch_size):
    """
    Prepares the text input format, tokenizes the input, records the inference time for each text generation, calculates the 
    performance metrics and returns the response and metrics.

    Args:
        role_instruction_text (str): Text containing the role instruction.
        prompt_text (str): Text containing the prompt.
        batch (list): List containing the "narrative" for each batch of inputs.
        pipe (pipeline): Pipeline for generating text from the input data.
        tokenizer (tokenizer): Tokenizer for calculating performance metrics.
        messages (list): List of dicts containing the input for the LLM model.
        model (AutoModelForCausalLM): Model for calculating performance metrics.
        log_path (str): Path to the logs folder.
        batch_size (int): The size of the batches.

    Returns:
        response (list): list of strings containing generated response output from the pipeline for each input in the batch.
        confidence_score_ls (list): list of floats containing final confidence scores of the model for each input in the batch.
        max_probability_ls (list): list of floats containing the maximum values of the probability tokens for each input in the 
            batch.
        perplexity_ls (list): list of floats containing the perplexity values of the model for each input in the batch.
        avg_entropy_ls (list): list of floats containing the entropy values of the model for each input in the batch.
        energy_ls (list): list of floats containing the energy values of the model for each input in the batch.
        generated_token_length_ls (list): list of integers containing the token lengths of the NLP output response for each input 
            in the batch.
        per_item_inference_time (list): list of floats containing the time taken for the output to be generated and the
            performance metrics to be calcluated for each input in the batch.
        gen_out (transformers.modeling_outputs.ModelOutput): Output from the LLM containing response and scores.
        enc (transformers.tokenization_utils_base.BatchEncoding): For calculating performance metrics of the outputs.
    """

    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())

    batch_messages = []
    for item in batch:
        messages = [
            {
                "role": "system",
                "content": role_instruction_text
            },
            {
                "role": "user",
                "content": prompt_text.format(narrative=item["narrative"])
            }
        ]

        batch_messages.append(messages)

    # Convert messages to prompts via chat template
    prompts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in batch_messages
    ]

    # Tokenize the batch 
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=2048,         
        return_tensors="pt"
    )

    # Move tensors in place without destroying BatchEncoding
    for k in enc:
        enc[k] = enc[k].to(model.device)

    # Set up item time recorder
    recorder = FinishTimeRecorder(
        eos_token_id=tokenizer.eos_token_id,
        batch_size=batch_size,
        start_time=time.perf_counter()
    )

    # Batched text generation with timestamps for each finish time
    model.eval()
    with torch.no_grad():
        gen_out = model.generate(
            **enc,
            max_new_tokens=1024,         
            do_sample=False,             
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=StoppingCriteriaList([recorder]),
        )

    # Decode responses per item from tokens
    input_len = enc["input_ids"].shape[1]
    response = []
    for i in range(batch_size):
        gen_ids_i = gen_out.sequences[i, input_len:]  # slice off the prompt
        gen_text = tokenizer.decode(gen_ids_i, skip_special_tokens=True)
        response.append(gen_text)

    # Individual row inference times per input in the batch
    batch_end_time = time.perf_counter()
    per_item_inference_time = [
        ((t if t is not None else batch_end_time) - recorder.start_time)
        for t in recorder.finished_at
    ]

    # Performance metrics calculations
    confidence_score_ls = []
    max_probability_ls = []
    perplexity_ls = []
    avg_entropy_ls = []
    energy_ls = []
    generated_scores_ls = []
    token_probabilities_ls = []
    scaled_probabilities_ls = []
    generated_token_length_ls = []

    for batch_idx in range(batch_size):
        (confidence_score, max_probability, perplexity, avg_entropy, energy,
         generated_scores, token_probabilities, scaled_probabilities,
         generated_token_length) = calculate_performance_metrics(gen_out, enc, batch_idx)

        with open(log_path, "a") as log_file:
            log_file.write(f"FINAL INPUT: {json.dumps(batch_messages[batch_idx], indent=2)}\n")
            log_file.write(f"format_and_run(): LLM response successfully generated\n")
            log_file.write(f"Raw LLM response: {response[batch_idx]}\n")

        data_storing.log_performance_metrics(
            log_path, confidence_score, max_probability, perplexity, avg_entropy,
            energy, generated_scores, token_probabilities, scaled_probabilities,
            generated_token_length
        )

        with open(log_path, "a") as log_file:
            log_file.write(f"Inference time: {per_item_inference_time[batch_idx]}s\n")

        confidence_score_ls.append(confidence_score)
        max_probability_ls.append(max_probability)
        perplexity_ls.append(perplexity)
        avg_entropy_ls.append(avg_entropy)
        energy_ls.append(energy)
        generated_scores_ls.append(generated_scores)
        token_probabilities_ls.append(token_probabilities)
        scaled_probabilities_ls.append(scaled_probabilities)
        generated_token_length_ls.append(generated_token_length)

    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_reserved())

    return (
        response,
        confidence_score_ls,
        max_probability_ls,
        perplexity_ls,
        avg_entropy_ls,
        energy_ls,
        generated_token_length_ls,
        per_item_inference_time,
        gen_out,
        enc
    )


# Define function to run format_and_run function for every batch in the dataloader
def process_batches_with_model(dataloader, role_instruction_text, prompt_text, pipe, tokenizer, messages, model, log_path, batch_size, dataset):
    """
    Runs the format_and_run function for every batch in the dataloader.

    Args:
        dataloader (dataloader): dataloader containing batches of radiology report data.
        role_instruction_text (str): Text containing the role instruction.
        prompt_text (str): Text containing the prompt.
        pipe (pipeline): Pipeline for generating text from the input data.
        tokenizer (tokenizer): Tokenizer for calculating performance metrics.
        messages (list): List of dicts containing the input for the LLM model.
        model (AutoModelForCausalLM): Model for calculating performance metrics.
        log_path (str): Path to the logs folder.
        dataset (dataset): dataset containing radiology reports.

    Returns:
        df (dataframe): dataframe containing the radiology reports, flags and generated response output from the pipeline.
        generated_token_length_list (list): List containing the token lengths for each output. 
        inference_time_list (list): List containing the inference times for each output.
        gen_out (transformers.modeling_outputs.ModelOutput): Output from the LLM containing response and scores.
        enc (transformers.tokenization_utils_base.BatchEncoding): For calculating performance metrics of the outputs.
    """

    try:
        gemmajson_outputs = []
        confidence_score_list = []
        max_probability_list = []
        perplexity_list = []
        avg_entropy_list = []
        energy_list = []
        generated_token_length_list = []
        inference_time_list = []

        for batch in tqdm(dataloader, desc="My bar!"):
            batch_results = [format_and_run_batch(role_instruction_text, prompt_text, batch, pipe, tokenizer, messages, model, log_path, batch_size)]

            for response, confidence_score_ls, max_probability_ls, perplexity_ls, avg_entropy_ls, energy_ls, generated_token_length_ls, per_item_inference_time, gen_out, enc in batch_results:
                gemmajson_outputs.extend(response)
                confidence_score_list.extend(confidence_score_ls)
                max_probability_list.extend(max_probability_ls)
                perplexity_list.extend(perplexity_ls)
                avg_entropy_list.extend(avg_entropy_ls)
                energy_list.extend(energy_ls)
                generated_token_length_list.extend(generated_token_length_ls)
                inference_time_list.extend(per_item_inference_time)

        df = dataset.df
        df['gemmajson'] = gemmajson_outputs
        df['Confidence_score'] = confidence_score_list
        df['Max_probability'] = max_probability_list
        df['Perplexity'] = perplexity_list
        df['Entropy'] = avg_entropy_list
        df['Energy'] = energy_list

        return df, generated_token_length_list, inference_time_list, gen_out, enc

    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"process_batches_with_model(): Error processing input text batches: {e}\n")
            traceback.print_exc()

        return "Error", "Error", "Error"

# Define function to extract output within the braces
def extract_brace_content(s, log_path):
    """
    Extracts the content from between the braces {}.

    Args:
        s (): String containg output text from pipeline.
        log_path (str): Path to the logs folder.

    Returns: 
        (str): All content between the braces.
    """

    try:
        start = s.find('{')
        end   = s.rfind('}') + 1

        return str(s[start:end])

        with open(log_path, "a") as log_file:
            log_file.write(f"extract_brace_content(): Content successfully extracted from between braces\n")

    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"extract_brace_content(): Error extracting brace content: {e}\n")

        return "Error extracting brace content"
    
# Define function to clean up extracted output and load as a json
def processmgoutput(outputtext, log_path):
    """
    Clean up the extracted output text and load it as a JSON.

    Args:
        outputtext (str): Extracted output text from pipeline.
        log_path (str): Path to the logs folder.

    Returns:
        jsondoc (json): JSON containing cleaned extracted output text from pipeline.
    """

    try:
        slicedtext  = extract_brace_content(str(outputtext), log_path)
        cleanedtext = slicedtext.replace("\\n", "").replace("    ", "")
        textstring = cleanedtext.replace("True","true").replace("False","false")
        with open(log_path, "a") as log_file:
            log_file.write(f"Cleaned Text for JSON input: {textstring}\n")

        jsondoc = json.loads(textstring)

        with open(log_path, "a") as log_file:
            log_file.write(f"processmgoutput(): Successfully generated JSON doc containing cleaned output from LLM\n")

        return jsondoc 

    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"processmgoutput(): Error cleaning text and loading as JSON: {e}\n")

        return "Error cleaning text and loading as JSON"

# Define function to return the the seperate features of the json relating to specific relevant information
def orchastorator_fordataprocessing(textstring, log_path):
    """
    Return the separate features of the JSON which relate to relevant feature information.

    Args:
        textstring(str): Cleaned extracted output from pipeline.
        log_path (str): Path to the logs folder.

    Returns:
        jsondoc (json): Updated jsondoc with column containing relevant feature information.
    """

    try:
        jsondoc = processmgoutput(textstring, log_path)
        return jsondoc['fracture_mentioned'], jsondoc['pathological_fracture'], jsondoc['evidence']['report_findings'], jsondoc['evidence']['rationale']

        with open(log_path, "a") as log_file:
            log_file.write(f"orchastorator_fordataprocessing(): Successfully added column containing all relevant feature information\n")

    except Exception as e:
        with open(log_path, "a") as log_file:
            log_file.write(f"orchastorator_fordataprocessing(): Error processing text: {e}\n")

        return "Error","Error","Error","Error"


def calculate_performance_metrics(gen_out, enc, batch_index):
    """
    Calculate the perfomance metrics to be logged and saved to MLFlow.

    Args:
        gen_out (transformers.modeling_outputs.ModelOutput): Output from the LLM containing response and scores.
        enc (transformers.tokenization_utils_base.BatchEncoding): For calculating performance metrics of the outputs.
        batch_index (int): index for the items in the batch of output responses.

    Returns:
        confidence_score (float): Final confidence score of the model.
        max_probability (float): The maximum value of the probability tokens.
        perplexity (float): Perplexity value of the model.
        avg_entropy (float): Entropy value of the model.
        energy (float): Energy value of the model.
        generated_scores (tuple): All generated logits of the model.
        token_probabilites (tuple): All token probabilities of the model.
        scaled_probabilities (tuple): All temperature scaled probabilities of the model.
        generated_token_lengths (int): Token length of the generated output.
    """

    generated_token_ids = gen_out.sequences[batch_index, enc.input_ids.shape[-1]:]

    # list of length = #generated tokens
    generated_scores = [scores[batch_index] for scores in gen_out.scores]

    # Calculate the probability for the specific token that was chosen at each step
    token_probabilities = []
    entropies = []
    for i, step_scores in enumerate(generated_scores):
        # Convert logits to a probability distribution using the softmax function
        step_probs = F.softmax(step_scores, dim=-1)

        # Get the ID of the token that was actually generated at this step
        generated_token_id = generated_token_ids[i]

        # Get the probability of that specific token from the distribution
        token_prob = step_probs[generated_token_id].item()
        token_probabilities.append(token_prob)
        
        # Compute entropy for this step (Shannon entropy)
        step_entropy = -torch.sum(step_probs * torch.log(step_probs + 1e-12)).item()
        entropies.append(step_entropy)

    # The final confidence score is the average of these probabilities
    if token_probabilities:
        confidence_score = sum(token_probabilities) / len(token_probabilities)
    else:
        confidence_score = 0

    # Maximum probability
    max_probability = np.mean([torch.max(F.softmax(step_scores, dim=-1)).item() for step_scores in generated_scores])

    # Perplexity (PPL)
    safe_probs = np.clip(token_probabilities, 1e-12, 1)
    perplexity = np.exp(-np.mean(np.log(safe_probs)))

    # Entropy 
    avg_entropy = np.mean(entropies)

    # Energy
    safe_probs = torch.tensor(token_probabilities).clamp(min=1e-12)
    energy = -torch.sum(torch.log(safe_probs)).item()

    # Temperature of medgemma4b model 
    T = 1.0

    # Temperature scaling
    scaled_probabilities = [F.softmax(logits / T, dim=-1) for logits in generated_scores]

    # Number of generated tokens
    prompt_len = int(enc.attention_mask[batch_index].sum().item())
    generated_token_ids = gen_out.sequences[batch_index, prompt_len:]
    generated_token_length = generated_token_ids.size(0)

    return confidence_score, max_probability, perplexity, avg_entropy, energy, generated_scores, token_probabilities, scaled_probabilities, generated_token_length