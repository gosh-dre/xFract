# Import necessary libraries
import yaml
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json
import mlflow

# Define function to load config file
def load_config(config_path):
    """
    Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        config (dict): Parsed configuration as a dictionary.
    """

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config

# Define custom dataset class for using dataloader for input data
class RadiologyDataset(Dataset):
    """
    Custom Pytorch Dataset class for loading text strings from a CSV file.

    Defines a class to read the csv file, load the data from the "Synthetic" column, extract the number of radiology reports in the data and separate the csv file into individual reports.

    Parameters
    ----------
    radiology_report_data_path: str or Path
        String or Path pointing to CSV file containing the radiology reports and flags.
    """

    def __init__(self, input_path):
        """
        Load and read the CSV file and extract the data from the "Synthetic" column.

        Parameters
        ----------
        radiology_report_data_path: str or Path 
            String or Path pointing to CSV file containing the radiology reports and flags.
        """

        self.df = pd.read_csv(input_path)
        self.narrative = self.df["narrative"]
        self.data = list(zip(self.narrative))

    def __len__(self):
        """
        Return the number of reports in the CSV file.

        Returns
        -------
        int
            The number of reports in the CSV file.
        """

        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the report from the data at the given index.

        Parameters
        ----------
        idx: int
            The index representing each report to retrieve from the data.

        Returns
        -------
        str
            The report for each given index of the data.
        """

        i = self.data[idx]
        return {
            "narrative": i,
        }

# Define function to load role instruction and prompt for MedGemma model from json if mlflow disabled
def load_prompt_json(role_instruction_path, prompt_path):
    """
    Loads the role instruction and prompt texts and version numbers from the JSON files.

    Args:
        role_instruction_path (str): Path to the role instruction JSON file.
        prompt_path (str): Path to the prompt JSON file.

    Returns:
        role_instruction_text (str): Text from the role instruction JSON file.
        role_instruction_version (int): Role instruction version number.
        prompt_text (str): Text from the prompt JSON file.
        prompt_version (int): Prompt version number.
    """

    with open(role_instruction_path, "r") as f:
        role_instruction_data = json.load(f)

    role_instruction_text = role_instruction_data["prompt"]
    role_instruction_version = role_instruction_data["version"]

    with open(prompt_path, "r") as f:
        prompt_data = json.load(f)

    prompt_text = prompt_data["prompt"]
    prompt_version = prompt_data["version"]

    return role_instruction_text, role_instruction_version, prompt_text, prompt_version

# Define function to load role instruction and prompt from MLflow and extract the text and version numbers
def load_prompt_mlflow(mlflow_role_instruction_path, mlflow_prompt_path):
    """
    Loads the role instruction and prompt texts and version numbers from MLflow.

    Args:
        mlflow_role_instruction_path (str): Path to the role instruction MLflow.
        mlflow_prompt_path (str): Path to the prompt MLflow.

    Returns:
        role_instruction_text (str): Text from the role instruction in MLflow.
        prompt_text (str): Text from the prompt in MLflow.
        role_instruction_version (int): Role instruction version number.
        prompt_version (int): Prompt version number.
    """

    # Load role instructions and prompt for MedGemma model from mlflow if enabled
    mlflow_role_instruction = mlflow.load_prompt(name_or_uri=mlflow_role_instruction_path)
    mlflow_prompt = mlflow.load_prompt(name_or_uri=mlflow_prompt_path)

    # Extract text from prompts
    role_instruction_text = mlflow_role_instruction.format()
    prompt_text = mlflow_prompt.format()

    # Log role_instructions and prompt versions
    role_instruction_version = float(mlflow_role_instruction_path.split("/")[-1])
    prompt_version = float(mlflow_prompt_path.split("/")[-1])

    return role_instruction_text, prompt_text, role_instruction_version, prompt_version


def load_NLI_prompt_mlflow(mlflow_NLI_prompt_path):
    """
    Loads the role instruction and prompt text and version number from MLflow.

    Args:
        mlflow_NLI_prompt_path (str): Path to the NLI prompt MLflow.

    Returns:
        NLI_prompt_text (str): Text from the prompt in MLflow.
        NLI_prompt_version (int): Prompt version number.
    """

    # Load prompt for MediPhi-Clinical model from mlflow if enabled
    mlflow_NLI_prompt = mlflow.load_prompt(name_or_uri=mlflow_NLI_prompt_path)

    # Extract text from prompt
    NLI_prompt_text = mlflow_NLI_prompt.format()

    # Extract prompt version
    NLI_prompt_version = float(mlflow_NLI_prompt_path.split("/")[-1])

    return NLI_prompt_text, NLI_prompt_version


def load_NLI_prompt_json(NLI_prompt_path):
    """
    Loads the NLI prompt text and version number from the JSON file.

    Args:
        NLI_prompt_path (str): Path to the NLI prompt JSON file.

    Returns:
        NLI_prompt_text (str): Text from the prompt JSON file.
        NLI_prompt_version (int): Prompt version number.
    """

    with open(NLI_prompt_path, "r") as f:
        NLI_prompt_data = json.load(f)

    NLI_prompt_text = NLI_prompt_data["prompt"]
    NLI_prompt_version = NLI_prompt_data["version"]

    return NLI_prompt_text, NLI_prompt_version