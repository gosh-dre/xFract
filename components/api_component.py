import requests
from requests_kerberos import HTTPKerberosAuth
import pandas as pd
import ast
import os
import sys

def connect_api(input_path, log_path):
    """
    Connects to the API to retrieve input data, converts the .txt data to .csv format and saves the csv data to be passed into 
    the pipeline.

    Args:
        input_path (str): Path to where the input data from the API will be saved.
        log_path (str): Path to the log file.
    """
    
    r = requests.get("<ADD YOUR API URL HERE>", auth=HTTPKerberosAuth())

    with open(log_path, "a") as log_file:
        log_file.write(f"Packages loaded succesfully!\n")
        log_file.write(f"STATUS: $ {r.status_code}\n")      # Shows HTTP status (should be 200)

    if r.status_code == 200:
        with open(log_path, "a") as log_file:
            log_file.write(f"CONTENT:\n {r.text}")      # Shows the HTML/text content
        
        # Convert response text to DataFrame
        data = ast.literal_eval(r.text)  # Use json.loads if it's valid JSON
        df = pd.DataFrame(data)
        print(df.head())  # Show first few rows

        # Ensure destination folder exists
        os.makedirs(os.path.dirname(input_path), exist_ok=True)

        # Save as csv to input folder
        df.to_csv(input_path, index=False)

    else:
        print(f"ERROR: Failed to fetch data: {r.status_code}\n")
        sys.exit(1) 
