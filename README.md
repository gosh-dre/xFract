# xFract (V1.0.0)

## 🚀 Introduction
 
This repository allows the prediction of pathological fractures through the integration of the pre-trained deep learning model MedGemma4b. It works by taking radiology reports as an input and generating report findings and rationales to justify the prediction of the presence of pathological fractures. This pipeline also provides confidence scores, perplexities, entropies and an NLI component to verify the output of the model.

## Pipeline description

### 1. Dataset



CSV input data should be organized in the following way:

```
narrative,Flag
"22/09/2025, 11:40, XR FEMUR LEFT 22/09/2025, 11:32, XR TIBIA AND FIBULA LEFT

Case History:

Reason for Request: New onset pain / guarding/ resisting movement of left leg - painful on examination of lower tibia and ankle (left)

? Fracture, New onset pain / guarding/ resisting movement of left leg - painful on examination of lower tibia and ankle (left) ? Fracture From clinical notes: Osteogenesis imperfecta type III. Secondary hyperparathyroidism. Previous multiple fractures. Chronic kidney disease stage 4, Vitamin D deficiency, Mobility impairment.

Findings: There is diffuse osteopenia in keeping with known hyperparathyroidism. There are transverse fractures of the distal left femur, proximal and distal left tibia and fibula, with some adjacent periosteal reaction but minimal displacement. These represents pathological fractures which occurred several days ago, in a patient with pre-existing metabolic bone disease. An orthopaedic opinion regarding alignment should be considered. Dr Rebecca Thompson, Consultant Radiologist 22/09/2025, 13:15",TRUE
```

### 2. Models

This repository requires two pre-trained models: MedGemma4b LLM for generating output predictions and MediPhi-Clinical NLI for verifying the output of the LLM. Models can be loaded both locally from a named volume.

### 3. Results 

All results are saved in the ```output/``` folder in the container and either the ```output/``` folder in the named volume or in the ```output/``` folder in the experiment run in mlflow. After running the pipeline the folders will contain:

- ```intermediate_results.csv``` For each report, the following information is saved: gemmajson, containing the JSON formatted output of the MedGemma4b model; confidence score in the prediction; max probability; perplexity; entropy; energy
 
- ```final_results.csv``` Contains all the data from the intermediate_results.csv but also the gemmajson data is split into fracture_mentioned, pathological_fracture, report_findings

- ```radiology_qi_nli.csv``` For each report, it contains the prediction; report_findings with the corresponding NLI Answer, Score and Rationale; Medgemma prediction_rationale with the corresponding NLI Answer, Score and Rationale

- ```config.yaml``` A copy of the config file which contains the paths to the inputs, models, outputs, prompts, mlflow and a variable to control the batch size for the data loader. 

- ```QI_pathological_fracture.log``` Contains all the details of the run including formatted inputs, outputs, inference times and possibly errors.

## Components
### API component
The API component pulls the input data from the API and saves it as a csv as well as loading it as a pandas dataframe. 

### Data Ingestion
Firstly loads the config file which contains the paths to the logs, input and output folder as well as paths to the mounted volume and MLFlow tracking server. It also contains the variable for the batch size for parallel processing. The data pulled from the API is then handled to extract the "narrative" column to be used inside the prompt for the LLM. It  also loads the role instructions and prompt from either the JSON files in the prompts folder or directly from the MLflow server.

### Data Processing
Loads the MedGemma LLM model from the mounted volume for generating outputs and the tokenizer for calculating performance metrics. It then processes the radiology reports in parallel using H100 GPUs and a data loader and calculates performance metrics, inference times and token lengths for each output. The MedGemma outputs are then cleaned to ensure that they have a valid JSON structure and are saved into a structured pandas dataframe including the initial output, classification of pathological fracture, report findings, rationale and performance metrics.

### Results inferencing
The NLI model mediphi is loaded from the mounted volume along with it's tokenizer. The report findings are then passed into the model along with the NLI role instructions, NLI prompt and narrative from the original radiology report and then the NLI outputs whether or not the MedGemma report findings logically follow from the radiology report. The same process is then done for the MedGemma rationale, verifying the validity of the MedGemma rationale against the original radiology report. The NLI outputs are then combined into one dataframe containing whether or not the report findings and rationale entail, contradict or are neutral to the original report along with a confidence score for each output.

### Data Storing
This component handles saving the inputs, outputs, logs, config, inference times, token lengths and metrics to either MLflow or to the mounted volume. 

### UI
The UI folder currently exists only as a representative example due to our information governance situation and so is not included in the pipeline.

## 📝 Requirements
```bash
1. git
2. podman
```
Use pip to install the above.


## 📚 Installations

### Run using podman

1. Clone the repository
2. Use the Dockerfile script in the development server to create the container image

```
podman build -t pathological-fractures .
```

3. Create .tar image in dev server:
```
podman save -o pathological-fractures.tar localhost/pathological_fractures:latest
```
4. Create alpine image in dev server for copying models into the volume later 
```
podman pull docker.io/alpine:latest
```
5. Create .tar image for alpine in dev server
```
podman save -o alpine.tar docker.io/alpine:latest
```
4. scp files into staging server
```
 scp {username}@{dev_server_IP}:~{path_to_tar_file} .
```
5. Load container image
```
podman load -i pathological_fractures.tar
```
6. Load alpine image
```
podman load -i alpine.tar
```
6. Create a named volume
```
podman volume create pathological-fractures
```
7. Copy models from drebulkstorage in staging 03 into your named volume

```
podman run --rm \
  -v /drebulkstorage/babygrams/models/medgemma4b:/src/medgemma4b:ro \
  -v /drebulkstorage/MediPhi-Clinical:/src/MediPhi-Clinical:ro \
  -v pathological-fractures:/data \
  alpine sh -c "cp -r /src/medgemma4b /data && cp -r /src/MediPhi-Clinical /data"
```


### Usage

**Environment variables**

- ```MLFLOW_TRACKING_ENABLED```: When set to true will save all logs and outputs to mflow. When false, will save to a new numbered experiment folder in the outputs in the mounted volume

### Opening  the container

1. Setting ```MLFLOW_TRACKING_ENABLED=True``` with GPU available (IDEAL):

```
podman run --rm --network=host --device nvidia.com/gpu=all -v pathological-fractures:/data -e MLFLOW_TRACKING_ENABLED=true -it --entrypoint /bin/bash pathological-fractures
```

2. Setting ```MLFLOW_TRACKING_ENABLED=True``` with GPU unavailable:

```
podman run --rm --network=host -v pathological-fractures:/data -e MLFLOW_TRACKING_ENABLED=true -it --entrypoint /bin/bash pathological-fractures
```
3. Setting ```MLFLOW_TRACKING_ENABLED=False``` with GPU available:
```
podman run --rm --device nvidia.com/gpu=all -v pathological-fractures:/data -it --entrypoint /bin/bash pathological-fractures
```
4. Setting ```MLFLOW_TRACKING_ENABLED=False``` with GPU unavailable:
```
podman run --rm -v pathological-fractures:/data -it --entrypoint /bin/bash pathological-fractures
```
5. To use the API add this environmental variable to your run command after mounting the local volume:
```
-e API_ENABLED=true
```

### Running the container (Steps 1 and 2 for API use only):
1. Enter username for API (OPTIONAL):
```
KRB5_CONFIG=krb5.conf kinit user_name
```

2. Enter password for API (OPTIONAL):

3. Run pipeline:
```
python main.py
```

##  🤝 Contributors

* Ryan Howard - ML Scientist @ GOSH DRIVE (Core Contributor)
* Sebin Sabu - Senior NLP Specialist @ GOSH DRIVE (Core Contributor & Technical Lead)
* Dr. Pavithra Rajendran - NLP & Computer Vision Lead @ GOSH DRIVE (Project Lead)

##  🤝 Acknowledgements
* DRIVE engineering team
* Alexander Chesover (Clinical Lead)
* Nuwanthi Yapa Mahathanthila (QI Project Lead)

## Citing & Authors

If you find this repository helpful, feel free to cite our publication (to be updated):

```
@inproceedings{
    
}
```

### 📃 Licenses

Code in this repository is covered by the MIT License and for all documentation the [Open Government License (OGL)](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) is used.

Copyright (c) 2024 Crown Copyright
