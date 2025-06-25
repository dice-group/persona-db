# persona-db
This repository host the implementation of the persona databases project

## Setup Instructions

1. **Clone the repository and change to its working directory:**
    ```bash
    git clone https://github.com/dice-group/persona-db.git
    cd persona-db

2. Create and activate a Conda environment:
    ```bash
    conda create -n .venv python=3.11.13 --no-default-packages
    conda activate .venv
   
3. Install pip and dependencies:
    ```bash 
    conda install pip3
    pip3 install -r requirements.txt

4. Configure your model and dataset split:
    - Open `config.py`.
      - Set MODEL_NAME to the path of your quantized model.
      - Set DATASET_SPLIT to the portion of the dataset you want to run. This helps avoid reprocessing data.
    - Examples:
      - DATASET_SPLIT = "train"           # Runs on the entire training set
      - DATASET_SPLIT = "train[:100]"     # Runs on the first 100 samples only
      - DATASET_SPLIT = "train[101:1001]" # Runs from sample 101 to 1000

5. Run the main script:
    ```bash
    python3 main.py