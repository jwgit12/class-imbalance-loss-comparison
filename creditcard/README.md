# Credit Card Fraud Detection Experiments


This repository demonstrates training multiple MLP models on the **Credit Card Fraud dataset** from Kaggle. The experiments explore:

- Different **model sizes** (small, medium, large)  
- Various **loss functions**: L2, Binary Cross-Entropy (CE), Weighted CE (WCE), Focal Loss  
- Different **dataset imbalance levels**  
- Logging and tracking with **MLflow**

---

## Quick Start
#### Using `uv` and install dependencies in .venv

#### Using `mlflow` and make sure it's running. 
Insert the correct _PORT_ in `experiments_config.yaml` -> `mlflow_port: [PORT]`

#### Using `kaggle` **Credit Card Fraud dataset**:
1. Get Kaggle API credentials:
   1. Go to Kaggle Account → API → “Create New API Token” 
   2. This downloads `kaggle.json`. 
   3. Place kaggle.json in:
      - **Windows**: `C:\Users\<USERNAME>\.kaggle\kaggle.json` or
      - **Linux/macOS**: `~/.kaggle/kaggle.json`
   4. Or set environment variables:
       ```bash
       # Windows PowerShell
       setx KAGGLE_USERNAME "your_username"
       setx KAGGLE_KEY "your_key"
        
       # Linux/macOS
       export KAGGLE_USERNAME="your_username"
       export KAGGLE_KEY="your_key"
       ```
5. Download and **unzip** the dataset:
    ```bash
    kaggle datasets download -d mlg-ulb/creditcardfraud -p data
    ```
After this, data/creditcard.csv is ready for use.

