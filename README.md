# Tweet Sentiment Extraction - Group 15 Topic 4

This project implements a Deep Learning model based on **RoBERTa** to solve the Tweet Sentiment Extraction task. The objective is to extract the specific span of text from a tweet that supports a given sentiment (positive, negative, or neutral).

## Project Structure

- `62FIT4ATI_Group_15_Topic_4.ipynb`: The main Jupyter Notebook containing data loading, preprocessing, model definition, training loop (5-Fold Cross-Validation), and inference logic.
- `62FIT4ATI_Group_15_Topic_4.docx`: The written report analyzing results
- `62FIT4ATI_Group_15_Topic_4.pptx`: The presentation slide 

## Prerequisites

The code is designed to run in a Python environment, preferably **Google Colab** or a local Jupyter environment with GPU support.

### Dependencies
The following Python libraries are required:
- `torch`
- `transformers`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`

To install dependencies locally:
```bash
pip install torch transformers pandas numpy scikit-learn matplotlib seaborn tqdm
```

## Dataset Setup

The model requires the **Tweet Sentiment Extraction** dataset (commonly found on Kaggle). You need two files:

1.  **train.csv**: Must contain columns `textID`, `text`, `selected_text`, `sentiment`.
2.  **test.csv**: Must contain columns `textID`, `text`, `sentiment`.

**Note on File Paths:**
The notebook is configured to look for data files at the root directory (`/train.csv` and `/test.csv`).
- If using **Google Colab**, upload these files directly to the content root (the default folder when you open the file browser on the left).
- Alternatively, modify the `Config` class in the notebook to point to your specific file paths (e.g., inside Google Drive).

## Reproduction Instructions

### 1. Environment Setup
1. Open the notebook `62FIT4ATI_Group_15_Topic_4.ipynb`.
2. Ensure a GPU is available. In Colab: `Runtime` > `Change runtime type` > `T4 GPU` (or better).

### 2. Configuration
Locate the `Config` class in the notebook (Cell 2) to adjust parameters if necessary:
```python
class Config:
    SEED = 42
    MAX_LEN = 96
    BATCH_SIZE = 32
    EPOCHS = 4
    LEARNING_RATE = 3e-5
    MODEL_NAME = "roberta-base"
    TRAIN_FILE = "/train.csv"  # Update this path if your data is elsewhere
    TEST_FILE  = "/test.csv"   # Update this path if your data is elsewhere
```

### 3. Training
Run the notebook cells sequentially.
- **Data Loading**: Loads and visualizes the dataset distribution.
- **Preprocessing**: Tokenizes text using `roberta-base` and generates offset mappings for span alignment.
- **Model Training**: Executes a **5-Fold Cross-Validation** loop.
    - The model saves the best weights to `best_model.bin` based on the validation Jaccard score.
    - Training logs (Loss, Jaccard, Exact Match) are printed after every epoch.

### 4. Evaluation & Inference
After training, the notebook automatically:
- Runs a detailed evaluation calculating Word-level and Character-level Jaccard scores.
- Visualizes error distributions.
- Performs inference on 50 samples (5 manual examples + 45 from the test set).
- Saves the inference results to `inference_results_50_samples.csv`.

## Model Architecture

- **Backbone**: Pre-trained `roberta-base` from Hugging Face.
- **Head**: A linear layer (`nn.Linear(768, 2)`) predicting `start_logits` and `end_logits`.
- **Post-processing**: Implements a "Neutral Trick" (returns full text for neutral sentiment) and span decoding logic to ensure valid start/end indices.