# Medical Notes ICD/CPT Auto-Coding System
## Overview
This project implements an NLP-based system for automatically suggesting ICD-10 diagnosis codes and CPT procedure codes from clinical notes. The system uses advanced natural language processing techniques to extract medical entities and map them to standardized coding systems.

   
       
## Features
- **Clinical Note Processing**: Parse and preprocess free-text medical notes
- **Medical Entity Recognition**: Extract medical conditions, procedures, and symptoms
- **ICD-10 Code Prediction**: Suggest top 3 most relevant ICD-10 diagnosis codes
- **CPT Code Prediction**: Suggest top 3 most relevant CPT procedure codes
- **Accuracy Evaluation**: Comprehensive evaluation metrics including precision, recall, and F1-score
- **Web Interface**: User-friendly interface for real-time code suggestions

## Technology Stack
- **Python 3.8+**
- **spaCy**: Medical NLP and entity recognition
- **scikit-learn**: Machine learning models
- **Transformers (Hugging Face)**: Pre-trained language models
- **pandas**: Data manipulation
- **Flask**: Web interface
- **matplotlib/seaborn**: Visualization
## Website
<img width="1894" height="767" alt="image" src="https://github.com/user-attachments/assets/02e37026-cbcf-4b51-9df9-b1195878934c" />

<img width="1414" height="505" alt="image" src="https://github.com/user-attachments/assets/291750ca-ba98-4fce-baed-19c9c2e172cf" />

## Project Structure
```
medical-coding-nlp/
├── src/
│   ├── data_preprocessing.py    # Data cleaning and preprocessing
│   ├── feature_extraction.py    # Text feature extraction
│   ├── model_training.py        # Model training pipeline
│   ├── prediction.py           # Code prediction logic
│   ├── evaluation.py           # Model evaluation metrics
│   └── web_app.py              # Flask web application
├── data/
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Cleaned datasets
│   └── reference/              # ICD-10 and CPT code references
├── models/                     # Trained models
├── notebooks/                  # Jupyter notebooks for analysis
├── utils/                      # Utility functions
├── tests/                      # Unit tests
└── requirements.txt            # Dependencies

```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd medical-coding-nlp
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy medical model:
```bash
python -m spacy download en_core_web_sm
```

## Data Sources
- **MIMIC-III Clinical Notes**: De-identified clinical notes dataset
- **ICD-10-CM Official Guidelines**: Diagnosis codes reference
- **CPT Code Database**: Procedure codes reference
- **Medical Text Datasets from Kaggle**: Additional training data

## Usage

### 1. Data Preprocessing
```bash
python src/data_preprocessing.py
```

### 2. Train Models
```bash
python src/model_training.py
```

### 3. Run Predictions
```bash
python src/prediction.py --input "Patient presents with Type 2 diabetes mellitus without complications..."
```

### 4. Start Web Interface
```bash
python src/web_app.py
```

## Model Performance
- **ICD-10 Accuracy**: ~85% top-3 accuracy
- **CPT Accuracy**: ~82% top-3 accuracy
- **Processing Speed**: <2 seconds per note

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License
This project is for educational purposes. Please ensure compliance with healthcare data regulations (HIPAA, etc.) when using with real patient data.



