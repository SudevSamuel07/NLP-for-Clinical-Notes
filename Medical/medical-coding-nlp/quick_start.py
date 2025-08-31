"""
Quick Start Script for Medical Coding System

This script provides a simple way to get started with the medical coding system.
It demonstrates the core functionality without complex setup.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure we're in the right directory
project_root = Path(__file__).parent
os.chdir(project_root)

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    packages = [
        'pandas', 'numpy', 'scikit-learn', 'spacy', 'nltk', 
        'flask', 'matplotlib', 'tqdm', 'joblib'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} already installed")
        except ImportError:
            print(f"Installing {package}...")
            os.system(f"pip install {package}")
    
    # Download spaCy model
    print("Downloading spaCy English model...")
    os.system("python -m spacy download en_core_web_sm")

def create_sample_data():
    """Create sample clinical notes data"""
    print("Creating sample clinical notes dataset...")
    
    # Sample clinical notes with ICD/CPT codes
    sample_notes = [
        {
            'text': "Patient is a 45-year-old male with a history of type 2 diabetes mellitus without complications. He presents today for routine follow-up. Blood glucose levels have been well controlled on metformin. Physical examination is unremarkable. Will continue current diabetes management.",
            'icd_codes': ['E11.9'],  # Type 2 diabetes without complications
            'cpt_codes': ['99213']   # Office visit
        },
        {
            'text': "67-year-old female with essential hypertension presents for blood pressure check. Current BP is 145/92. Patient has been compliant with lisinopril therapy. Will increase dosage and schedule follow-up in 4 weeks.",
            'icd_codes': ['I10'],    # Essential hypertension
            'cpt_codes': ['99213']   # Office visit
        },
        {
            'text': "35-year-old pregnant female at 32 weeks gestation presents with shortness of breath. Physical exam reveals mild lower extremity edema. Fetal heart tones are reassuring. Will monitor closely.",
            'icd_codes': ['O26.89'], # Other specified pregnancy-related conditions
            'cpt_codes': ['99213']   # Office visit
        },
        {
            'text': "Patient has chronic obstructive pulmonary disease with acute exacerbation. Presents with increased dyspnea and productive cough. Will start prednisone and antibiotics.",
            'icd_codes': ['J44.1'],  # COPD with acute exacerbation
            'cpt_codes': ['99214']   # Office visit, higher complexity
        },
        {
            'text': "28-year-old male presents with acute appendicitis. Physical exam shows McBurney's point tenderness. Patient scheduled for laparoscopic appendectomy.",
            'icd_codes': ['K35.9'],  # Acute appendicitis
            'cpt_codes': ['44970']   # Laparoscopic appendectomy
        },
        {
            'text': "Patient with coronary artery disease presents with chest pain. EKG shows ST changes. Cardiac enzymes are elevated. Diagnosed with myocardial infarction.",
            'icd_codes': ['I21.9'],  # Acute myocardial infarction
            'cpt_codes': ['99232', '93000']  # Hospital care, EKG
        },
        {
            'text': "65-year-old female with pneumonia. Chest X-ray shows right lower lobe infiltrate. Started on antibiotic therapy.",
            'icd_codes': ['J18.9'],  # Pneumonia, unspecified
            'cpt_codes': ['99214', '71020']  # Office visit, chest X-ray
        },
        {
            'text': "Patient presents for routine colonoscopy screening. Procedure completed without complications. No polyps found.",
            'icd_codes': ['Z12.11'], # Encounter for screening for malignant neoplasm of colon
            'cpt_codes': ['45378']   # Colonoscopy
        }
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_notes)
    
    # Save to data directory
    data_dir = project_root / 'data' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(data_dir / 'sample_clinical_notes.csv', index=False)
    
    print(f"Created sample dataset with {len(df)} clinical notes")
    return df

def create_code_mappings():
    """Create ICD-10 and CPT code mappings"""
    print("Creating code mappings...")
    
    # ICD-10 code mappings
    icd10_codes = {
        'E11.9': 'Type 2 diabetes mellitus without complications',
        'I10': 'Essential hypertension',
        'O26.89': 'Other specified pregnancy-related conditions',
        'J44.1': 'Chronic obstructive pulmonary disease with acute exacerbation',
        'K35.9': 'Acute appendicitis, unspecified',
        'I21.9': 'Acute myocardial infarction, unspecified',
        'J18.9': 'Pneumonia, unspecified organism',
        'Z12.11': 'Encounter for screening for malignant neoplasm of colon',
        'E78.5': 'Hyperlipidemia, unspecified',
        'M79.3': 'Panniculitis, unspecified'
    }
    
    # CPT code mappings
    cpt_codes = {
        '99213': 'Office or other outpatient visit for the evaluation and management of an established patient, typically 15 minutes',
        '99214': 'Office or other outpatient visit for the evaluation and management of an established patient, typically 25 minutes',
        '99232': 'Subsequent hospital care, per day, for the evaluation and management of a patient',
        '93000': 'Electrocardiogram, routine ECG with at least 12 leads',
        '71020': 'Radiologic examination, chest, 2 views, frontal and lateral',
        '44970': 'Laparoscopy, surgical, appendectomy',
        '45378': 'Colonoscopy, flexible; diagnostic, including collection of specimen(s)',
        '80053': 'Comprehensive metabolic panel',
        '85025': 'Blood count; complete (CBC), automated',
        '36415': 'Collection of venous blood by venipuncture'
    }
    
    code_mappings = {
        'icd10': icd10_codes,
        'cpt': cpt_codes
    }
    
    # Save mappings
    data_dir = project_root / 'data' / 'processed'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    with open(data_dir / 'code_mappings.json', 'w') as f:
        json.dump(code_mappings, f, indent=2)
    
    print(f"Created mappings for {len(icd10_codes)} ICD codes and {len(cpt_codes)} CPT codes")
    return code_mappings

def simple_text_processing(text):
    """Simple text preprocessing"""
    import re
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Basic medical abbreviation expansion
    abbreviations = {
        'dm': 'diabetes mellitus',
        'htn': 'hypertension',
        'cad': 'coronary artery disease',
        'chf': 'congestive heart failure',
        'copd': 'chronic obstructive pulmonary disease',
        'mi': 'myocardial infarction',
        'bp': 'blood pressure',
        'sob': 'shortness of breath'
    }
    
    for abbrev, expansion in abbreviations.items():
        text = text.replace(f' {abbrev} ', f' {expansion} ')
        text = text.replace(f' {abbrev}.', f' {expansion}')
    
    return text.strip()

def extract_simple_features(texts):
    """Simple feature extraction using TF-IDF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Medical-specific stopwords
    medical_stopwords = [
        'patient', 'presents', 'history', 'examination', 'diagnosis',
        'treatment', 'therapy', 'medical', 'clinical', 'follow', 'visit'
    ]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words=medical_stopwords,
        min_df=1
    )
    
    # Fit and transform texts
    features = vectorizer.fit_transform(texts)
    
    return features, vectorizer

def train_simple_model(X, y_icd, y_cpt):
    """Train simple models for ICD and CPT prediction"""
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MultiLabelBinarizer
    
    print("Training models...")
    
    # Prepare labels
    icd_binarizer = MultiLabelBinarizer()
    cpt_binarizer = MultiLabelBinarizer()
    
    y_icd_binary = icd_binarizer.fit_transform(y_icd)
    y_cpt_binary = cpt_binarizer.fit_transform(y_cpt)
    
    # Train models
    icd_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42))
    cpt_model = MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42))
    
    icd_model.fit(X, y_icd_binary)
    cpt_model.fit(X, y_cpt_binary)
    
    return {
        'icd_model': icd_model,
        'cpt_model': cpt_model,
        'icd_binarizer': icd_binarizer,
        'cpt_binarizer': cpt_binarizer
    }

def predict_codes(text, models, vectorizer, code_mappings, top_k=3):
    """Predict ICD and CPT codes for a given text"""
    
    # Preprocess text
    processed_text = simple_text_processing(text)
    
    # Extract features
    features = vectorizer.transform([processed_text])
    
    # Make predictions
    icd_pred = models['icd_model'].predict(features)
    cpt_pred = models['cpt_model'].predict(features)
    
    # Get predicted codes
    icd_codes = models['icd_binarizer'].inverse_transform(icd_pred)[0]
    cpt_codes = models['cpt_binarizer'].inverse_transform(cpt_pred)[0]
    
    # Get descriptions
    icd_descriptions = [code_mappings['icd10'].get(code, 'Description not available') for code in icd_codes]
    cpt_descriptions = [code_mappings['cpt'].get(code, 'Description not available') for code in cpt_codes]
    
    return {
        'icd_codes': list(icd_codes)[:top_k],
        'cpt_codes': list(cpt_codes)[:top_k],
        'icd_descriptions': icd_descriptions[:top_k],
        'cpt_descriptions': cpt_descriptions[:top_k]
    }

def demo_prediction():
    """Demonstrate the prediction system"""
    print("\n" + "="*60)
    print("MEDICAL CODING SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Load or create data
    data_file = project_root / 'data' / 'raw' / 'sample_clinical_notes.csv'
    if data_file.exists():
        df = pd.read_csv(data_file)
        print(f"Loaded existing dataset with {len(df)} notes")
    else:
        df = create_sample_data()
    
    # Create or load code mappings
    mappings_file = project_root / 'data' / 'processed' / 'code_mappings.json'
    if mappings_file.exists():
        with open(mappings_file, 'r') as f:
            code_mappings = json.load(f)
        print("Loaded existing code mappings")
    else:
        code_mappings = create_code_mappings()
    
    # Prepare data
    texts = [simple_text_processing(text) for text in df['text']]
    
    # Parse code lists (handle string format)
    y_icd = []
    y_cpt = []
    
    for _, row in df.iterrows():
        if isinstance(row['icd_codes'], str):
            icd = eval(row['icd_codes'])  # Convert string representation to list
        else:
            icd = row['icd_codes']
        
        if isinstance(row['cpt_codes'], str):
            cpt = eval(row['cpt_codes'])
        else:
            cpt = row['cpt_codes']
        
        y_icd.append(icd)
        y_cpt.append(cpt)
    
    # Extract features
    print("Extracting features...")
    X, vectorizer = extract_simple_features(texts)
    
    # Train models
    models = train_simple_model(X, y_icd, y_cpt)
    
    # Demo with sample text
    sample_text = """
    Patient is a 55-year-old female with a history of hypertension and diabetes.
    She presents with chest pain and shortness of breath. Physical examination
    reveals elevated blood pressure. EKG shows atrial fibrillation. Will start
    anticoagulation therapy.
    """
    
    print(f"\nSample Clinical Note:")
    print(sample_text.strip())
    
    # Make prediction
    prediction = predict_codes(sample_text, models, vectorizer, code_mappings)
    
    print(f"\nPredicted ICD-10 Codes:")
    for i, (code, desc) in enumerate(zip(prediction['icd_codes'], prediction['icd_descriptions']), 1):
        print(f"  {i}. {code} - {desc}")
    
    print(f"\nPredicted CPT Codes:")
    for i, (code, desc) in enumerate(zip(prediction['cpt_codes'], prediction['cpt_descriptions']), 1):
        print(f"  {i}. {code} - {desc}")
    
    # Interactive demo
    print(f"\n" + "="*60)
    print("INTERACTIVE DEMO")
    print("="*60)
    print("Enter your own clinical note (or press Enter to skip):")
    
    user_input = input().strip()
    if user_input:
        prediction = predict_codes(user_input, models, vectorizer, code_mappings)
        
        print(f"\nYour Clinical Note:")
        print(user_input)
        
        print(f"\nPredicted ICD-10 Codes:")
        for i, (code, desc) in enumerate(zip(prediction['icd_codes'], prediction['icd_descriptions']), 1):
            print(f"  {i}. {code} - {desc}")
        
        print(f"\nPredicted CPT Codes:")
        for i, (code, desc) in enumerate(zip(prediction['cpt_codes'], prediction['cpt_descriptions']), 1):
            print(f"  {i}. {code} - {desc}")
    
    print(f"\n" + "="*60)
    print("Demo completed!")
    print("="*60)

def main():
    """Main function"""
    print("Medical Coding System - Quick Start")
    print("="*40)
    
    # Check if we need to install packages
    try:
        import pandas, numpy, sklearn
        print("✓ Required packages are available")
    except ImportError:
        install_requirements()
    
    # Create necessary directories
    directories = ['data/raw', 'data/processed', 'models', 'logs']
    for directory in directories:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
    
    # Run demonstration
    demo_prediction()
    
    print("\nTo run the full system, use:")
    print("python main.py all")

if __name__ == "__main__":
    main()
