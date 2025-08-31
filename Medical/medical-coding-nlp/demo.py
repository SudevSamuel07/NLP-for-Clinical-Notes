"""
Medical Coding System - Working Demo

This script provides a complete working demonstration of the medical coding system.
It includes a simple but effective implementation that actually works.
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MedicalCodingSystem:
    """Complete medical coding system implementation"""
    
    def __init__(self):
        self.vectorizer = None
        self.icd_model = None
        self.cpt_model = None
        self.icd_binarizer = MultiLabelBinarizer()
        self.cpt_binarizer = MultiLabelBinarizer()
        self.code_mappings = None
        self.is_trained = False
        
        # Medical abbreviations for text preprocessing
        self.abbreviations = {
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'mi': 'myocardial infarction',
            'bp': 'blood pressure',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'ekg': 'electrocardiogram',
            'ecg': 'electrocardiogram'
        }
    
    def preprocess_text(self, text):
        """Preprocess clinical text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Expand medical abbreviations
        for abbrev, expansion in self.abbreviations.items():
            text = text.replace(f' {abbrev} ', f' {expansion} ')
            text = text.replace(f' {abbrev}.', f' {expansion}')
            text = text.replace(f'{abbrev},', f'{expansion},')
        
        return text.strip()
    
    def prepare_data(self, df):
        """Prepare training data"""
        # Preprocess texts
        texts = [self.preprocess_text(text) for text in df['text']]
        
        # Parse code lists
        icd_codes = []
        cpt_codes = []
        
        for _, row in df.iterrows():
            # Handle string representation of lists
            if isinstance(row['icd_codes'], str):
                icd = eval(row['icd_codes'])
            else:
                icd = row['icd_codes']
            
            if isinstance(row['cpt_codes'], str):
                cpt = eval(row['cpt_codes'])
            else:
                cpt = row['cpt_codes']
            
            icd_codes.append(icd)
            cpt_codes.append(cpt)
        
        return texts, icd_codes, cpt_codes
    
    def train(self, df, code_mappings):
        """Train the medical coding models"""
        print("Training medical coding models...")
        
        self.code_mappings = code_mappings
        
        # Prepare data
        texts, icd_codes, cpt_codes = self.prepare_data(df)
        
        print(f"Training on {len(texts)} clinical notes...")
        
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            min_df=1,
            stop_words=['patient', 'presents', 'history', 'examination']
        )
        
        X = self.vectorizer.fit_transform(texts)
        
        # Prepare labels
        y_icd = self.icd_binarizer.fit_transform(icd_codes)
        y_cpt = self.cpt_binarizer.fit_transform(cpt_codes)
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"ICD label matrix shape: {y_icd.shape}")
        print(f"CPT label matrix shape: {y_cpt.shape}")
        
        # Train models
        self.icd_model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.cpt_model = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        
        print("Training ICD model...")
        self.icd_model.fit(X, y_icd)
        
        print("Training CPT model...")
        self.cpt_model.fit(X, y_cpt)
        
        self.is_trained = True
        print("Training completed!")
        
        # Evaluate on training data (quick check)
        icd_pred = self.icd_model.predict(X)
        cpt_pred = self.cpt_model.predict(X)
        
        icd_accuracy = accuracy_score(y_icd, icd_pred)
        cpt_accuracy = accuracy_score(y_cpt, cpt_pred)
        
        print(f"Training ICD accuracy: {icd_accuracy:.3f}")
        print(f"Training CPT accuracy: {cpt_accuracy:.3f}")
    
    def predict(self, text, top_k=3):
        """Predict ICD and CPT codes for clinical text"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract features
        X = self.vectorizer.transform([processed_text])
        
        # Get prediction probabilities for each class
        icd_probabilities = []
        cpt_probabilities = []
        
        # Get probabilities for each ICD code
        for estimator in self.icd_model.estimators_:
            probs = estimator.predict_proba(X)[0]
            if len(probs) > 1:  # Binary classification returns [prob_0, prob_1]
                icd_probabilities.append(probs[1])  # Probability of positive class
            else:
                icd_probabilities.append(0.0)
        
        # Get probabilities for each CPT code
        for estimator in self.cpt_model.estimators_:
            probs = estimator.predict_proba(X)[0]
            if len(probs) > 1:
                cpt_probabilities.append(probs[1])
            else:
                cpt_probabilities.append(0.0)
        
        # Get top ICD codes
        icd_codes_with_probs = list(zip(self.icd_binarizer.classes_, icd_probabilities))
        icd_codes_with_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top CPT codes  
        cpt_codes_with_probs = list(zip(self.cpt_binarizer.classes_, cpt_probabilities))
        cpt_codes_with_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Select top codes (minimum threshold of 0.1 or top 1)
        icd_codes = []
        icd_descriptions = []
        for code, prob in icd_codes_with_probs[:top_k]:
            if prob > 0.1 or len(icd_codes) == 0:  # At least one result
                icd_codes.append(code)
                desc = self.code_mappings.get('icd10', {}).get(code, 'Description not available')
                icd_descriptions.append(desc)
        
        cpt_codes = []
        cpt_descriptions = []
        for code, prob in cpt_codes_with_probs[:top_k]:
            if prob > 0.1 or len(cpt_codes) == 0:  # At least one result
                cpt_codes.append(code)
                desc = self.code_mappings.get('cpt', {}).get(code, 'Description not available')
                cpt_descriptions.append(desc)
        
        return {
            'text': text,
            'processed_text': processed_text,
            'icd_codes': icd_codes,
            'cpt_codes': cpt_codes,
            'icd_descriptions': icd_descriptions,
            'cpt_descriptions': cpt_descriptions
        }

def load_data():
    """Load the sample data and code mappings"""
    project_root = Path(__file__).parent
    
    # Load clinical notes
    data_file = project_root / 'data' / 'raw' / 'sample_clinical_notes.csv'
    df = pd.read_csv(data_file)
    
    # Load code mappings
    mappings_file = project_root / 'data' / 'processed' / 'code_mappings.json'
    with open(mappings_file, 'r') as f:
        code_mappings = json.load(f)
    
    return df, code_mappings

def demonstrate_system():
    """Demonstrate the complete medical coding system"""
    print("🏥 Medical Coding System - Complete Demo")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    df, code_mappings = load_data()
    print(f"✓ Loaded {len(df)} clinical notes")
    print(f"✓ Loaded {len(code_mappings['icd10'])} ICD codes and {len(code_mappings['cpt'])} CPT codes")
    
    # Initialize and train system
    system = MedicalCodingSystem()
    system.train(df, code_mappings)
    
    # Test predictions on sample notes
    print("\n" + "=" * 60)
    print("TESTING ON SAMPLE CLINICAL NOTES")
    print("=" * 60)
    
    sample_notes = [
        "Patient with diabetes mellitus and hypertension presents for routine follow-up.",
        "65-year-old male with chest pain and elevated cardiac enzymes. EKG shows ST changes.",
        "Patient presents with acute appendicitis. Surgery scheduled."
    ]
    
    for i, note in enumerate(sample_notes, 1):
        print(f"\nTest Case {i}:")
        print(f"Clinical Note: {note}")
        
        result = system.predict(note)
        
        print("Predicted ICD-10 Codes:")
        for j, (code, desc) in enumerate(zip(result['icd_codes'], result['icd_descriptions']), 1):
            print(f"  {j}. {code} - {desc}")
        
        print("Predicted CPT Codes:")
        for j, (code, desc) in enumerate(zip(result['cpt_codes'], result['cpt_descriptions']), 1):
            print(f"  {j}. {code} - {desc}")
    
    # Interactive demo
    print("\n" + "=" * 60)
    print("INTERACTIVE DEMO")
    print("=" * 60)
    print("Enter your own clinical note (or press Enter to skip):")
    
    try:
        user_input = input(">>> ").strip()
        if user_input:
            print("\nYour Clinical Note:")
            print(f"'{user_input}'")
            
            result = system.predict(user_input)
            
            print("\nPredicted ICD-10 Codes:")
            if result['icd_codes']:
                for i, (code, desc) in enumerate(zip(result['icd_codes'], result['icd_descriptions']), 1):
                    print(f"  {i}. {code} - {desc}")
            else:
                print("  No ICD codes predicted")
            
            print("\nPredicted CPT Codes:")
            if result['cpt_codes']:
                for i, (code, desc) in enumerate(zip(result['cpt_codes'], result['cpt_descriptions']), 1):
                    print(f"  {i}. {code} - {desc}")
            else:
                print("  No CPT codes predicted")
        else:
            print("Skipping interactive demo...")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nWhat this system demonstrates:")
    print("• Preprocessing of clinical notes")
    print("• TF-IDF feature extraction for medical text")
    print("• Multi-label classification for ICD-10 and CPT codes")
    print("• Real-time prediction of medical codes")
    print("• Integration of medical terminology and abbreviations")
    
    print("\nPotential improvements:")
    print("• Use medical-specific word embeddings (BioBERT)")
    print("• Implement hierarchical code classification")
    print("• Add confidence scores for predictions")
    print("• Include more comprehensive medical dictionaries")
    print("• Train on larger, real-world datasets")
    
    print("\n🎯 Resume Highlight:")
    print("'Developed an NLP-based ICD/CPT auto-coding system using scikit-learn")
    print("and TF-IDF features, achieving automated medical code prediction from")
    print("clinical notes to improve healthcare billing accuracy and efficiency.'")

if __name__ == "__main__":
    try:
        demonstrate_system()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted. Goodbye!")
    except Exception as e:
        print(f"\nError running demo: {e}")
        print("Please check that all required packages are installed.")
        print("Run: pip install pandas numpy scikit-learn")
