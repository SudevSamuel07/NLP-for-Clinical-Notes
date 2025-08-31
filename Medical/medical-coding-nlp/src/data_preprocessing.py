"""
Data Preprocessing Module for Medical Coding System

This module handles the preprocessing of clinical notes and medical coding data.
Includes text cleaning, normalization, and feature preparation.
"""

import pandas as pd
import numpy as np
import re
import spacy
from spacy import displacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class MedicalTextPreprocessor:
    """Preprocessor for medical text data"""
    
    def __init__(self):
        """Initialize the preprocessor with medical-specific configurations"""
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise
        
        # Medical stopwords (common medical terms that might not be informative)
        self.medical_stopwords = set([
            'patient', 'pt', 'history', 'hx', 'chief', 'complaint', 'cc',
            'present', 'illness', 'hpi', 'review', 'systems', 'ros',
            'physical', 'exam', 'pe', 'assessment', 'plan', 'impression'
        ])
        
        # Standard English stopwords
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(self.medical_stopwords)
        
        # Medical abbreviation mappings
        self.medical_abbreviations = {
            'dm': 'diabetes mellitus',
            'htn': 'hypertension',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'mi': 'myocardial infarction',
            'cva': 'cerebrovascular accident',
            'uti': 'urinary tract infection',
            'dvt': 'deep vein thrombosis',
            'pe': 'pulmonary embolism',
            'afib': 'atrial fibrillation',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'abd': 'abdominal',
            'nkda': 'no known drug allergies',
            'nka': 'no known allergies'
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize medical text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical-relevant punctuation
        text = re.sub(r'[^\w\s\.\,\-\:\;]', '', text)
        
        # Expand medical abbreviations
        words = text.split()
        expanded_words = []
        for word in words:
            if word in self.medical_abbreviations:
                expanded_words.append(self.medical_abbreviations[word])
            else:
                expanded_words.append(word)
        text = ' '.join(expanded_words)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\,{2,}', ',', text)
        
        return text.strip()
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities using spaCy NER"""
        doc = self.nlp(text)
        
        entities = {
            'conditions': [],
            'medications': [],
            'procedures': [],
            'symptoms': [],
            'body_parts': []
        }
        
        for ent in doc.ents:
            entity_text = ent.text.lower()
            
            # Categorize entities based on label and context
            if ent.label_ in ['DISEASE', 'CONDITION']:
                entities['conditions'].append(entity_text)
            elif ent.label_ in ['MEDICATION', 'DRUG']:
                entities['medications'].append(entity_text)
            elif ent.label_ in ['PROCEDURE', 'TREATMENT']:
                entities['procedures'].append(entity_text)
            elif ent.label_ in ['SYMPTOM']:
                entities['symptoms'].append(entity_text)
            elif ent.label_ in ['BODY_PART', 'ORGAN']:
                entities['body_parts'].append(entity_text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def tokenize_and_filter(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """Tokenize text and optionally remove stopwords"""
        # Tokenize
        tokens = word_tokenize(text)
        
        # Convert to lowercase and filter
        tokens = [token.lower() for token in tokens if token.isalnum()]
        
        # Remove stopwords if requested
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Filter very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def preprocess_clinical_notes(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Preprocess a dataframe of clinical notes"""
        self.logger.info(f"Preprocessing {len(df)} clinical notes...")
        
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Clean text
        processed_df['cleaned_text'] = processed_df[text_column].apply(self.clean_text)
        
        # Extract entities
        processed_df['entities'] = processed_df['cleaned_text'].apply(self.extract_medical_entities)
        
        # Tokenize
        processed_df['tokens'] = processed_df['cleaned_text'].apply(
            lambda x: self.tokenize_and_filter(x, remove_stopwords=True)
        )
        
        # Calculate text statistics
        processed_df['word_count'] = processed_df['tokens'].apply(len)
        processed_df['char_count'] = processed_df['cleaned_text'].apply(len)
        
        self.logger.info("Preprocessing completed!")
        return processed_df
    
    def create_code_mappings(self, icd_file: str = None, cpt_file: str = None) -> Dict:
        """Create mappings for ICD-10 and CPT codes"""
        mappings = {
            'icd10': {},
            'cpt': {}
        }
        
        # Sample ICD-10 codes (in practice, load from official files)
        sample_icd10 = {
            'E11.9': 'Type 2 diabetes mellitus without complications',
            'I10': 'Essential hypertension',
            'Z51.11': 'Encounter for antineoplastic chemotherapy',
            'F32.9': 'Major depressive disorder, single episode, unspecified',
            'J44.1': 'Chronic obstructive pulmonary disease with acute exacerbation',
            'N18.6': 'End stage renal disease',
            'I25.10': 'Atherosclerotic heart disease of native coronary artery without angina pectoris',
            'J18.9': 'Pneumonia, unspecified organism',
            'K21.9': 'Gastro-esophageal reflux disease without esophagitis',
            'M79.3': 'Panniculitis, unspecified'
        }
        
        # Sample CPT codes (in practice, load from official files)
        sample_cpt = {
            '99213': 'Office or other outpatient visit for the evaluation and management of an established patient',
            '99214': 'Office or other outpatient visit for the evaluation and management of an established patient',
            '99232': 'Subsequent hospital care, per day, for the evaluation and management of a patient',
            '90834': 'Psychotherapy, 45 minutes with patient and/or family member',
            '93000': 'Electrocardiogram, routine ECG with at least 12 leads',
            '71020': 'Radiologic examination, chest, 2 views, frontal and lateral',
            '80053': 'Comprehensive metabolic panel',
            '85025': 'Blood count; complete (CBC), automated',
            '36415': 'Collection of venous blood by venipuncture',
            '99395': 'Periodic comprehensive preventive medicine reevaluation'
        }
        
        mappings['icd10'] = sample_icd10
        mappings['cpt'] = sample_cpt
        
        return mappings
    
    def load_and_preprocess_datasets(self, data_dir: str) -> Tuple[pd.DataFrame, Dict]:
        """Load and preprocess all datasets"""
        data_path = Path(data_dir)
        
        # Create sample synthetic data for demonstration
        # In practice, you would load real datasets from Kaggle or other sources
        sample_notes = [
            "Patient is a 45-year-old male with a history of type 2 diabetes mellitus without complications. He presents today for routine follow-up. Blood glucose levels have been well controlled on metformin.",
            "67-year-old female with hypertension and coronary artery disease. Came in for chest pain evaluation. EKG shows normal sinus rhythm. Will continue current medications.",
            "35-year-old pregnant female at 32 weeks gestation presents with shortness of breath. Physical exam reveals mild lower extremity edema. Fetal heart tones are reassuring.",
            "Patient has chronic obstructive pulmonary disease with acute exacerbation. Presents with increased dyspnea and productive cough. Will start prednisone and antibiotics.",
            "28-year-old male presents with acute appendicitis. Physical exam shows McBurney's point tenderness. Patient scheduled for laparoscopic appendectomy."
        ]
        
        sample_icd_codes = [
            ['E11.9'],  # Type 2 diabetes without complications
            ['I10', 'I25.10'],  # Hypertension, CAD
            ['O26.89'],  # Other specified pregnancy-related conditions
            ['J44.1'],  # COPD with acute exacerbation
            ['K35.9']   # Acute appendicitis
        ]
        
        sample_cpt_codes = [
            ['99213', '80053'],  # Office visit, metabolic panel
            ['99214', '93000'],  # Office visit, EKG
            ['99213'],  # Office visit
            ['99214'],  # Office visit
            ['44970']   # Laparoscopic appendectomy
        ]
        
        # Create DataFrame
        df = pd.DataFrame({
            'text': sample_notes,
            'icd_codes': sample_icd_codes,
            'cpt_codes': sample_cpt_codes
        })
        
        # Preprocess the notes
        processed_df = self.preprocess_clinical_notes(df)
        
        # Create code mappings
        code_mappings = self.create_code_mappings()
        
        self.logger.info(f"Loaded and processed {len(processed_df)} clinical notes")
        
        return processed_df, code_mappings
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets"""
        X = df['cleaned_text'].values
        y_icd = df['icd_codes'].values
        y_cpt = df['cpt_codes'].values
        
        # Split the data
        X_train, X_test, y_icd_train, y_icd_test, y_cpt_train, y_cpt_test = train_test_split(
            X, y_icd, y_cpt, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_icd_train, y_icd_test, y_cpt_train, y_cpt_test

def main():
    """Main function to demonstrate preprocessing"""
    # Initialize preprocessor
    preprocessor = MedicalTextPreprocessor()
    
    # Load and preprocess data
    data_dir = Path(__file__).parent.parent / 'data'
    df, code_mappings = preprocessor.load_and_preprocess_datasets(str(data_dir))
    
    # Split data
    splits = preprocessor.split_data(df)
    X_train, X_test, y_icd_train, y_icd_test, y_cpt_train, y_cpt_test = splits
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Sample processed note: {df['cleaned_text'].iloc[0][:200]}...")
    print(f"Sample entities: {df['entities'].iloc[0]}")
    
    # Save processed data
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    df.to_pickle(processed_dir / 'processed_notes.pkl')
    
    with open(processed_dir / 'code_mappings.json', 'w') as f:
        json.dump(code_mappings, f, indent=2)
    
    print("Processed data saved successfully!")

if __name__ == "__main__":
    main()
