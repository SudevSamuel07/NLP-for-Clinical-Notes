"""
Data Collection Utilities for Medical Coding System

This module provides utilities to download and prepare datasets from Kaggle
and other sources for training the medical coding system.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from pathlib import Path
import zipfile
import kaggle
from typing import List, Dict, Tuple
import logging

class MedicalDataCollector:
    """Utility class for collecting medical datasets"""
    
    def __init__(self, data_dir: str):
        """Initialize the data collector"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / 'raw').mkdir(exist_ok=True)
        (self.data_dir / 'processed').mkdir(exist_ok=True)
        (self.data_dir / 'reference').mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def download_kaggle_dataset(self, dataset_name: str, extract_to: str = None):
        """Download dataset from Kaggle"""
        try:
            if extract_to is None:
                extract_to = str(self.data_dir / 'raw')
            
            self.logger.info(f"Downloading Kaggle dataset: {dataset_name}")
            kaggle.api.dataset_download_files(dataset_name, path=extract_to, unzip=True)
            self.logger.info("Download completed!")
            
        except Exception as e:
            self.logger.error(f"Error downloading dataset {dataset_name}: {e}")
            self.logger.info("Make sure you have Kaggle API credentials configured")
            self.logger.info("Visit: https://www.kaggle.com/docs/api")
    
    def download_medical_datasets(self):
        """Download recommended medical datasets"""
        datasets = [
            # Medical notes and coding datasets
            "chadbellous/medical-coding-classification",
            "finalepoch/medical-ner",
            "jpmiller/layoutlm-invoices",  # For structure understanding
            "kmader/skin-cancer-mnist-ham10000",  # For medical classification example
        ]
        
        self.logger.info("Downloading medical datasets...")
        
        for dataset in datasets:
            try:
                self.download_kaggle_dataset(dataset)
            except Exception as e:
                self.logger.warning(f"Could not download {dataset}: {e}")
                continue
    
    def download_icd10_codes(self):
        """Download ICD-10 code mappings"""
        try:
            self.logger.info("Downloading ICD-10 codes...")
            
            # CMS ICD-10 codes (public domain)
            icd10_url = "https://www.cms.gov/files/zip/2024-icd-10-cm-codes-file.zip"
            
            response = requests.get(icd10_url, timeout=30)
            if response.status_code == 200:
                zip_path = self.data_dir / 'reference' / 'icd10_codes.zip'
                with open(zip_path, 'wb') as f:
                    f.write(response.content)
                
                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_dir / 'reference' / 'icd10')
                
                self.logger.info("ICD-10 codes downloaded successfully!")
            else:
                self.logger.warning("Could not download ICD-10 codes from CMS")
                
        except Exception as e:
            self.logger.error(f"Error downloading ICD-10 codes: {e}")
    
    def download_cpt_codes(self):
        """Download CPT code information (limited free access)"""
        try:
            self.logger.info("Note: CPT codes are copyrighted by AMA")
            self.logger.info("Using sample CPT codes for demonstration")
            
            # Create sample CPT codes for development
            sample_cpt = {
                "99213": "Office/outpatient visit est patient",
                "99214": "Office/outpatient visit est patient",
                "99215": "Office/outpatient visit est patient",
                "99232": "Subsequent hospital care",
                "99233": "Subsequent hospital care",
                "99234": "Observation or inpatient care",
                "90834": "Psychotherapy 45 minutes",
                "90837": "Psychotherapy 60 minutes",
                "93000": "Electrocardiogram routine",
                "93005": "Electrocardiogram tracing",
                "71020": "Chest x-ray 2 views",
                "71045": "Chest x-ray single view",
                "80053": "Comprehensive metabolic panel",
                "80061": "Lipid panel",
                "85025": "Complete blood count",
                "85027": "Complete blood count automated",
                "36415": "Venous blood collection",
                "99395": "Preventive visit 18-39 years",
                "99396": "Preventive visit 40-64 years",
                "99397": "Preventive visit 65+ years"
            }
            
            cpt_file = self.data_dir / 'reference' / 'sample_cpt_codes.json'
            with open(cpt_file, 'w') as f:
                json.dump(sample_cpt, f, indent=2)
            
            self.logger.info("Sample CPT codes created")
            
        except Exception as e:
            self.logger.error(f"Error creating CPT codes: {e}")
    
    def create_synthetic_dataset(self, num_samples: int = 1000):
        """Create synthetic medical notes dataset for training"""
        self.logger.info(f"Creating synthetic dataset with {num_samples} samples...")
        
        # Templates for different types of medical notes
        templates = [
            # Diabetes templates
            {
                "template": "Patient is a {age}-year-old {gender} with a history of type {dm_type} diabetes mellitus {complications}. {presentation} Blood glucose levels {control} on {medication}. {exam_findings} {plan}",
                "variables": {
                    "age": list(range(25, 85)),
                    "gender": ["male", "female"],
                    "dm_type": ["1", "2"],
                    "complications": ["without complications", "with complications", "with nephropathy", "with retinopathy"],
                    "presentation": ["Presents for routine follow-up.", "Presents with polyuria and polydipsia.", "Presents for diabetes management."],
                    "control": ["have been well controlled", "have been poorly controlled", "are elevated"],
                    "medication": ["metformin", "insulin", "glipizide", "metformin and insulin"],
                    "exam_findings": ["Physical examination is unremarkable.", "Physical exam reveals diabetic retinopathy.", "Examination shows peripheral neuropathy."],
                    "plan": ["Will continue current medications.", "Will adjust insulin dosage.", "Will refer to endocrinologist."]
                },
                "icd_codes": [["E11.9"], ["E10.9"], ["E11.21"], ["E11.31"]],
                "cpt_codes": [["99213"], ["99214"], ["99213", "80053"]]
            },
            
            # Hypertension templates
            {
                "template": "{age}-year-old {gender} with {htn_type} hypertension presents for {visit_type}. Current BP is {bp}. Patient {compliance} with {medication} therapy. {plan}",
                "variables": {
                    "age": list(range(30, 80)),
                    "gender": ["male", "female"],
                    "htn_type": ["essential", "secondary"],
                    "visit_type": ["blood pressure check", "routine follow-up", "medication adjustment"],
                    "bp": ["145/92", "160/95", "130/85", "140/90"],
                    "compliance": ["has been compliant", "has been non-compliant", "reports difficulty"],
                    "medication": ["lisinopril", "amlodipine", "hydrochlorothiazide", "losartan"],
                    "plan": ["Will increase dosage.", "Will add second medication.", "Will continue current regimen."]
                },
                "icd_codes": [["I10"], ["I15.9"]],
                "cpt_codes": [["99213"], ["99214"]]
            },
            
            # Pneumonia templates
            {
                "template": "{age}-year-old {gender} presents with {symptoms} for {duration}. Physical exam reveals {findings}. {imaging} {diagnosis} {treatment}",
                "variables": {
                    "age": list(range(20, 85)),
                    "gender": ["male", "female"],
                    "symptoms": ["fever and cough", "shortness of breath", "productive cough", "chest pain and fever"],
                    "duration": ["3 days", "1 week", "2 days", "5 days"],
                    "findings": ["crackles in right lower lobe", "decreased breath sounds", "bronchial breath sounds", "dullness to percussion"],
                    "imaging": ["Chest X-ray shows pneumonia.", "CT chest shows consolidation.", "Chest X-ray reveals infiltrate."],
                    "diagnosis": "Pneumonia diagnosed.",
                    "treatment": ["Will start antibiotic therapy.", "Started on azithromycin.", "Prescribed amoxicillin."]
                },
                "icd_codes": [["J18.9"], ["J15.9"], ["J44.0"]],
                "cpt_codes": [["99214"], ["71020"], ["99214", "71020"]]
            },
            
            # Cardiac templates
            {
                "template": "{age}-year-old {gender} with history of {cardiac_hx} presents with {symptoms}. {exam} {ekg} {diagnosis} {plan}",
                "variables": {
                    "age": list(range(40, 85)),
                    "gender": ["male", "female"],
                    "cardiac_hx": ["coronary artery disease", "myocardial infarction", "atrial fibrillation", "heart failure"],
                    "symptoms": ["chest pain", "shortness of breath", "palpitations", "fatigue"],
                    "exam": ["Physical exam reveals irregular rhythm.", "Examination shows murmur.", "Heart sounds are normal."],
                    "ekg": ["EKG shows atrial fibrillation.", "EKG reveals ST changes.", "EKG is normal."],
                    "diagnosis": ["Atrial fibrillation.", "Stable angina.", "Heart failure exacerbation."],
                    "plan": ["Will start anticoagulation.", "Will optimize medications.", "Will refer to cardiology."]
                },
                "icd_codes": [["I48.91"], ["I25.10"], ["I50.9"]],
                "cpt_codes": [["99214"], ["93000"], ["99214", "93000"]]
            }
        ]
        
        # Generate synthetic data
        synthetic_data = []
        
        for i in range(num_samples):
            # Choose random template
            template_data = np.random.choice(templates)
            template = template_data["template"]
            variables = template_data["variables"]
            
            # Fill template with random values
            note_vars = {}
            for var, options in variables.items():
                note_vars[var] = np.random.choice(options)
            
            # Generate note text
            clinical_note = template.format(**note_vars)
            
            # Assign codes
            icd_codes = np.random.choice(template_data["icd_codes"])
            cpt_codes = np.random.choice(template_data["cpt_codes"])
            
            synthetic_data.append({
                'note_id': f'synthetic_{i:04d}',
                'text': clinical_note,
                'icd_codes': icd_codes,
                'cpt_codes': cpt_codes,
                'source': 'synthetic'
            })
        
        # Create DataFrame and save
        df = pd.DataFrame(synthetic_data)
        
        # Save to file
        output_file = self.data_dir / 'raw' / 'synthetic_clinical_notes.csv'
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Synthetic dataset created: {output_file}")
        self.logger.info(f"Dataset shape: {df.shape}")
        
        return df
    
    def prepare_mimic_format(self, notes_file: str = None):
        """Prepare data in MIMIC-III format (if available)"""
        if notes_file and Path(notes_file).exists():
            self.logger.info("Processing MIMIC-III style data...")
            
            # This would process real MIMIC data if available
            # Note: MIMIC data requires special access and training
            df = pd.read_csv(notes_file)
            
            # Basic preprocessing for MIMIC format
            if 'TEXT' in df.columns:
                df = df.rename(columns={'TEXT': 'text'})
            
            # Save processed data
            output_file = self.data_dir / 'processed' / 'mimic_processed.csv'
            df.to_csv(output_file, index=False)
            
            self.logger.info(f"MIMIC data processed: {output_file}")
            
        else:
            self.logger.info("MIMIC data not available - using synthetic data")
    
    def setup_kaggle_api(self):
        """Setup Kaggle API credentials"""
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_file = kaggle_dir / 'kaggle.json'
        
        if not kaggle_file.exists():
            self.logger.warning("Kaggle API not configured!")
            self.logger.info("To download datasets from Kaggle:")
            self.logger.info("1. Go to https://www.kaggle.com/account")
            self.logger.info("2. Create a new API token")
            self.logger.info("3. Download kaggle.json")
            self.logger.info(f"4. Place it in {kaggle_dir}")
            self.logger.info("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
            
            return False
        
        return True
    
    def collect_all_data(self):
        """Collect all necessary data for the project"""
        self.logger.info("Starting data collection process...")
        
        # Check Kaggle API setup
        if self.setup_kaggle_api():
            # Download Kaggle datasets
            try:
                self.download_medical_datasets()
            except Exception as e:
                self.logger.warning(f"Kaggle download failed: {e}")
        
        # Download reference data
        self.download_icd10_codes()
        self.download_cpt_codes()
        
        # Create synthetic dataset
        self.create_synthetic_dataset(1000)
        
        self.logger.info("Data collection completed!")
        self.logger.info(f"Data stored in: {self.data_dir}")
        
        # List collected files
        self.list_collected_data()
    
    def list_collected_data(self):
        """List all collected data files"""
        self.logger.info("\nCollected data files:")
        self.logger.info("=" * 40)
        
        for subdir in ['raw', 'processed', 'reference']:
            subdir_path = self.data_dir / subdir
            if subdir_path.exists():
                self.logger.info(f"\n{subdir.upper()}:")
                for file_path in subdir_path.rglob('*'):
                    if file_path.is_file():
                        size = file_path.stat().st_size / 1024  # Size in KB
                        self.logger.info(f"  {file_path.name} ({size:.1f} KB)")

def main():
    """Main function to collect data"""
    # Initialize data collector
    data_dir = Path(__file__).parent.parent / 'data'
    collector = MedicalDataCollector(str(data_dir))
    
    # Collect all data
    collector.collect_all_data()

if __name__ == "__main__":
    main()
