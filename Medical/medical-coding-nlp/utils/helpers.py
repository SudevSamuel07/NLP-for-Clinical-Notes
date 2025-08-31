"""
Utility functions for the Medical Coding System

This module contains helper functions and utilities used across the project.
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns

def setup_logging(log_level: str = 'INFO', log_file: str = None) -> logging.Logger:
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper())
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def save_json(data: Dict, filepath: str, indent: int = 2):
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(filepath: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(obj: Any, filepath: str):
    """Save object to pickle file"""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath: str) -> Any:
    """Load object from pickle file"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def validate_medical_codes(codes: List[str], code_type: str = 'icd10') -> List[bool]:
    """Validate medical codes format"""
    valid_flags = []
    
    for code in codes:
        if code_type.lower() == 'icd10':
            # ICD-10 format: Letter + 2-3 digits + optional decimal + 1-2 digits
            valid = len(code) >= 3 and code[0].isalpha() and code[1:3].isdigit()
        elif code_type.lower() == 'cpt':
            # CPT format: 5 digits
            valid = len(code) == 5 and code.isdigit()
        else:
            valid = False
        
        valid_flags.append(valid)
    
    return valid_flags

def calculate_code_statistics(df: pd.DataFrame, code_column: str) -> Dict:
    """Calculate statistics for medical codes"""
    # Flatten all codes
    all_codes = []
    for code_list in df[code_column]:
        if isinstance(code_list, list):
            all_codes.extend(code_list)
        elif isinstance(code_list, str):
            all_codes.extend(code_list.split('|'))
    
    # Calculate statistics
    code_counts = pd.Series(all_codes).value_counts()
    
    stats = {
        'total_codes': len(all_codes),
        'unique_codes': len(code_counts),
        'most_common': code_counts.head(10).to_dict(),
        'avg_codes_per_note': len(all_codes) / len(df),
        'code_distribution': code_counts.describe().to_dict()
    }
    
    return stats

def create_code_frequency_plot(df: pd.DataFrame, code_column: str, 
                              title: str = 'Code Frequency', top_n: int = 20):
    """Create frequency plot for medical codes"""
    # Flatten codes
    all_codes = []
    for code_list in df[code_column]:
        if isinstance(code_list, list):
            all_codes.extend(code_list)
        elif isinstance(code_list, str):
            all_codes.extend(code_list.split('|'))
    
    # Get top codes
    code_counts = pd.Series(all_codes).value_counts().head(top_n)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    code_counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel('Medical Codes')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def split_code_string(code_string: str, delimiter: str = '|') -> List[str]:
    """Split code string into list of codes"""
    if isinstance(code_string, str):
        return [code.strip() for code in code_string.split(delimiter) if code.strip()]
    return []

def join_code_list(code_list: List[str], delimiter: str = '|') -> str:
    """Join list of codes into string"""
    if isinstance(code_list, list):
        return delimiter.join(code_list)
    return str(code_list)

def clean_medical_text(text: str) -> str:
    """Basic text cleaning for medical notes"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep medical punctuation
    import re
    text = re.sub(r'[^\w\s\.\,\-\:\;\(\)]', '', text)
    
    return text.strip()

def extract_patient_demographics(text: str) -> Dict[str, Optional[str]]:
    """Extract basic patient demographics from text"""
    import re
    
    demographics = {
        'age': None,
        'gender': None
    }
    
    # Extract age
    age_pattern = r'(\d{1,3})-year-old'
    age_match = re.search(age_pattern, text, re.IGNORECASE)
    if age_match:
        demographics['age'] = age_match.group(1)
    
    # Extract gender
    gender_pattern = r'(male|female|man|woman)'
    gender_match = re.search(gender_pattern, text, re.IGNORECASE)
    if gender_match:
        gender = gender_match.group(1).lower()
        if gender in ['male', 'man']:
            demographics['gender'] = 'male'
        elif gender in ['female', 'woman']:
            demographics['gender'] = 'female'
    
    return demographics

def calculate_text_statistics(df: pd.DataFrame, text_column: str) -> Dict:
    """Calculate text statistics for clinical notes"""
    texts = df[text_column].dropna()
    
    # Calculate various metrics
    word_counts = texts.apply(lambda x: len(str(x).split()))
    char_counts = texts.apply(lambda x: len(str(x)))
    sentence_counts = texts.apply(lambda x: len(str(x).split('.')))
    
    stats = {
        'total_notes': len(texts),
        'avg_words_per_note': word_counts.mean(),
        'avg_chars_per_note': char_counts.mean(),
        'avg_sentences_per_note': sentence_counts.mean(),
        'word_count_distribution': word_counts.describe().to_dict(),
        'char_count_distribution': char_counts.describe().to_dict()
    }
    
    return stats

def create_text_length_distribution(df: pd.DataFrame, text_column: str):
    """Create distribution plot for text lengths"""
    texts = df[text_column].dropna()
    word_counts = texts.apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    word_counts.hist(bins=30, alpha=0.7)
    plt.title('Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    word_counts.plot(kind='box')
    plt.title('Word Count Box Plot')
    plt.ylabel('Number of Words')
    
    plt.tight_layout()
    plt.show()

def check_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict:
    """Check data quality and completeness"""
    quality_report = {
        'total_rows': len(df),
        'duplicate_rows': df.duplicated().sum(),
        'missing_data': {},
        'data_types': df.dtypes.to_dict(),
        'column_coverage': {}
    }
    
    # Check missing data
    for col in required_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            quality_report['missing_data'][col] = {
                'count': missing_count,
                'percentage': missing_percent
            }
            quality_report['column_coverage'][col] = 100 - missing_percent
        else:
            quality_report['missing_data'][col] = {
                'count': len(df),
                'percentage': 100.0
            }
            quality_report['column_coverage'][col] = 0.0
    
    return quality_report

def create_project_summary(project_dir: str) -> Dict:
    """Create summary of project structure and data"""
    project_path = Path(project_dir)
    
    summary = {
        'project_structure': {},
        'data_files': {},
        'model_files': {},
        'total_size_mb': 0
    }
    
    # Scan project structure
    for subdir in ['src', 'data', 'models', 'notebooks', 'utils']:
        subdir_path = project_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.rglob('*'))
            file_info = []
            total_size = 0
            
            for file_path in files:
                if file_path.is_file():
                    size = file_path.stat().st_size
                    total_size += size
                    file_info.append({
                        'name': file_path.name,
                        'size_kb': size / 1024,
                        'type': file_path.suffix
                    })
            
            summary['project_structure'][subdir] = {
                'file_count': len(file_info),
                'total_size_mb': total_size / (1024 * 1024),
                'files': file_info
            }
            summary['total_size_mb'] += total_size / (1024 * 1024)
    
    return summary

def export_results_to_excel(results: Dict, output_file: str):
    """Export results to Excel file with multiple sheets"""
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        # Summary sheet
        if 'summary' in results:
            summary_df = pd.DataFrame([results['summary']])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Metrics sheet
        if 'metrics' in results:
            metrics_df = pd.DataFrame([results['metrics']])
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Code performance sheet
        if 'code_performance' in results:
            code_df = pd.DataFrame(results['code_performance']).T
            code_df.to_excel(writer, sheet_name='Code_Performance')
        
        # Predictions sheet (if available)
        if 'predictions' in results and isinstance(results['predictions'], pd.DataFrame):
            results['predictions'].to_excel(writer, sheet_name='Predictions', index=False)

def create_model_comparison_table(model_results: Dict) -> pd.DataFrame:
    """Create comparison table for different models"""
    comparison_data = []
    
    for model_name, metrics in model_results.items():
        row = {'Model': model_name}
        row.update(metrics)
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

def validate_predictions(predictions: List[Dict], required_fields: List[str]) -> bool:
    """Validate prediction results format"""
    if not predictions:
        return False
    
    for pred in predictions:
        if not isinstance(pred, dict):
            return False
        
        for field in required_fields:
            if field not in pred:
                return False
    
    return True

def create_medical_terminology_glossary() -> Dict[str, str]:
    """Create glossary of common medical terms"""
    return {
        'ICD-10': 'International Classification of Diseases, 10th Revision',
        'CPT': 'Current Procedural Terminology',
        'NLP': 'Natural Language Processing',
        'EHR': 'Electronic Health Record',
        'HPI': 'History of Present Illness',
        'ROS': 'Review of Systems',
        'PE': 'Physical Examination',
        'A&P': 'Assessment and Plan',
        'PMH': 'Past Medical History',
        'FH': 'Family History',
        'SH': 'Social History',
        'NKDA': 'No Known Drug Allergies',
        'SOB': 'Shortness of Breath',
        'CP': 'Chest Pain',
        'HTN': 'Hypertension',
        'DM': 'Diabetes Mellitus',
        'CAD': 'Coronary Artery Disease',
        'CHF': 'Congestive Heart Failure',
        'COPD': 'Chronic Obstructive Pulmonary Disease'
    }

def main():
    """Demonstration of utility functions"""
    print("Medical Coding System - Utility Functions")
    print("=" * 50)
    
    # Example usage
    sample_codes = ['E11.9', 'I10', '99213', '71020']
    icd_valid = validate_medical_codes(sample_codes[:2], 'icd10')
    cpt_valid = validate_medical_codes(sample_codes[2:], 'cpt')
    
    print(f"ICD code validation: {icd_valid}")
    print(f"CPT code validation: {cpt_valid}")
    
    # Create sample data
    sample_data = {
        'text': ['Patient with diabetes', 'Patient with hypertension'],
        'icd_codes': [['E11.9'], ['I10']],
        'cpt_codes': [['99213'], ['99214']]
    }
    df = pd.DataFrame(sample_data)
    
    # Calculate statistics
    stats = calculate_text_statistics(df, 'text')
    print(f"\nText statistics: {stats}")
    
    # Get glossary
    glossary = create_medical_terminology_glossary()
    print(f"\nMedical terms in glossary: {len(glossary)}")

if __name__ == "__main__":
    main()
