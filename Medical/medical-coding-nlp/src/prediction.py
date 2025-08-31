"""
Prediction Module for Medical Coding System

This module provides prediction capabilities for ICD-10 and CPT codes
from clinical notes using trained models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import logging
from pathlib import Path
import argparse
from data_preprocessing import MedicalTextPreprocessor
from feature_extraction import MedicalFeatureExtractor
from model_training import MedicalCodingModel

class MedicalCodingPredictor:
    """Complete prediction pipeline for medical coding"""
    
    def __init__(self, model_path: str = None, extractors_path: str = None):
        """Initialize the predictor with trained components"""
        self.preprocessor = MedicalTextPreprocessor()
        self.extractor = MedicalFeatureExtractor()
        self.model = MedicalCodingModel()
        
        self.code_mappings = {}
        self.feature_names = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load components if paths provided
        if model_path:
            self.load_model(model_path)
        if extractors_path:
            self.load_extractors(extractors_path)
    
    def load_model(self, model_path: str):
        """Load trained model"""
        self.model.load_model(model_path)
        self.logger.info(f"Model loaded from {model_path}")
    
    def load_extractors(self, extractors_path: str):
        """Load feature extractors"""
        self.extractor.load_extractors(extractors_path)
        self.logger.info(f"Feature extractors loaded from {extractors_path}")
    
    def load_code_mappings(self, mappings_path: str):
        """Load ICD/CPT code mappings"""
        with open(mappings_path, 'r') as f:
            self.code_mappings = json.load(f)
        self.logger.info("Code mappings loaded")
    
    def load_feature_names(self, feature_names_path: str):
        """Load feature names for interpretation"""
        with open(feature_names_path, 'r') as f:
            self.feature_names = json.load(f)
        self.logger.info("Feature names loaded")
    
    def predict_single_note(self, clinical_note: str, top_k: int = 3, 
                          include_confidence: bool = True) -> Dict:
        """Predict codes for a single clinical note"""
        # Preprocess the note
        cleaned_text = self.preprocessor.clean_text(clinical_note)
        entities = self.preprocessor.extract_medical_entities(cleaned_text)
        
        # Extract features
        features = self.extractor.transform_features([cleaned_text], self.code_mappings)
        X = self.extractor.combine_features(
            features, 
            feature_types=['tfidf', 'topics', 'medical_keywords', 'statistical']
        )
        
        # Make predictions
        predictions = self.model.predict(X, top_k=top_k)
        
        # Format results
        result = {
            'original_text': clinical_note,
            'cleaned_text': cleaned_text,
            'extracted_entities': entities,
            'predictions': {
                'icd_codes': predictions['icd_predictions'][0],
                'cpt_codes': predictions['cpt_predictions'][0]
            }
        }
        
        # Add code descriptions if mappings available
        if self.code_mappings:
            result['predictions']['icd_descriptions'] = [
                self.code_mappings.get('icd10', {}).get(code, 'Description not available')
                for code in predictions['icd_predictions'][0]
            ]
            result['predictions']['cpt_descriptions'] = [
                self.code_mappings.get('cpt', {}).get(code, 'Description not available')
                for code in predictions['cpt_predictions'][0]
            ]
        
        return result
    
    def predict_batch(self, clinical_notes: List[str], top_k: int = 3) -> List[Dict]:
        """Predict codes for multiple clinical notes"""
        self.logger.info(f"Processing {len(clinical_notes)} clinical notes...")
        
        results = []
        for i, note in enumerate(clinical_notes):
            try:
                result = self.predict_single_note(note, top_k=top_k)
                result['note_id'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error processing note {i}: {e}")
                results.append({
                    'note_id': i,
                    'error': str(e),
                    'original_text': note
                })
        
        return results
    
    def predict_from_file(self, input_file: str, output_file: str = None, 
                         text_column: str = 'text', top_k: int = 3):
        """Predict codes from a CSV file of clinical notes"""
        # Read input file
        if input_file.endswith('.csv'):
            df = pd.read_csv(input_file)
        elif input_file.endswith('.json'):
            df = pd.read_json(input_file)
        else:
            raise ValueError("Input file must be CSV or JSON format")
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in input file")
        
        # Make predictions
        notes = df[text_column].tolist()
        predictions = self.predict_batch(notes, top_k=top_k)
        
        # Create results dataframe
        results_data = []
        for i, pred in enumerate(predictions):
            if 'error' not in pred:
                row = {
                    'note_id': i,
                    'original_text': pred['original_text'],
                    'icd_codes': '|'.join(pred['predictions']['icd_codes']),
                    'cpt_codes': '|'.join(pred['predictions']['cpt_codes'])
                }
                
                if 'icd_descriptions' in pred['predictions']:
                    row['icd_descriptions'] = '|'.join(pred['predictions']['icd_descriptions'])
                if 'cpt_descriptions' in pred['predictions']:
                    row['cpt_descriptions'] = '|'.join(pred['predictions']['cpt_descriptions'])
                
                results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Save results
        if output_file:
            if output_file.endswith('.csv'):
                results_df.to_csv(output_file, index=False)
            elif output_file.endswith('.json'):
                results_df.to_json(output_file, orient='records', indent=2)
            else:
                results_df.to_csv(output_file + '.csv', index=False)
            
            self.logger.info(f"Results saved to {output_file}")
        
        return results_df
    
    def explain_prediction(self, clinical_note: str, top_features: int = 10) -> Dict:
        """Explain prediction by showing most important features"""
        # Get prediction
        prediction_result = self.predict_single_note(clinical_note)
        
        # Get feature importance if available
        feature_importance = self.model.get_feature_importance(
            self.feature_names, top_n=top_features
        )
        
        # Extract features for this note
        cleaned_text = self.preprocessor.clean_text(clinical_note)
        features = self.extractor.transform_features([cleaned_text], self.code_mappings)
        X = self.extractor.combine_features(
            features, 
            feature_types=['tfidf', 'topics', 'medical_keywords', 'statistical']
        )
        
        # Get feature values for this note
        feature_values = {}
        if self.feature_names and len(self.feature_names) == X.shape[1]:
            for i, name in enumerate(self.feature_names):
                feature_values[name] = X[0, i]
        
        explanation = {
            'prediction': prediction_result,
            'feature_importance': feature_importance,
            'feature_values': feature_values,
            'top_active_features': self._get_top_active_features(
                feature_values, feature_importance, top_features
            )
        }
        
        return explanation
    
    def _get_top_active_features(self, feature_values: Dict, 
                                feature_importance: Dict, top_n: int) -> Dict:
        """Get top features that are both important and active in this note"""
        top_active = {'icd': [], 'cpt': []}
        
        for code_type in ['icd', 'cpt']:
            if code_type in feature_importance:
                active_important = []
                
                for feature_name, importance in feature_importance[code_type]:
                    if feature_name in feature_values and feature_values[feature_name] > 0:
                        active_important.append({
                            'feature': feature_name,
                            'importance': importance,
                            'value': feature_values[feature_name],
                            'score': importance * feature_values[feature_name]
                        })
                
                # Sort by combined score and take top_n
                active_important.sort(key=lambda x: x['score'], reverse=True)
                top_active[code_type] = active_important[:top_n]
        
        return top_active
    
    def batch_evaluate(self, test_file: str, text_column: str = 'text',
                      icd_column: str = 'icd_codes', cpt_column: str = 'cpt_codes') -> Dict:
        """Evaluate model performance on a test dataset"""
        # Load test data
        if test_file.endswith('.csv'):
            df = pd.read_csv(test_file)
        else:
            df = pd.read_json(test_file)
        
        # Prepare data
        texts = df[text_column].tolist()
        true_icd = df[icd_column].apply(lambda x: x.split('|') if isinstance(x, str) else []).tolist()
        true_cpt = df[cpt_column].apply(lambda x: x.split('|') if isinstance(x, str) else []).tolist()
        
        # Preprocess
        cleaned_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Extract features
        features = self.extractor.transform_features(cleaned_texts, self.code_mappings)
        X = self.extractor.combine_features(
            features, 
            feature_types=['tfidf', 'topics', 'medical_keywords', 'statistical']
        )
        
        # Evaluate
        metrics = self.model.evaluate(X, true_icd, true_cpt)
        
        return metrics

def main():
    """Main function for command-line interface"""
    parser = argparse.ArgumentParser(description='Medical Coding Prediction System')
    parser.add_argument('--input', required=True, help='Input clinical note or file path')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--extractors', required=True, help='Path to feature extractors')
    parser.add_argument('--output', help='Output file path for batch processing')
    parser.add_argument('--top-k', type=int, default=3, help='Number of top predictions')
    parser.add_argument('--explain', action='store_true', help='Provide explanation')
    parser.add_argument('--batch', action='store_true', help='Batch processing mode')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = MedicalCodingPredictor(
        model_path=args.model,
        extractors_path=args.extractors
    )
    
    # Load additional components
    models_dir = Path(args.model).parent
    
    try:
        predictor.load_code_mappings(str(models_dir / 'code_mappings.json'))
    except FileNotFoundError:
        print("Warning: Code mappings not found")
    
    try:
        predictor.load_feature_names(str(models_dir / 'feature_names.json'))
    except FileNotFoundError:
        print("Warning: Feature names not found")
    
    if args.batch:
        # Batch processing
        results = predictor.predict_from_file(
            args.input, 
            args.output, 
            top_k=args.top_k
        )
        print(f"Processed {len(results)} notes")
        print(results.head())
        
    else:
        # Single note processing
        if Path(args.input).exists():
            # Read from file
            with open(args.input, 'r') as f:
                clinical_note = f.read()
        else:
            # Treat as direct text input
            clinical_note = args.input
        
        if args.explain:
            # Get explanation
            explanation = predictor.explain_prediction(clinical_note)
            print(json.dumps(explanation, indent=2, default=str))
        else:
            # Get simple prediction
            result = predictor.predict_single_note(clinical_note, top_k=args.top_k)
            print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    # Example usage
    sample_note = """
    Patient is a 65-year-old male with a history of type 2 diabetes mellitus and hypertension.
    He presents today with chest pain and shortness of breath. Physical examination reveals
    elevated blood pressure and irregular heart rhythm. EKG shows atrial fibrillation.
    Laboratory results show elevated glucose levels. Patient will be started on anticoagulation
    therapy and diabetes management will be optimized.
    """
    
    print("Sample Clinical Note Prediction:")
    print("=" * 50)
    print(sample_note)
    print("=" * 50)
    
    # This would normally use trained models
    print("Note: Run with trained models for actual predictions")
    print("Usage: python prediction.py --input 'clinical note' --model path/to/model --extractors path/to/extractors")
