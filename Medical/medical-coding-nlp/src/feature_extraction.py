"""
Feature Extraction Module for Medical Coding System

This module extracts features from preprocessed clinical notes for machine learning models.
Includes TF-IDF, word embeddings, and medical-specific features.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import pickle
import json

class MedicalFeatureExtractor:
    """Extract features from medical text for coding prediction"""
    
    def __init__(self, max_features: int = 10000, min_df: int = 2, max_df: float = 0.95):
        """Initialize feature extractor with parameters"""
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 3),  # unigrams, bigrams, and trigrams
            stop_words='english'
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Medical-specific feature extractors
        self.medical_keywords = self._load_medical_keywords()
        
        # Initialize transformer model for embeddings
        self.tokenizer = None
        self.transformer_model = None
        self._init_transformer()
        
        # Topic modeling
        self.lda_model = None
        self.n_topics = 20
        
        # Feature storage
        self.feature_names = []
        self.is_fitted = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _init_transformer(self):
        """Initialize transformer model for embeddings"""
        try:
            model_name = "bert-base-uncased"  # Can switch to BioClinicalBERT for better medical performance
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.transformer_model = AutoModel.from_pretrained(model_name)
            self.logger.info(f"Loaded transformer model: {model_name}")
        except Exception as e:
            self.logger.warning(f"Could not load transformer model: {e}")
    
    def _load_medical_keywords(self) -> Dict[str, List[str]]:
        """Load medical keyword categories for feature extraction"""
        return {
            'symptoms': [
                'pain', 'fever', 'nausea', 'vomiting', 'headache', 'fatigue',
                'shortness of breath', 'chest pain', 'abdominal pain', 'dizziness',
                'cough', 'dyspnea', 'syncope', 'palpitations', 'weakness'
            ],
            'conditions': [
                'diabetes', 'hypertension', 'cancer', 'pneumonia', 'asthma',
                'depression', 'anxiety', 'arthritis', 'copd', 'heart disease',
                'stroke', 'kidney disease', 'liver disease', 'infection'
            ],
            'procedures': [
                'surgery', 'biopsy', 'endoscopy', 'catheterization', 'dialysis',
                'chemotherapy', 'radiation', 'transplant', 'anesthesia',
                'injection', 'examination', 'consultation', 'therapy'
            ],
            'medications': [
                'antibiotic', 'insulin', 'aspirin', 'morphine', 'metformin',
                'lisinopril', 'atorvastatin', 'omeprazole', 'prednisone',
                'warfarin', 'metoprolol', 'hydrochlorothiazide'
            ],
            'body_parts': [
                'heart', 'lung', 'kidney', 'liver', 'brain', 'stomach',
                'intestine', 'spine', 'knee', 'shoulder', 'chest', 'abdomen',
                'pelvis', 'extremity', 'head', 'neck'
            ]
        }
    
    def extract_medical_keyword_features(self, texts: List[str]) -> np.ndarray:
        """Extract binary features based on medical keyword presence"""
        features = []
        
        for text in texts:
            text_lower = text.lower()
            feature_vector = []
            
            for category, keywords in self.medical_keywords.items():
                for keyword in keywords:
                    # Binary feature: 1 if keyword present, 0 otherwise
                    feature_vector.append(1 if keyword in text_lower else 0)
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        """Extract statistical features from texts"""
        features = []
        
        for text in texts:
            feature_vector = [
                len(text),  # Text length
                len(text.split()),  # Word count
                len(text.split('.')) - 1,  # Sentence count
                text.count(','),  # Comma count
                text.count(';'),  # Semicolon count
                len([w for w in text.split() if w.isupper()]),  # Uppercase words
                len([w for w in text.split() if any(c.isdigit() for c in w)]),  # Words with numbers
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def extract_tfidf_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Extract TF-IDF features"""
        if fit:
            self.logger.info("Fitting TF-IDF vectorizer...")
            tfidf_features = self.tfidf_vectorizer.fit_transform(texts)
        else:
            tfidf_features = self.tfidf_vectorizer.transform(texts)
        
        return tfidf_features.toarray()
    
    def extract_topic_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        """Extract topic modeling features using LDA"""
        if fit:
            self.logger.info("Fitting topic model...")
            # First get count vectors for LDA
            count_features = self.count_vectorizer.fit_transform(texts)
            
            # Fit LDA model
            self.lda_model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=10
            )
            topic_features = self.lda_model.fit_transform(count_features)
        else:
            count_features = self.count_vectorizer.transform(texts)
            topic_features = self.lda_model.transform(count_features)
        
        return topic_features
    
    def extract_transformer_embeddings(self, texts: List[str], max_length: int = 512) -> np.ndarray:
        """Extract embeddings using transformer model"""
        if self.tokenizer is None or self.transformer_model is None:
            self.logger.warning("Transformer model not available, returning zeros")
            return np.zeros((len(texts), 768))  # BERT embedding size
        
        embeddings = []
        
        for text in texts:
            # Tokenize and encode
            inputs = self.tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors='pt'
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.transformer_model(**inputs)
                # Use CLS token embedding as sentence representation
                embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def extract_code_specific_features(self, texts: List[str], code_mappings: Dict) -> Dict[str, np.ndarray]:
        """Extract features specific to ICD and CPT codes"""
        features = {
            'icd_indicators': [],
            'cpt_indicators': []
        }
        
        # Keywords that often indicate diagnosis vs procedure
        diagnosis_keywords = [
            'diagnosis', 'condition', 'disease', 'disorder', 'syndrome',
            'history of', 'presents with', 'suffering from'
        ]
        
        procedure_keywords = [
            'procedure', 'surgery', 'operation', 'treatment', 'therapy',
            'performed', 'administered', 'conducted', 'examination'
        ]
        
        for text in texts:
            text_lower = text.lower()
            
            # ICD (diagnosis) indicators
            icd_score = sum(1 for keyword in diagnosis_keywords if keyword in text_lower)
            features['icd_indicators'].append(icd_score)
            
            # CPT (procedure) indicators
            cpt_score = sum(1 for keyword in procedure_keywords if keyword in text_lower)
            features['cpt_indicators'].append(cpt_score)
        
        return {k: np.array(v).reshape(-1, 1) for k, v in features.items()}
    
    def fit_extract_features(self, texts: List[str], code_mappings: Dict = None) -> Dict[str, np.ndarray]:
        """Fit extractors and extract all features"""
        self.logger.info(f"Extracting features from {len(texts)} texts...")
        
        features = {}
        
        # TF-IDF features
        features['tfidf'] = self.extract_tfidf_features(texts, fit=True)
        
        # Topic features
        features['topics'] = self.extract_topic_features(texts, fit=True)
        
        # Medical keyword features
        features['medical_keywords'] = self.extract_medical_keyword_features(texts)
        
        # Statistical features
        features['statistical'] = self.extract_statistical_features(texts)
        
        # Transformer embeddings (optional, can be slow)
        # features['embeddings'] = self.extract_transformer_embeddings(texts)
        
        # Code-specific features
        if code_mappings:
            code_features = self.extract_code_specific_features(texts, code_mappings)
            features.update(code_features)
        
        self.is_fitted = True
        self.logger.info("Feature extraction completed!")
        
        return features
    
    def transform_features(self, texts: List[str], code_mappings: Dict = None) -> Dict[str, np.ndarray]:
        """Transform new texts using fitted extractors"""
        if not self.is_fitted:
            raise ValueError("Feature extractors must be fitted first!")
        
        features = {}
        
        # TF-IDF features
        features['tfidf'] = self.extract_tfidf_features(texts, fit=False)
        
        # Topic features
        features['topics'] = self.extract_topic_features(texts, fit=False)
        
        # Medical keyword features
        features['medical_keywords'] = self.extract_medical_keyword_features(texts)
        
        # Statistical features
        features['statistical'] = self.extract_statistical_features(texts)
        
        # Code-specific features
        if code_mappings:
            code_features = self.extract_code_specific_features(texts, code_mappings)
            features.update(code_features)
        
        return features
    
    def combine_features(self, feature_dict: Dict[str, np.ndarray], 
                        feature_types: List[str] = None) -> np.ndarray:
        """Combine multiple feature types into a single matrix"""
        if feature_types is None:
            feature_types = list(feature_dict.keys())
        
        feature_matrices = []
        feature_names = []
        
        for feature_type in feature_types:
            if feature_type in feature_dict:
                matrix = feature_dict[feature_type]
                if len(matrix.shape) == 1:
                    matrix = matrix.reshape(-1, 1)
                
                feature_matrices.append(matrix)
                
                # Add feature names for interpretability
                if feature_type == 'tfidf':
                    feature_names.extend([f"tfidf_{i}" for i in range(matrix.shape[1])])
                elif feature_type == 'topics':
                    feature_names.extend([f"topic_{i}" for i in range(matrix.shape[1])])
                elif feature_type == 'medical_keywords':
                    feature_names.extend([f"keyword_{i}" for i in range(matrix.shape[1])])
                elif feature_type == 'statistical':
                    stat_names = ['text_length', 'word_count', 'sentence_count', 
                                'comma_count', 'semicolon_count', 'uppercase_words', 'numeric_words']
                    feature_names.extend(stat_names[:matrix.shape[1]])
                else:
                    feature_names.extend([f"{feature_type}_{i}" for i in range(matrix.shape[1])])
        
        if not feature_matrices:
            raise ValueError("No valid feature types provided!")
        
        combined_features = np.concatenate(feature_matrices, axis=1)
        self.feature_names = feature_names
        
        self.logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def save_extractors(self, filepath: str):
        """Save fitted extractors to disk"""
        extractor_data = {
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'count_vectorizer': self.count_vectorizer,
            'lda_model': self.lda_model,
            'medical_keywords': self.medical_keywords,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(extractor_data, f)
        
        self.logger.info(f"Feature extractors saved to {filepath}")
    
    def load_extractors(self, filepath: str):
        """Load fitted extractors from disk"""
        with open(filepath, 'rb') as f:
            extractor_data = pickle.load(f)
        
        self.tfidf_vectorizer = extractor_data['tfidf_vectorizer']
        self.count_vectorizer = extractor_data['count_vectorizer']
        self.lda_model = extractor_data['lda_model']
        self.medical_keywords = extractor_data['medical_keywords']
        self.feature_names = extractor_data['feature_names']
        self.is_fitted = extractor_data['is_fitted']
        
        self.logger.info(f"Feature extractors loaded from {filepath}")

def main():
    """Main function to demonstrate feature extraction"""
    from data_preprocessing import MedicalTextPreprocessor
    
    # Initialize components
    preprocessor = MedicalTextPreprocessor()
    extractor = MedicalFeatureExtractor(max_features=5000)
    
    # Load processed data
    data_dir = Path(__file__).parent.parent / 'data'
    
    # Load sample data
    df, code_mappings = preprocessor.load_and_preprocess_datasets(str(data_dir))
    
    # Extract features
    texts = df['cleaned_text'].tolist()
    features = extractor.fit_extract_features(texts, code_mappings)
    
    # Combine features
    combined_features = extractor.combine_features(
        features, 
        feature_types=['tfidf', 'topics', 'medical_keywords', 'statistical']
    )
    
    print(f"Combined features shape: {combined_features.shape}")
    print(f"Number of feature types: {len(features)}")
    print(f"Feature types: {list(features.keys())}")
    
    # Save extractors
    models_dir = data_dir.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    extractor.save_extractors(str(models_dir / 'feature_extractors.pkl'))
    
    # Save features
    np.save(models_dir / 'training_features.npy', combined_features)
    
    print("Feature extraction completed and saved!")

if __name__ == "__main__":
    main()
