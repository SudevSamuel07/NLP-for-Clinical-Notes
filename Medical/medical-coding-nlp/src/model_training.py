"""
Model Training Module for Medical Coding System

This module trains machine learning models for ICD-10 and CPT code prediction.
Includes multiple algorithms and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import pickle
from typing import List, Dict, Tuple, Any
import logging
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class MedicalCodingModel:
    """Machine learning model for medical coding prediction"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """Initialize the model with specified type"""
        self.model_type = model_type
        self.icd_model = None
        self.cpt_model = None
        self.icd_binarizer = MultiLabelBinarizer()
        self.cpt_binarizer = MultiLabelBinarizer()
        self.scaler = StandardScaler()
        
        self.is_trained = False
        self.feature_importance = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize models
        self._init_models()
    
    def _init_models(self):
        """Initialize ML models based on type"""
        if self.model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boost':
            base_model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        elif self.model_type == 'logistic':
            base_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            base_model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        # Use MultiOutputClassifier for multi-label prediction
        self.icd_model = MultiOutputClassifier(base_model)
        self.cpt_model = MultiOutputClassifier(base_model)
        
        self.logger.info(f"Initialized {self.model_type} models")
    
    def prepare_labels(self, icd_codes: List[List[str]], cpt_codes: List[List[str]]) -> Tuple:
        """Prepare multi-label targets"""
        # Fit and transform ICD codes
        icd_binary = self.icd_binarizer.fit_transform(icd_codes)
        
        # Fit and transform CPT codes
        cpt_binary = self.cpt_binarizer.fit_transform(cpt_codes)
        
        self.logger.info(f"ICD label shape: {icd_binary.shape}")
        self.logger.info(f"CPT label shape: {cpt_binary.shape}")
        
        return icd_binary, cpt_binary
    
    def train(self, X: np.ndarray, icd_codes: List[List[str]], 
              cpt_codes: List[List[str]], scale_features: bool = True):
        """Train both ICD and CPT models"""
        self.logger.info("Starting model training...")
        
        # Scale features if requested
        if scale_features:
            X = self.scaler.fit_transform(X)
        
        # Prepare labels
        y_icd, y_cpt = self.prepare_labels(icd_codes, cpt_codes)
        
        # Train ICD model
        self.logger.info("Training ICD model...")
        self.icd_model.fit(X, y_icd)
        
        # Train CPT model
        self.logger.info("Training CPT model...")
        self.cpt_model.fit(X, y_cpt)
        
        # Extract feature importance if available
        self._extract_feature_importance()
        
        self.is_trained = True
        self.logger.info("Model training completed!")
    
    def _extract_feature_importance(self):
        """Extract feature importance from trained models"""
        try:
            if hasattr(self.icd_model.estimators_[0], 'feature_importances_'):
                # Average feature importance across all estimators
                icd_importances = []
                cpt_importances = []
                
                for estimator in self.icd_model.estimators_:
                    icd_importances.append(estimator.feature_importances_)
                
                for estimator in self.cpt_model.estimators_:
                    cpt_importances.append(estimator.feature_importances_)
                
                self.feature_importance['icd'] = np.mean(icd_importances, axis=0)
                self.feature_importance['cpt'] = np.mean(cpt_importances, axis=0)
                
                self.logger.info("Feature importance extracted")
        except Exception as e:
            self.logger.warning(f"Could not extract feature importance: {e}")
    
    def predict(self, X: np.ndarray, top_k: int = 3, scale_features: bool = True) -> Dict:
        """Predict ICD and CPT codes for new samples"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction!")
        
        # Scale features if requested
        if scale_features:
            X = self.scaler.transform(X)
        
        # Get predictions and probabilities
        icd_pred = self.icd_model.predict(X)
        cpt_pred = self.cpt_model.predict(X)
        
        # Get prediction probabilities for top-k selection
        try:
            icd_proba = self.icd_model.predict_proba(X)
            cpt_proba = self.cpt_model.predict_proba(X)
            
            # Convert to top-k predictions
            icd_top_k = self._get_top_k_predictions(icd_proba, self.icd_binarizer.classes_, top_k)
            cpt_top_k = self._get_top_k_predictions(cpt_proba, self.cpt_binarizer.classes_, top_k)
            
        except Exception as e:
            self.logger.warning(f"Could not get probabilities: {e}")
            # Fallback to binary predictions
            icd_top_k = [self.icd_binarizer.inverse_transform(pred.reshape(1, -1))[0] 
                        for pred in icd_pred]
            cpt_top_k = [self.cpt_binarizer.inverse_transform(pred.reshape(1, -1))[0] 
                        for pred in cpt_pred]
        
        return {
            'icd_predictions': icd_top_k,
            'cpt_predictions': cpt_top_k,
            'icd_binary': icd_pred,
            'cpt_binary': cpt_pred
        }
    
    def _get_top_k_predictions(self, probabilities: List[np.ndarray], 
                              classes: np.ndarray, k: int) -> List[List[str]]:
        """Get top-k predictions from probability arrays"""
        top_k_predictions = []
        
        for sample_idx in range(len(probabilities[0])):
            # Collect probabilities for all classes for this sample
            sample_probs = []
            for class_idx, class_probs in enumerate(probabilities):
                # Get probability of positive class
                if len(class_probs.shape) > 1 and class_probs.shape[1] > 1:
                    prob = class_probs[sample_idx, 1]  # Positive class probability
                else:
                    prob = class_probs[sample_idx]
                sample_probs.append((prob, classes[class_idx]))
            
            # Sort by probability and get top-k
            sample_probs.sort(reverse=True, key=lambda x: x[0])
            top_k_codes = [code for _, code in sample_probs[:k]]
            top_k_predictions.append(top_k_codes)
        
        return top_k_predictions
    
    def evaluate(self, X_test: np.ndarray, icd_true: List[List[str]], 
                cpt_true: List[List[str]], scale_features: bool = True) -> Dict:
        """Evaluate model performance"""
        # Prepare test labels
        y_icd_true = self.icd_binarizer.transform(icd_true)
        y_cpt_true = self.cpt_binarizer.transform(cpt_true)
        
        # Get predictions
        predictions = self.predict(X_test, scale_features=scale_features)
        y_icd_pred = predictions['icd_binary']
        y_cpt_pred = predictions['cpt_binary']
        
        # Calculate metrics
        metrics = {
            'icd_accuracy': accuracy_score(y_icd_true, y_icd_pred),
            'cpt_accuracy': accuracy_score(y_cpt_true, y_cpt_pred),
            'icd_f1_micro': f1_score(y_icd_true, y_icd_pred, average='micro'),
            'cpt_f1_micro': f1_score(y_cpt_true, y_cpt_pred, average='micro'),
            'icd_f1_macro': f1_score(y_icd_true, y_icd_pred, average='macro'),
            'cpt_f1_macro': f1_score(y_cpt_true, y_cpt_pred, average='macro')
        }
        
        # Calculate top-k accuracy
        icd_top_k_acc = self._calculate_top_k_accuracy(
            predictions['icd_predictions'], icd_true, k=3
        )
        cpt_top_k_acc = self._calculate_top_k_accuracy(
            predictions['cpt_predictions'], cpt_true, k=3
        )
        
        metrics['icd_top3_accuracy'] = icd_top_k_acc
        metrics['cpt_top3_accuracy'] = cpt_top_k_acc
        
        return metrics
    
    def _calculate_top_k_accuracy(self, predictions: List[List[str]], 
                                 true_labels: List[List[str]], k: int) -> float:
        """Calculate top-k accuracy"""
        correct = 0
        total = 0
        
        for pred, true in zip(predictions, true_labels):
            for true_code in true:
                total += 1
                if true_code in pred[:k]:
                    correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def get_feature_importance(self, feature_names: List[str] = None, top_n: int = 20) -> Dict:
        """Get top feature importances"""
        if not self.feature_importance:
            return {}
        
        importance_dict = {}
        
        for code_type in ['icd', 'cpt']:
            if code_type in self.feature_importance:
                importances = self.feature_importance[code_type]
                
                if feature_names and len(feature_names) == len(importances):
                    # Create tuples of (feature_name, importance)
                    feature_imp = list(zip(feature_names, importances))
                else:
                    # Use indices as feature names
                    feature_imp = list(zip(range(len(importances)), importances))
                
                # Sort by importance and get top_n
                feature_imp.sort(key=lambda x: x[1], reverse=True)
                importance_dict[code_type] = feature_imp[:top_n]
        
        return importance_dict
    
    def save_model(self, filepath: str):
        """Save trained model to disk"""
        model_data = {
            'icd_model': self.icd_model,
            'cpt_model': self.cpt_model,
            'icd_binarizer': self.icd_binarizer,
            'cpt_binarizer': self.cpt_binarizer,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.icd_model = model_data['icd_model']
        self.cpt_model = model_data['cpt_model']
        self.icd_binarizer = model_data['icd_binarizer']
        self.cpt_binarizer = model_data['cpt_binarizer']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        self.feature_importance = model_data.get('feature_importance', {})
        
        self.logger.info(f"Model loaded from {filepath}")

class ModelTrainer:
    """Utility class for training and comparing multiple models"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def train_multiple_models(self, X_train: np.ndarray, X_test: np.ndarray,
                            icd_train: List[List[str]], icd_test: List[List[str]],
                            cpt_train: List[List[str]], cpt_test: List[List[str]]):
        """Train and evaluate multiple model types"""
        model_types = ['random_forest', 'gradient_boost', 'logistic']
        
        for model_type in model_types:
            self.logger.info(f"Training {model_type} model...")
            
            try:
                # Initialize and train model
                model = MedicalCodingModel(model_type=model_type)
                model.train(X_train, icd_train, cpt_train)
                
                # Evaluate model
                metrics = model.evaluate(X_test, icd_test, cpt_test)
                
                # Store results
                self.models[model_type] = model
                self.results[model_type] = metrics
                
                self.logger.info(f"{model_type} - ICD Top-3 Accuracy: {metrics['icd_top3_accuracy']:.3f}")
                self.logger.info(f"{model_type} - CPT Top-3 Accuracy: {metrics['cpt_top3_accuracy']:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}")
    
    def get_best_model(self, metric: str = 'icd_top3_accuracy') -> Tuple[str, MedicalCodingModel]:
        """Get the best performing model based on specified metric"""
        if not self.results:
            raise ValueError("No models have been trained yet!")
        
        best_score = -1
        best_model_name = None
        
        for model_name, metrics in self.results.items():
            if metric in metrics and metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = model_name
        
        return best_model_name, self.models[best_model_name]
    
    def print_comparison(self):
        """Print comparison of all trained models"""
        if not self.results:
            print("No models trained yet!")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        
        for model_name, metrics in self.results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  ICD Top-3 Accuracy: {metrics['icd_top3_accuracy']:.3f}")
            print(f"  CPT Top-3 Accuracy: {metrics['cpt_top3_accuracy']:.3f}")
            print(f"  ICD F1 (Micro):     {metrics['icd_f1_micro']:.3f}")
            print(f"  CPT F1 (Micro):     {metrics['cpt_f1_micro']:.3f}")
        
        # Identify best models
        best_icd = max(self.results.items(), key=lambda x: x[1]['icd_top3_accuracy'])
        best_cpt = max(self.results.items(), key=lambda x: x[1]['cpt_top3_accuracy'])
        
        print(f"\nBEST ICD MODEL: {best_icd[0]} ({best_icd[1]['icd_top3_accuracy']:.3f})")
        print(f"BEST CPT MODEL: {best_cpt[0]} ({best_cpt[1]['cpt_top3_accuracy']:.3f})")

def main():
    """Main function to demonstrate model training"""
    from data_preprocessing import MedicalTextPreprocessor
    from feature_extraction import MedicalFeatureExtractor
    
    # Initialize components
    preprocessor = MedicalTextPreprocessor()
    extractor = MedicalFeatureExtractor(max_features=5000)
    
    # Load and preprocess data
    data_dir = Path(__file__).parent.parent / 'data'
    df, code_mappings = preprocessor.load_and_preprocess_datasets(str(data_dir))
    
    # Split data
    splits = preprocessor.split_data(df)
    X_train_text, X_test_text, y_icd_train, y_icd_test, y_cpt_train, y_cpt_test = splits
    
    # Extract features
    train_features = extractor.fit_extract_features(X_train_text.tolist(), code_mappings)
    test_features = extractor.transform_features(X_test_text.tolist(), code_mappings)
    
    # Combine features
    X_train = extractor.combine_features(
        train_features, 
        feature_types=['tfidf', 'topics', 'medical_keywords', 'statistical']
    )
    X_test = extractor.combine_features(
        test_features, 
        feature_types=['tfidf', 'topics', 'medical_keywords', 'statistical']
    )
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_multiple_models(
        X_train, X_test,
        y_icd_train.tolist(), y_icd_test.tolist(),
        y_cpt_train.tolist(), y_cpt_test.tolist()
    )
    
    # Print results
    trainer.print_comparison()
    
    # Save best model
    best_name, best_model = trainer.get_best_model()
    
    models_dir = data_dir.parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    best_model.save_model(str(models_dir / f'best_model_{best_name}.pkl'))
    
    # Save feature names for interpretation
    with open(models_dir / 'feature_names.json', 'w') as f:
        json.dump(extractor.feature_names, f, indent=2)
    
    print(f"\nBest model ({best_name}) saved to models directory!")

if __name__ == "__main__":
    main()
