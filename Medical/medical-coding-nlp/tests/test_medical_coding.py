"""
Test Suite for Medical Coding System

This module contains unit tests for the medical coding system components.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_preprocessing import MedicalTextPreprocessor
from feature_extraction import MedicalFeatureExtractor
from model_training import MedicalCodingModel
from prediction import MedicalCodingPredictor

class TestMedicalTextPreprocessor(unittest.TestCase):
    """Test cases for MedicalTextPreprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = MedicalTextPreprocessor()
        self.sample_text = "Patient is a 65-year-old male with DM and HTN. Presents with SOB."
    
    def test_clean_text(self):
        """Test text cleaning functionality"""
        cleaned = self.preprocessor.clean_text(self.sample_text)
        
        # Should expand abbreviations
        self.assertIn('diabetes mellitus', cleaned)
        self.assertIn('hypertension', cleaned)
        self.assertIn('shortness of breath', cleaned)
        
        # Should be lowercase
        self.assertEqual(cleaned, cleaned.lower())
    
    def test_extract_medical_entities(self):
        """Test medical entity extraction"""
        entities = self.preprocessor.extract_medical_entities(self.sample_text)
        
        # Should return dictionary with expected keys
        expected_keys = ['conditions', 'medications', 'procedures', 'symptoms', 'body_parts']
        for key in expected_keys:
            self.assertIn(key, entities)
            self.assertIsInstance(entities[key], list)
    
    def test_tokenize_and_filter(self):
        """Test tokenization and filtering"""
        tokens = self.preprocessor.tokenize_and_filter(self.sample_text)
        
        # Should return list of tokens
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)
        
        # Tokens should be lowercase and alphanumeric
        for token in tokens:
            self.assertTrue(token.isalnum())
            self.assertEqual(token, token.lower())

class TestMedicalFeatureExtractor(unittest.TestCase):
    """Test cases for MedicalFeatureExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.extractor = MedicalFeatureExtractor(max_features=100)
        self.sample_texts = [
            "Patient with diabetes mellitus",
            "Patient with hypertension and chest pain",
            "Patient presents for routine examination"
        ]
    
    def test_extract_medical_keyword_features(self):
        """Test medical keyword feature extraction"""
        features = self.extractor.extract_medical_keyword_features(self.sample_texts)
        
        # Should return numpy array
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.sample_texts))
        
        # Features should be binary (0 or 1)
        self.assertTrue(np.all(np.isin(features, [0, 1])))
    
    def test_extract_statistical_features(self):
        """Test statistical feature extraction"""
        features = self.extractor.extract_statistical_features(self.sample_texts)
        
        # Should return numpy array with correct shape
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.sample_texts))
        self.assertTrue(features.shape[1] > 0)
    
    def test_fit_extract_features(self):
        """Test feature fitting and extraction"""
        features = self.extractor.fit_extract_features(self.sample_texts)
        
        # Should return dictionary of feature arrays
        self.assertIsInstance(features, dict)
        self.assertTrue(len(features) > 0)
        
        # Each feature type should have correct number of samples
        for feature_type, feature_array in features.items():
            self.assertEqual(feature_array.shape[0], len(self.sample_texts))

class TestMedicalCodingModel(unittest.TestCase):
    """Test cases for MedicalCodingModel"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = MedicalCodingModel(model_type='random_forest')
        
        # Create sample training data
        self.X_train = np.random.rand(10, 20)
        self.icd_codes = [['E11.9'], ['I10'], ['E11.9'], ['I10'], ['E11.9'],
                         ['I10'], ['E11.9'], ['I10'], ['E11.9'], ['I10']]
        self.cpt_codes = [['99213'], ['99214'], ['99213'], ['99214'], ['99213'],
                         ['99214'], ['99213'], ['99214'], ['99213'], ['99214']]
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model.icd_model)
        self.assertIsNotNone(self.model.cpt_model)
        self.assertFalse(self.model.is_trained)
    
    def test_prepare_labels(self):
        """Test label preparation"""
        icd_binary, cpt_binary = self.model.prepare_labels(self.icd_codes, self.cpt_codes)
        
        # Should return binary matrices
        self.assertIsInstance(icd_binary, np.ndarray)
        self.assertIsInstance(cpt_binary, np.ndarray)
        self.assertEqual(icd_binary.shape[0], len(self.icd_codes))
        self.assertEqual(cpt_binary.shape[0], len(self.cpt_codes))
    
    def test_training(self):
        """Test model training"""
        self.model.train(self.X_train, self.icd_codes, self.cpt_codes)
        
        # Model should be trained
        self.assertTrue(self.model.is_trained)
    
    def test_prediction_after_training(self):
        """Test prediction after training"""
        # Train the model first
        self.model.train(self.X_train, self.icd_codes, self.cpt_codes)
        
        # Make predictions
        X_test = np.random.rand(3, 20)
        predictions = self.model.predict(X_test, top_k=2)
        
        # Should return prediction dictionary
        self.assertIsInstance(predictions, dict)
        self.assertIn('icd_predictions', predictions)
        self.assertIn('cpt_predictions', predictions)
        
        # Should have correct number of predictions
        self.assertEqual(len(predictions['icd_predictions']), 3)
        self.assertEqual(len(predictions['cpt_predictions']), 3)

class TestMedicalCodingPredictor(unittest.TestCase):
    """Test cases for MedicalCodingPredictor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = MedicalCodingPredictor()
        self.sample_note = "Patient is a 65-year-old male with diabetes mellitus presenting for follow-up."
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        self.assertIsNotNone(self.predictor.preprocessor)
        self.assertIsNotNone(self.predictor.extractor)
        self.assertIsNotNone(self.predictor.model)
    
    def test_predict_single_note_without_trained_model(self):
        """Test prediction without trained model (should handle gracefully)"""
        # This should not crash but may not give meaningful results
        try:
            result = self.predictor.predict_single_note(self.sample_note)
            # If it doesn't crash, check that it returns a dict
            self.assertIsInstance(result, dict)
        except ValueError:
            # Expected if model is not trained
            pass

class TestDataIntegration(unittest.TestCase):
    """Integration tests for data flow"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_data = pd.DataFrame({
            'text': [
                "Patient with type 2 diabetes mellitus",
                "Patient with essential hypertension",
                "Patient presents with pneumonia"
            ],
            'icd_codes': [['E11.9'], ['I10'], ['J18.9']],
            'cpt_codes': [['99213'], ['99214'], ['99214']]
        })
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from preprocessing to prediction"""
        # Initialize components
        preprocessor = MedicalTextPreprocessor()
        extractor = MedicalFeatureExtractor(max_features=50)
        model = MedicalCodingModel(model_type='random_forest')
        
        # Preprocess data
        processed_df = preprocessor.preprocess_clinical_notes(self.sample_data)
        
        # Extract features
        texts = processed_df['cleaned_text'].tolist()
        features = extractor.fit_extract_features(texts)
        X = extractor.combine_features(features)
        
        # Train model
        icd_codes = self.sample_data['icd_codes'].tolist()
        cpt_codes = self.sample_data['cpt_codes'].tolist()
        model.train(X, icd_codes, cpt_codes)
        
        # Make predictions
        predictions = model.predict(X, top_k=1)
        
        # Verify results
        self.assertIsInstance(predictions, dict)
        self.assertIn('icd_predictions', predictions)
        self.assertIn('cpt_predictions', predictions)

class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_code_validation(self):
        """Test medical code validation"""
        from utils.helpers import validate_medical_codes
        
        # Test valid ICD codes
        valid_icd = ['E11.9', 'I10', 'J18.9']
        icd_results = validate_medical_codes(valid_icd, 'icd10')
        self.assertTrue(all(icd_results))
        
        # Test valid CPT codes
        valid_cpt = ['99213', '99214', '71020']
        cpt_results = validate_medical_codes(valid_cpt, 'cpt')
        self.assertTrue(all(cpt_results))
        
        # Test invalid codes
        invalid_codes = ['invalid', '123', 'xyz']
        invalid_results = validate_medical_codes(invalid_codes, 'icd10')
        self.assertFalse(any(invalid_results))

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestMedicalTextPreprocessor,
        TestMedicalFeatureExtractor,
        TestMedicalCodingModel,
        TestMedicalCodingPredictor,
        TestDataIntegration,
        TestUtilityFunctions
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("Running Medical Coding System Test Suite")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    exit(0 if success else 1)
