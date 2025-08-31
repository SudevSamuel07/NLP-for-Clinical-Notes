"""
Main entry point for the Medical Coding System

This script runs the complete pipeline: data collection, preprocessing,
feature extraction, model training, and evaluation.
"""

import sys
import logging
from pathlib import Path
import argparse

# Add src directory to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root / 'src'))
sys.path.append(str(project_root / 'utils'))

def setup_project():
    """Setup project directories and logging"""
    # Create necessary directories
    directories = ['data/raw', 'data/processed', 'data/reference', 'models', 'uploads', 'logs']
    
    for directory in directories:
        (project_root / directory).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = project_root / 'logs' / 'medical_coding.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Project setup completed")
    return logger

def collect_data():
    """Collect and prepare datasets"""
    logger = logging.getLogger(__name__)
    logger.info("Starting data collection...")
    
    try:
        from data_collection import MedicalDataCollector
        
        collector = MedicalDataCollector(str(project_root / 'data'))
        collector.collect_all_data()
        
        logger.info("Data collection completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        return False

def train_models():
    """Train the medical coding models"""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training...")
    
    try:
        from data_preprocessing import MedicalTextPreprocessor
        from feature_extraction import MedicalFeatureExtractor
        from model_training import ModelTrainer
        
        # Initialize components
        preprocessor = MedicalTextPreprocessor()
        extractor = MedicalFeatureExtractor(max_features=5000)
        trainer = ModelTrainer()
        
        # Load and preprocess data
        data_dir = project_root / 'data'
        df, code_mappings = preprocessor.load_and_preprocess_datasets(str(data_dir))
        
        logger.info(f"Loaded {len(df)} clinical notes for training")
        
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
        
        logger.info(f"Training features shape: {X_train.shape}")
        logger.info(f"Testing features shape: {X_test.shape}")
        
        # Train models
        trainer.train_multiple_models(
            X_train, X_test,
            y_icd_train.tolist(), y_icd_test.tolist(),
            y_cpt_train.tolist(), y_cpt_test.tolist()
        )
        
        # Print results
        trainer.print_comparison()
        
        # Save best model
        best_name, best_model = trainer.get_best_model()
        
        models_dir = project_root / 'models'
        best_model.save_model(str(models_dir / f'best_model_{best_name}.pkl'))
        
        # Save feature extractors and mappings
        extractor.save_extractors(str(models_dir / 'feature_extractors.pkl'))
        
        import json
        with open(models_dir / 'feature_names.json', 'w') as f:
            json.dump(extractor.feature_names, f, indent=2)
        
        with open(models_dir / 'code_mappings.json', 'w') as f:
            json.dump(code_mappings, f, indent=2)
        
        logger.info(f"Best model ({best_name}) saved successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        return False

def run_predictions(note_text: str = None):
    """Run predictions on sample or provided text"""
    logger = logging.getLogger(__name__)
    logger.info("Running predictions...")
    
    try:
        from prediction import MedicalCodingPredictor
        
        # Load trained components
        models_dir = project_root / 'models'
        model_files = list(models_dir.glob('best_model_*.pkl'))
        
        if not model_files:
            logger.error("No trained models found. Please train models first.")
            return False
        
        predictor = MedicalCodingPredictor(
            model_path=str(model_files[0]),
            extractors_path=str(models_dir / 'feature_extractors.pkl')
        )
        
        # Load additional components
        try:
            predictor.load_code_mappings(str(models_dir / 'code_mappings.json'))
            predictor.load_feature_names(str(models_dir / 'feature_names.json'))
        except FileNotFoundError:
            logger.warning("Some model components not found")
        
        # Use provided text or sample
        if note_text is None:
            note_text = """
            Patient is a 65-year-old male with a history of type 2 diabetes mellitus and hypertension.
            He presents today with chest pain and shortness of breath. Physical examination reveals
            elevated blood pressure and irregular heart rhythm. EKG shows atrial fibrillation.
            Laboratory results show elevated glucose levels. Patient will be started on anticoagulation
            therapy and diabetes management will be optimized.
            """
        
        # Make prediction
        result = predictor.predict_single_note(note_text, top_k=3)
        
        # Display results
        print("\n" + "="*60)
        print("MEDICAL CODING PREDICTION RESULTS")
        print("="*60)
        print(f"\nClinical Note:")
        print(note_text.strip())
        print(f"\nPredicted ICD-10 Codes:")
        for i, code in enumerate(result['predictions']['icd_codes'], 1):
            desc = result['predictions'].get('icd_descriptions', [''])[i-1] if 'icd_descriptions' in result['predictions'] else ''
            print(f"  {i}. {code} - {desc}")
        
        print(f"\nPredicted CPT Codes:")
        for i, code in enumerate(result['predictions']['cpt_codes'], 1):
            desc = result['predictions'].get('cpt_descriptions', [''])[i-1] if 'cpt_descriptions' in result['predictions'] else ''
            print(f"  {i}. {code} - {desc}")
        
        print(f"\nExtracted Medical Entities:")
        entities = result['extracted_entities']
        for category, entity_list in entities.items():
            if entity_list:
                print(f"  {category.title()}: {', '.join(entity_list)}")
        
        print("="*60)
        
        logger.info("Prediction completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False

def run_web_app():
    """Start the web application"""
    logger = logging.getLogger(__name__)
    logger.info("Starting web application...")
    
    try:
        from web_app import app
        
        # Check if models exist
        models_dir = project_root / 'models'
        model_files = list(models_dir.glob('best_model_*.pkl'))
        
        if not model_files:
            logger.warning("No trained models found. Web app will run with limited functionality.")
        
        print("\n" + "="*60)
        print("MEDICAL CODING WEB APPLICATION")
        print("="*60)
        print("Starting web server...")
        print("Access the application at: http://localhost:5000")
        print("Press Ctrl+C to stop the server")
        print("="*60)
        
        app.run(debug=False, host='0.0.0.0', port=5000)
        
        return True
        
    except Exception as e:
        logger.error(f"Web application failed: {e}")
        return False

def run_tests():
    """Run the test suite"""
    logger = logging.getLogger(__name__)
    logger.info("Running test suite...")
    
    try:
        from test_medical_coding import run_tests
        
        success = run_tests()
        
        if success:
            logger.info("All tests passed!")
        else:
            logger.error("Some tests failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(description='Medical Coding System')
    parser.add_argument('command', choices=[
        'setup', 'collect-data', 'train', 'predict', 'web', 'test', 'all'
    ], help='Command to execute')
    parser.add_argument('--text', help='Clinical note text for prediction')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup project
    logger = setup_project()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    success = True
    
    if args.command == 'setup':
        logger.info("Project setup completed!")
        
    elif args.command == 'collect-data':
        success = collect_data()
        
    elif args.command == 'train':
        success = train_models()
        
    elif args.command == 'predict':
        success = run_predictions(args.text)
        
    elif args.command == 'web':
        success = run_web_app()
        
    elif args.command == 'test':
        success = run_tests()
        
    elif args.command == 'all':
        logger.info("Running complete pipeline...")
        
        # Run all steps
        steps = [
            ('Collecting data', collect_data),
            ('Training models', train_models),
            ('Running sample prediction', lambda: run_predictions()),
            ('Running tests', run_tests)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            if not step_func():
                logger.error(f"Failed at step: {step_name}")
                success = False
                break
            logger.info(f"Completed: {step_name}")
        
        if success:
            logger.info("Complete pipeline executed successfully!")
            print("\nTo start the web application, run:")
            print("python main.py web")
    
    if success:
        logger.info("Command completed successfully!")
    else:
        logger.error("Command failed!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
