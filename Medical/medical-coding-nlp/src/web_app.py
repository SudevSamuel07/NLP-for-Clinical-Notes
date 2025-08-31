"""
Web Application for Medical Coding System

This module provides a Flask-based web interface for the medical coding prediction system.
Users can input clinical notes and get real-time ICD/CPT code suggestions.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import json
import logging
from pathlib import Path
import traceback
from typing import Dict, List
import os

# Import our modules
from prediction import MedicalCodingPredictor
from data_preprocessing import MedicalTextPreprocessor
from feature_extraction import MedicalFeatureExtractor
from model_training import MedicalCodingModel

app = Flask(__name__)
app.secret_key = 'medical_coding_secret_key_2024'
CORS(app)

# Global variables for models
predictor = None
models_loaded = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_models():
    """Load trained models and components"""
    global predictor, models_loaded
    
    try:
        # Path to models directory
        models_dir = Path(__file__).parent.parent / 'models'
        
        # Check if models exist
        model_files = list(models_dir.glob('best_model_*.pkl'))
        extractor_file = models_dir / 'feature_extractors.pkl'
        
        if not model_files or not extractor_file.exists():
            logger.warning("Models not found. Please train models first.")
            return False
        
        # Load the first available model
        model_path = str(model_files[0])
        extractors_path = str(extractor_file)
        
        predictor = MedicalCodingPredictor(
            model_path=model_path,
            extractors_path=extractors_path
        )
        
        # Load additional components if available
        try:
            predictor.load_code_mappings(str(models_dir / 'code_mappings.json'))
        except FileNotFoundError:
            logger.warning("Code mappings not found")
        
        try:
            predictor.load_feature_names(str(models_dir / 'feature_names.json'))
        except FileNotFoundError:
            logger.warning("Feature names not found")
        
        models_loaded = True
        logger.info("Models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict ICD/CPT codes for clinical note"""
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please train models first.',
                'success': False
            })
        
        # Get input data
        data = request.get_json()
        clinical_note = data.get('clinical_note', '').strip()
        top_k = int(data.get('top_k', 3))
        include_explanation = data.get('include_explanation', False)
        
        if not clinical_note:
            return jsonify({
                'error': 'Please provide a clinical note.',
                'success': False
            })
        
        # Make prediction
        if include_explanation:
            result = predictor.explain_prediction(clinical_note, top_features=10)
        else:
            result = predictor.predict_single_note(clinical_note, top_k=top_k)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple clinical notes"""
    try:
        if not models_loaded:
            return jsonify({
                'error': 'Models not loaded. Please train models first.',
                'success': False
            })
        
        # Get input data
        data = request.get_json()
        clinical_notes = data.get('clinical_notes', [])
        top_k = int(data.get('top_k', 3))
        
        if not clinical_notes:
            return jsonify({
                'error': 'Please provide clinical notes.',
                'success': False
            })
        
        # Make batch prediction
        results = predictor.predict_batch(clinical_notes, top_k=top_k)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'success': False
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process file with clinical notes"""
    try:
        if not models_loaded:
            flash('Models not loaded. Please train models first.', 'error')
            return redirect(url_for('index'))
        
        if 'file' not in request.files:
            flash('No file selected.', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'error')
            return redirect(url_for('index'))
        
        if file and file.filename.endswith(('.csv', '.json')):
            # Save uploaded file temporarily
            upload_dir = Path(__file__).parent.parent / 'uploads'
            upload_dir.mkdir(exist_ok=True)
            
            file_path = upload_dir / file.filename
            file.save(str(file_path))
            
            # Process file
            top_k = int(request.form.get('top_k', 3))
            text_column = request.form.get('text_column', 'text')
            
            # Generate output filename
            output_path = upload_dir / f'predictions_{file.filename}'
            
            results_df = predictor.predict_from_file(
                str(file_path), 
                str(output_path), 
                text_column=text_column,
                top_k=top_k
            )
            
            # Clean up input file
            file_path.unlink()
            
            flash(f'File processed successfully! {len(results_df)} notes processed.', 'success')
            
            # Return download link
            return render_template('download.html', 
                                 filename=f'predictions_{file.filename}',
                                 results_count=len(results_df))
        
        else:
            flash('Please upload a CSV or JSON file.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"File upload error: {e}")
        flash(f'File processing failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed results file"""
    try:
        upload_dir = Path(__file__).parent.parent / 'uploads'
        file_path = upload_dir / filename
        
        if file_path.exists():
            from flask import send_file
            return send_file(str(file_path), as_attachment=True)
        else:
            flash('File not found.', 'error')
            return redirect(url_for('index'))
            
    except Exception as e:
        logger.error(f"Download error: {e}")
        flash(f'Download failed: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/train_models', methods=['POST'])
def train_models():
    """Train new models (development endpoint)"""
    try:
        # This would trigger model training
        # For production, this should be secured or removed
        from model_training import main as train_main
        
        # Run training in background (simplified for demo)
        train_main()
        
        # Reload models
        global models_loaded
        models_loaded = load_models()
        
        return jsonify({
            'success': True,
            'message': 'Models trained successfully!'
        })
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({
            'error': f'Training failed: {str(e)}',
            'success': False
        })

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': models_loaded,
        'version': '1.0.0'
    })

@app.route('/api/codes')
def get_code_mappings():
    """Get available ICD/CPT codes"""
    try:
        if not models_loaded or not predictor.code_mappings:
            return jsonify({
                'error': 'Code mappings not available',
                'success': False
            })
        
        return jsonify({
            'success': True,
            'icd_codes': predictor.code_mappings.get('icd10', {}),
            'cpt_codes': predictor.code_mappings.get('cpt', {})
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

def create_templates():
    """Create HTML templates for the web application"""
    templates_dir = Path(__file__).parent.parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Main page template
    index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Coding Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .result-card { margin-top: 20px; }
        .code-badge { margin: 2px; }
        .entity-highlight { background-color: #fffacd; padding: 2px 4px; border-radius: 3px; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="#">🏥 Medical Coding Assistant</a>
        </div>
    </nav>

    <div class="container mt-4">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else 'success' }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5>Clinical Note Input</h5>
                    </div>
                    <div class="card-body">
                        <form id="predictionForm">
                            <div class="mb-3">
                                <label for="clinicalNote" class="form-label">Clinical Note:</label>
                                <textarea class="form-control" id="clinicalNote" rows="8" 
                                         placeholder="Enter clinical note here..."></textarea>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <label for="topK" class="form-label">Number of suggestions:</label>
                                    <select class="form-select" id="topK">
                                        <option value="1">Top 1</option>
                                        <option value="3" selected>Top 3</option>
                                        <option value="5">Top 5</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <div class="form-check mt-4">
                                        <input class="form-check-input" type="checkbox" id="includeExplanation">
                                        <label class="form-check-label" for="includeExplanation">
                                            Include explanation
                                        </label>
                                    </div>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-primary mt-3">
                                <span id="loadingSpinner" class="spinner-border spinner-border-sm d-none"></span>
                                Predict Codes
                            </button>
                        </form>
                    </div>
                </div>

                <div class="card mt-4">
                    <div class="card-header">
                        <h5>File Upload</h5>
                    </div>
                    <div class="card-body">
                        <form action="/upload" method="post" enctype="multipart/form-data">
                            <div class="mb-3">
                                <label for="file" class="form-label">Upload CSV/JSON file:</label>
                                <input class="form-control" type="file" id="file" name="file" accept=".csv,.json">
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <label for="textColumn" class="form-label">Text column name:</label>
                                    <input type="text" class="form-control" id="textColumn" name="text_column" value="text">
                                </div>
                                <div class="col-md-6">
                                    <label for="fileTopK" class="form-label">Number of suggestions:</label>
                                    <select class="form-select" id="fileTopK" name="top_k">
                                        <option value="1">Top 1</option>
                                        <option value="3" selected>Top 3</option>
                                        <option value="5">Top 5</option>
                                    </select>
                                </div>
                            </div>
                            <button type="submit" class="btn btn-success mt-3">Upload and Process</button>
                        </form>
                    </div>
                </div>
            </div>

            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>System Status</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Models Status:</strong> 
                            <span class="badge bg-{{ 'success' if models_loaded else 'danger' }}">
                                {{ 'Loaded' if models_loaded else 'Not Loaded' }}
                            </span>
                        </p>
                        {% if not models_loaded %}
                            <button class="btn btn-warning btn-sm" onclick="trainModels()">Train Models</button>
                        {% endif %}
                    </div>
                </div>

                <div class="card mt-3">
                    <div class="card-header">
                        <h5>Sample Clinical Notes</h5>
                    </div>
                    <div class="card-body">
                        <button class="btn btn-outline-secondary btn-sm mb-2" onclick="loadSample(1)">
                            Sample 1: Diabetes
                        </button><br>
                        <button class="btn btn-outline-secondary btn-sm mb-2" onclick="loadSample(2)">
                            Sample 2: Hypertension
                        </button><br>
                        <button class="btn btn-outline-secondary btn-sm mb-2" onclick="loadSample(3)">
                            Sample 3: Pneumonia
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <div id="results" class="mt-4"></div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        const samples = {
            1: "Patient is a 65-year-old male with a history of type 2 diabetes mellitus without complications. He presents today for routine follow-up. Blood glucose levels have been well controlled on metformin. Physical examination is unremarkable. Will continue current diabetes management.",
            2: "67-year-old female with essential hypertension presents for blood pressure check. Current BP is 145/92. Patient has been compliant with lisinopril therapy. Will increase dosage and schedule follow-up in 4 weeks.",
            3: "45-year-old male presents with fever, productive cough, and shortness of breath for 3 days. Physical exam reveals crackles in right lower lobe. Chest X-ray shows right lower lobe pneumonia. Will start antibiotic therapy."
        };

        function loadSample(sampleNum) {
            document.getElementById('clinicalNote').value = samples[sampleNum];
        }

        function trainModels() {
            fetch('/train_models', {method: 'POST'})
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Models trained successfully! Reloading page...');
                        location.reload();
                    } else {
                        alert('Training failed: ' + data.error);
                    }
                });
        }

        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const clinicalNote = document.getElementById('clinicalNote').value;
            const topK = document.getElementById('topK').value;
            const includeExplanation = document.getElementById('includeExplanation').checked;
            
            if (!clinicalNote.trim()) {
                alert('Please enter a clinical note.');
                return;
            }

            const spinner = document.getElementById('loadingSpinner');
            spinner.classList.remove('d-none');

            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    clinical_note: clinicalNote,
                    top_k: parseInt(topK),
                    include_explanation: includeExplanation
                })
            })
            .then(response => response.json())
            .then(data => {
                spinner.classList.add('d-none');
                
                if (data.success) {
                    displayResults(data.result, includeExplanation);
                } else {
                    document.getElementById('results').innerHTML = 
                        '<div class="alert alert-danger">Error: ' + data.error + '</div>';
                }
            })
            .catch(error => {
                spinner.classList.add('d-none');
                document.getElementById('results').innerHTML = 
                    '<div class="alert alert-danger">Network error: ' + error + '</div>';
            });
        });

        function displayResults(result, includeExplanation) {
            let html = '<div class="card result-card"><div class="card-header"><h5>Prediction Results</h5></div><div class="card-body">';
            
            // ICD Codes
            html += '<h6>ICD-10 Diagnosis Codes:</h6>';
            if (includeExplanation && result.prediction) {
                const predictions = result.prediction.predictions;
                predictions.icd_codes.forEach((code, index) => {
                    const desc = predictions.icd_descriptions ? predictions.icd_descriptions[index] : '';
                    html += '<span class="badge bg-primary code-badge">' + code + '</span>';
                    if (desc) html += '<small class="text-muted"> - ' + desc + '</small><br>';
                });
            } else {
                result.predictions.icd_codes.forEach((code, index) => {
                    const desc = result.predictions.icd_descriptions ? result.predictions.icd_descriptions[index] : '';
                    html += '<span class="badge bg-primary code-badge">' + code + '</span>';
                    if (desc) html += '<small class="text-muted"> - ' + desc + '</small><br>';
                });
            }
            
            html += '<br><h6>CPT Procedure Codes:</h6>';
            if (includeExplanation && result.prediction) {
                const predictions = result.prediction.predictions;
                predictions.cpt_codes.forEach((code, index) => {
                    const desc = predictions.cpt_descriptions ? predictions.cpt_descriptions[index] : '';
                    html += '<span class="badge bg-success code-badge">' + code + '</span>';
                    if (desc) html += '<small class="text-muted"> - ' + desc + '</small><br>';
                });
            } else {
                result.predictions.cpt_codes.forEach((code, index) => {
                    const desc = result.predictions.cpt_descriptions ? result.predictions.cpt_descriptions[index] : '';
                    html += '<span class="badge bg-success code-badge">' + code + '</span>';
                    if (desc) html += '<small class="text-muted"> - ' + desc + '</small><br>';
                });
            }
            
            // Extracted entities
            const entities = includeExplanation ? result.prediction.extracted_entities : result.extracted_entities;
            if (entities && Object.keys(entities).length > 0) {
                html += '<br><h6>Extracted Medical Entities:</h6>';
                for (const [category, entityList] of Object.entries(entities)) {
                    if (entityList.length > 0) {
                        html += '<strong>' + category.charAt(0).toUpperCase() + category.slice(1) + ':</strong> ';
                        html += entityList.map(e => '<span class="entity-highlight">' + e + '</span>').join(', ') + '<br>';
                    }
                }
            }
            
            html += '</div></div>';
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
    '''
    
    with open(templates_dir / 'index.html', 'w') as f:
        f.write(index_html)
    
    # Error template
    error_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Medical Coding Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body text-center">
                        <h1 class="display-1">{{ error_code }}</h1>
                        <h4>{{ error_message }}</h4>
                        <a href="{{ url_for('index') }}" class="btn btn-primary mt-3">Go Home</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    '''
    
    with open(templates_dir / 'error.html', 'w') as f:
        f.write(error_html)
    
    # Download template
    download_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Download Results - Medical Coding Assistant</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-body text-center">
                        <h4>Processing Complete!</h4>
                        <p>Successfully processed {{ results_count }} clinical notes.</p>
                        <a href="{{ url_for('download_file', filename=filename) }}" 
                           class="btn btn-success">Download Results</a>
                        <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Home</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    '''
    
    with open(templates_dir / 'download.html', 'w') as f:
        f.write(download_html)

if __name__ == '__main__':
    # Create templates if they don't exist
    create_templates()
    
    # Try to load models
    load_models()
    
    # Create necessary directories
    upload_dir = Path(__file__).parent.parent / 'uploads'
    upload_dir.mkdir(exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)
