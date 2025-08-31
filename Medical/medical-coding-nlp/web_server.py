"""
Simple Flask Web Interface for Medical Coding System

This creates a basic web interface that runs on localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify
import json
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 Medical Coding Assistant</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 14px;
            resize: vertical;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 5px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .results {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .code-section {
            margin: 15px 0;
        }
        .code-section h3 {
            color: #333;
            margin-bottom: 10px;
        }
        .code-item {
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #28a745;
        }
        .sample-notes {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .sample-note {
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            border: 1px solid #ddd;
        }
        .sample-note:hover {
            background: #f0f0f0;
        }
        .loading {
            display: none;
            text-align: center;
            color: #667eea;
        }
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Medical Coding Assistant</h1>
        <p>NLP-Based ICD-10 & CPT Code Prediction System</p>
    </div>

    <div class="container">
        <h2>Clinical Note Input</h2>
        <textarea id="clinicalNote" placeholder="Enter clinical note here...

Example: Patient is a 65-year-old male with diabetes mellitus and hypertension. He presents with chest pain and shortness of breath. Physical examination reveals elevated blood pressure."></textarea>
        
        <div>
            <button class="btn" onclick="predictCodes()">🔍 Predict Medical Codes</button>
            <button class="btn" onclick="clearResults()">🗑️ Clear</button>
        </div>

        <div class="loading" id="loading">
            <p>🔄 Processing clinical note and predicting codes...</p>
        </div>

        <div id="results"></div>
    </div>

    <div class="container sample-notes">
        <h3>📋 Sample Clinical Notes (Click to Use)</h3>
        <div class="sample-note" onclick="loadSample(0)">
            <strong>Diabetes & Hypertension:</strong> Patient with type 2 diabetes mellitus and essential hypertension presents for routine follow-up.
        </div>
        <div class="sample-note" onclick="loadSample(1)">
            <strong>Chest Pain:</strong> 65-year-old male with coronary artery disease presents with chest pain. EKG shows ST changes.
        </div>
        <div class="sample-note" onclick="loadSample(2)">
            <strong>Pneumonia:</strong> Patient presents with fever, cough, and shortness of breath. Chest X-ray shows pneumonia.
        </div>
    </div>

    <script>
        const samples = [
            "Patient is a 45-year-old male with a history of type 2 diabetes mellitus and essential hypertension. He presents today for routine follow-up. Blood glucose levels have been well controlled on metformin. Blood pressure is elevated at 150/95. Will adjust antihypertensive medication.",
            "65-year-old male with history of coronary artery disease presents with acute chest pain. EKG shows ST segment changes. Cardiac enzymes are elevated. Diagnosed with myocardial infarction. Patient will be started on anticoagulation therapy.",
            "Patient presents with fever, productive cough, and shortness of breath for 3 days. Physical examination reveals crackles in right lower lobe. Chest X-ray shows right lower lobe pneumonia. Will start antibiotic therapy."
        ];

        function loadSample(index) {
            document.getElementById('clinicalNote').value = samples[index];
        }

        function clearResults() {
            document.getElementById('clinicalNote').value = '';
            document.getElementById('results').innerHTML = '';
        }

        async function predictCodes() {
            const noteText = document.getElementById('clinicalNote').value.trim();
            const resultsDiv = document.getElementById('results');
            const loadingDiv = document.getElementById('loading');

            if (!noteText) {
                resultsDiv.innerHTML = '<div class="error">⚠️ Please enter a clinical note.</div>';
                return;
            }

            // Show loading
            loadingDiv.style.display = 'block';
            resultsDiv.innerHTML = '';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: noteText })
                });

                const data = await response.json();
                loadingDiv.style.display = 'none';

                if (data.success) {
                    displayResults(data.result);
                } else {
                    resultsDiv.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                }
            } catch (error) {
                loadingDiv.style.display = 'none';
                resultsDiv.innerHTML = `<div class="error">❌ Network error: ${error.message}</div>`;
            }
        }

        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            
            let html = '<div class="results">';
            html += '<h3>🎯 Prediction Results</h3>';
            
            // ICD Codes
            html += '<div class="code-section">';
            html += '<h3>🏥 ICD-10 Diagnosis Codes</h3>';
            if (result.icd_codes && result.icd_codes.length > 0) {
                result.icd_codes.forEach((code, index) => {
                    const desc = result.icd_descriptions ? result.icd_descriptions[index] : 'Description not available';
                    html += `<div class="code-item"><strong>${code}</strong> - ${desc}</div>`;
                });
            } else {
                html += '<div class="code-item">No ICD codes predicted</div>';
            }
            html += '</div>';
            
            // CPT Codes
            html += '<div class="code-section">';
            html += '<h3>⚕️ CPT Procedure Codes</h3>';
            if (result.cpt_codes && result.cpt_codes.length > 0) {
                result.cpt_codes.forEach((code, index) => {
                    const desc = result.cpt_descriptions ? result.cpt_descriptions[index] : 'Description not available';
                    html += `<div class="code-item"><strong>${code}</strong> - ${desc}</div>`;
                });
            } else {
                html += '<div class="code-item">No CPT codes predicted</div>';
            }
            html += '</div>';
            
            html += '</div>';
            resultsDiv.innerHTML = html;
        }

        // Allow Enter key to trigger prediction
        document.getElementById('clinicalNote').addEventListener('keydown', function(event) {
            if (event.ctrlKey && event.key === 'Enter') {
                predictCodes();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict medical codes"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        # Import and use the demo system
        sys.path.append(str(Path(__file__).parent))
        from demo import MedicalCodingSystem, load_data
        
        # Load data and train model (in production, this would be pre-trained)
        df, code_mappings = load_data()
        system = MedicalCodingSystem()
        system.train(df, code_mappings)
        
        # Make prediction
        result = system.predict(text)
        
        return jsonify({
            'success': True,
            'result': {
                'icd_codes': result['icd_codes'],
                'cpt_codes': result['cpt_codes'],
                'icd_descriptions': result['icd_descriptions'],
                'cpt_descriptions': result['cpt_descriptions']
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Medical Coding System is running'})

if __name__ == '__main__':
    print("🏥 Medical Coding System - Web Interface")
    print("=" * 50)
    print("Starting web server...")
    print("")
    print("🌐 Open your browser and go to:")
    print("   http://localhost:5000")
    print("")
    print("📝 Features:")
    print("   • Enter clinical notes for real-time code prediction")
    print("   • Sample notes provided for testing")
    print("   • ICD-10 diagnosis code suggestions")
    print("   • CPT procedure code suggestions")
    print("")
    print("💡 Tip: Use Ctrl+Enter in the text area for quick prediction")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
