document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Script loaded - starting initialization...');
    
    // DOM Elements
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loading = document.getElementById('loading');
    const overallResult = document.getElementById('overallResult');
    const detailedResults = document.getElementById('detailedResults');
    
    // State
    let currentFile = null;
    
    // Initialize
    init();
    
    function init() {
        setupEventListeners();
        checkServerHealth();
    }
    
    function setupEventListeners() {
        uploadBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);
        analyzeBtn.addEventListener('click', (e) => {
            e.preventDefault();
            analyzeImage();
        });
        clearBtn.addEventListener('click', clearAll);
        setupDragAndDrop();
    }
    
    function setupDragAndDrop() {
        const uploadArea = document.getElementById('uploadArea');
        if (!uploadArea) return;
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#3498db';
            uploadArea.style.backgroundColor = '#f8f9fa';
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.style.borderColor = '#bdc3c7';
            uploadArea.style.backgroundColor = 'transparent';
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.style.borderColor = '#bdc3c7';
            uploadArea.style.backgroundColor = 'transparent';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });
    }
    
    function handleFileSelect() {
        if (fileInput.files.length > 0) {
            currentFile = fileInput.files[0];
            console.log('📄 File selected:', currentFile.name);
            const reader = new FileReader();
            reader.onload = function(event) {
                imagePreview.innerHTML = `
                    <div class="image-container">
                        <img src="${event.target.result}" alt="X-Ray Preview">
                        <div class="image-info">
                            <span>${currentFile.name}</span>
                            <span>${(currentFile.size / 1024 / 1024).toFixed(2)} MB</span>
                        </div>
                    </div>
                `;
                analyzeBtn.style.display = 'block';
                clearBtn.style.display = 'block';
                hideResults();
            };
            reader.readAsDataURL(currentFile);
        }
    }
    
    async function analyzeImage() {
        console.log('🔍 Starting analysis...');
        if (!currentFile) {
            showError('Please select an X-ray image first.');
            return;
        }
        showLoading();
        try {
            const formData = new FormData();
            formData.append('file', currentFile);
            console.log('📤 Sending request to http://localhost:5500/predict...');
            const response = await fetch('http://localhost:5500/predict', {
                method: 'POST',
                body: formData
            });
            console.log('📥 Prediction response status:', response.status);
            if (!response.ok) {
                const text = await response.text();
                throw new Error(`Server error: ${response.status} - ${text}`);
            }
            const data = await response.json();
            console.log('✅ Prediction success:', data);
            if (data.status === 'success') {
                displayResults(data);
            } else {
                throw new Error(data.error || 'Analysis failed');
            }
        } catch (error) {
            console.error('❌ Analysis failed:', error);
            showError(`Analysis failed: ${error.message}`);
        } finally {
            hideLoading();
        }
    }
    
    // function displayResults(data) {
    //     console.log('📊 Displaying results:', data);
    //     overallResult.style.display = 'block';
    //     detailedResults.style.display = 'block';
    //     updateOverallResult(data);
    //     updateDetailedResults(data);
    // }

    function displayResults(data) {
    console.log('📊 Displaying results:', data);
    
    // ADD THESE DEBUG LINES:
    console.log('🔍 DEBUG - Checking individual results:');
    console.log('Pneumonia:', data.pneumonia);
    console.log('Tuberculosis:', data.tuberculosis);
    console.log('Cardiomegaly:', data.cardiomegaly);
    
    overallResult.style.display = 'block';
    detailedResults.style.display = 'block';
    updateOverallResult(data);
    updateDetailedResults(data);
}
    
    function updateOverallResult(data) {
        const pneumonia = data.pneumonia || {};
        const tuberculosis = data.tuberculosis || {};
        const cardiomegaly = data.cardiomegaly || {};
        
        const overallHeader = document.getElementById('overallHeader');
        const resultSummary = document.getElementById('resultSummary');
        
        const anyDiseaseDetected = (pneumonia.has_disease || tuberculosis.has_disease || cardiomegaly.has_disease);
        
        if (anyDiseaseDetected) {
            overallHeader.innerHTML = `
                <i class="fas fa-exclamation-triangle" style="color:#e74c3c"></i>
                <span>Analysis Complete - Abnormal Findings Detected</span>
            `;
        } else {
            overallHeader.innerHTML = `
                <i class="fas fa-check-circle" style="color:#2ecc71"></i>
                <span>Analysis Complete - No Critical Findings</span>
            `;
        }
        
        const pneumoniaConf = pneumonia.status === 'success' ? ('Confidence: ' + (pneumonia.confidence * 100).toFixed(1) + '%') : 'Error';
        const tbConf = tuberculosis.status === 'success' ? ('Confidence: ' + (tuberculosis.confidence * 100).toFixed(1) + '%') : 'Error';
        const cardioConf = cardiomegaly.status === 'success' ? ('Confidence: ' + (cardiomegaly.confidence * 100).toFixed(1) + '%') : (cardiomegaly.status || 'Error');
        
        const summaryHTML = `
            <div class="result-item ${pneumonia.has_disease ? 'positive' : 'negative'}">
                <div class="result-disease">Pneumonia</div>
                <div class="result-value">${pneumonia.result || 'N/A'}</div>
                <div>${pneumoniaConf}</div>
            </div>
            <div class="result-item ${tuberculosis.has_disease ? 'positive' : 'negative'}">
                <div class="result-disease">Tuberculosis</div>
                <div class="result-value">${tuberculosis.result || 'N/A'}</div>
                <div>${tbConf}</div>
            </div>
            <div class="result-item ${cardiomegaly.has_disease ? 'positive' : 'negative'}">
                <div class="result-disease">Cardiomegaly</div>
                <div class="result-value">${cardiomegaly.result || 'N/A'}</div>
                <div>${cardioConf}</div>
            </div>
        `;
        
        resultSummary.innerHTML = summaryHTML;
    }
    
    function updateDetailedResults(data) {
        const resultsGrid = document.getElementById('resultsGrid');
        const pneumonia = data.pneumonia || {};
        const tuberculosis = data.tuberculosis || {};
        const cardiomegaly = data.cardiomegaly || {};
        
        const detailedHTML = `
            <div class="disease-result">
                <div class="disease-result-header">
                    <div class="disease-name"><i class="fas fa-lungs"></i> Pneumonia</div>
                    <div class="confidence ${pneumonia.has_disease ? 'high' : 'low'}">
                        ${pneumonia.has_disease ? 'High Probability' : pneumonia.status === 'success' ? 'Low Probability' : 'Error'}
                    </div>
                </div>
                <div class="disease-description">
                    ${pneumonia.status === 'success' ? 
                      (pneumonia.has_disease ? 
                       'Signs of pneumonia detected. Please consult a healthcare professional.' : 
                       'No significant consolidation or infiltrates detected.') :
                      `Error: ${pneumonia.error || 'Model not loaded'}`}
                </div>
                <div class="indicators">
                    <span class="indicator">Status: ${pneumonia.status || 'unknown'}</span>
                    ${pneumonia.status === 'success' ? `<span class="indicator">Confidence: ${(pneumonia.confidence * 100).toFixed(1)}%</span>` : ''}
                </div>
            </div>

            <div class="disease-result">
                <div class="disease-result-header">
                    <div class="disease-name"><i class="fas fa-virus"></i> Tuberculosis</div>
                    <div class="confidence ${tuberculosis.has_disease ? 'high' : 'low'}">
                        ${tuberculosis.has_disease ? 'High Probability' : tuberculosis.status === 'success' ? 'Low Probability' : 'Error'}
                    </div>
                </div>
                <div class="disease-description">
                    ${tuberculosis.status === 'success' ? 
                      (tuberculosis.has_disease ? 
                       'Signs of tuberculosis detected. Please consult a healthcare professional.' : 
                       'No cavitary lesions, nodules, or fibrotic changes suggestive of TB detected.') :
                      `Error: ${tuberculosis.error || 'Model not loaded'}`}
                </div>
                <div class="indicators">
                    <span class="indicator">Status: ${tuberculosis.status || 'unknown'}</span>
                    ${tuberculosis.status === 'success' ? `<span class="indicator">Confidence: ${(tuberculosis.confidence * 100).toFixed(1)}%</span>` : ''}
                </div>
            </div>

            <div class="disease-result">
                <div class="disease-result-header">
                    <div class="disease-name"><i class="fas fa-heart"></i> Cardiomegaly</div>
                    <div class="confidence ${cardiomegaly.has_disease ? 'high' : 'low'}">
                        ${cardiomegaly.has_disease ? 'High Probability' : cardiomegaly.status === 'success' ? 'Low Probability' : cardiomegaly.status || 'Error'}
                    </div>
                </div>
                <div class="disease-description">
                    ${cardiomegaly.status === 'success' ? 
                      (cardiomegaly.has_disease ? 
                       'Cardiomegaly likely — consult a cardiologist / radiologist.' :
                       'No cardiomegaly detected.') :
                      `Status: ${cardiomegaly.status || 'Model not loaded'}`}
                </div>
                <div class="indicators">
                    <span class="indicator">Status: ${cardiomegaly.status || 'unknown'}</span>
                    ${cardiomegaly.status === 'success' ? `<span class="indicator">Confidence: ${(cardiomegaly.confidence * 100).toFixed(1)}%</span>` : ''}
                </div>
            </div>
        `;
        
        resultsGrid.innerHTML = detailedHTML;
    }
    
    function clearAll() {
        fileInput.value = '';
        currentFile = null;
        imagePreview.innerHTML = '';
        analyzeBtn.style.display = 'none';
        clearBtn.style.display = 'none';
        hideResults();
        hideLoading();
    }
    
    function hideResults() {
        overallResult.style.display = 'none';
        detailedResults.style.display = 'none';
    }
    
    function showLoading() {
        loading.style.display = 'block';
        analyzeBtn.style.display = 'none';
        clearBtn.style.display = 'none';
    }
    
    function hideLoading() {
        loading.style.display = 'none';
        clearBtn.style.display = 'block';
    }
    
    function showError(message) {
        overallResult.style.display = 'block';
        detailedResults.style.display = 'none';
        document.getElementById('overallHeader').innerHTML = `
            <i class="fas fa-exclamation-circle" style="color:#e74c3c"></i>
            <span>Analysis Failed</span>
        `;
        document.getElementById('resultSummary').innerHTML = `
            <div class="result-item positive">
                <div class="result-disease">Error</div>
                <div class="result-value" style="color:#e74c3c">Failed</div>
                <div>${message}</div>
            </div>
        `;
        hideLoading();
    }
    
    async function checkServerHealth() {
        try {
            const response = await fetch('http://localhost:5500/health');
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }
            const data = await response.json();
            console.log('✅ Server health:', data);
            // optional: show a small UI indicator if models are not loaded
            if (!data.cardiomegaly_model_loaded) {
                console.warn('⚠️ Cardiomegaly model not loaded on server.');
            }
        } catch (error) {
            console.error('❌ Cannot connect to server:', error.message);
        }
    }
});
