# Early Dementia Detection Framework

## An Integrated Technological Framework for Early Dementia Detection: Combining Communication Analysis, Security Protocols, and Machine Learning

![Dementia Detection](https://github.com/yourusername/dementia-detection/raw/main/assets/logo.png)

## 📋 Overview

This project implements a comprehensive framework for early dementia detection using multiple data sources and machine learning techniques. The system combines clinical assessment, voice analysis, and sleep pattern monitoring to provide a holistic approach to dementia risk assessment.

Our framework offers a hierarchical security system, interactive visualizations, and automated reporting capabilities, making it accessible to patients, caregivers, clinical staff, and administrators.

## 🔑 Key Features

- **Multi-modal Risk Assessment**: Integration of clinical data, voice biomarkers, and sleep patterns
- **Hierarchical Security Protocol**: Role-based access control with NFC authentication
- **Interactive Voice Analysis**: Analysis of speech patterns for cognitive impairment indicators
- **Sleep Pattern Monitoring**: Sleep quality assessment and correlation with dementia risk
- **Automated PDF Reporting**: Comprehensive risk assessment reports with recommendations
- **Interactive Visualizations**: Risk breakdowns, feature importance, and temporal projections
- **HIPAA Compliant**: Adherence to healthcare privacy regulations

## 🛠️ Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Custom Naive Bayes implementation
- **Audio Analysis**: Librosa
- **Visualization**: Matplotlib, Plotly
- **Report Generation**: ReportLab
- **Security**: Custom authentication system

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dementia-detection.git
cd dementia-detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```bash
# Run the Streamlit application
streamlit run app.py
```

## 📱 Access Levels

The system provides different access levels with varying functionality:

1. **Patient**: Basic assessment, view simplified results
2. **Caregiver**: Patient management, view detailed results, access recommendations
3. **Clinical Staff**: Advanced assessments, detailed analytics, medical recommendations
4. **Administrator**: System configuration, model performance monitoring, user management

## 🔍 Components

### Authentication System

The security system implements role-based access control with simulated NFC authentication. Each access level has specific permissions and features.

```python
# Example: Authentication
access_granted = check_security_access("Clinical Staff", "CS_CODE_123")
```

### Clinical Assessment

Collects patient information, including demographics, health conditions, lifestyle factors, and clinical measurements.

### Voice Analysis

Extracts speech features (e.g., speech rate, pause frequency, pitch variance) from audio recordings to identify potential cognitive impairment markers.

```python
# Example: Voice Analysis
speech_features = analyze_voice(audio_file)
```

### Sleep Pattern Analysis

Analyzes sleep data (e.g., sleep quality, duration, consistency) to identify patterns associated with dementia risk.

```python
# Example: Sleep Analysis
sleep_risk = analyze_sleep_patterns(sleep_data)
```

### Machine Learning Model

Implements a Naive Bayes classifier to predict dementia risk based on the integrated data sources.

```python
# Example: Risk Prediction
risk_score = Predict(X, means, var, prior, classes)
```

### PDF Report Generation

Generates comprehensive PDF reports with patient information, risk scores, and personalized recommendations.

## 📊 Screenshots

![Authentication](https://github.com/yourusername/dementia-detection/raw/main/assets/auth.png)
*Authentication Screen*

![Voice Analysis](https://github.com/yourusername/dementia-detection/raw/main/assets/voice.png)
*Voice Analysis Dashboard*

![Risk Assessment](https://github.com/yourusername/dementia-detection/raw/main/assets/risk.png)
*Integrated Risk Assessment*

## 📝 Example Report

The system generates detailed PDF reports including:

- Patient demographics
- Risk assessment results
- Key risk factors
- Clinical recommendations
- Lifestyle recommendations
- Visualizations of risk components

## 🔬 Research Foundation

This project is based on research showing that early detection of dementia is possible through the analysis of:

- Speech patterns and language changes
- Sleep disturbances
- Clinical biomarkers
- Lifestyle and health factors

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

```bash
# Development workflow
git checkout -b feature/your-feature-name
# Make your changes
git commit -m "Add your feature"
git push origin feature/your-feature-name
# Submit a pull request
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Dinesh** - *Lead Developer*
- **Divyadharshini** - *Machine Learning Engineer*
- **Srinidhi** - *Machine Learning Engineer*

---

**Disclaimer**: This application is intended as a screening tool and does not replace professional medical diagnosis. Always consult healthcare professionals for medical advice.