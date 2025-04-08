# Early Dementia Detection Framework

## An Integrated Technological Framework for Early Dementia Detection: Combining Communication Analysis, Security Protocols, and Machine Learning

## 📋 Overview

This project implements a comprehensive framework for early dementia detection using multiple data sources and machine learning techniques. The system combines clinical assessment, voice analysis, and sleep pattern monitoring to provide a holistic approach to dementia risk assessment.

Our framework offers a hierarchical security system, interactive visualizations, and automated reporting capabilities, making it accessible to patients, caregivers, clinical staff, and administrators.

## 🔍 Research Background

Dementia represents a significant global health challenge affecting over 50 million individuals worldwide, with projections suggesting an increase to 152 million by 2050. Early detection is crucial for timely intervention and effective management strategies.

Traditional diagnostic methods (neuropsychological assessments, neuroimaging) are often time-consuming, expensive, and require specialized expertise. Our research explores a novel approach by analyzing communication patterns, sleep data, and implementing advanced security protocols.

## 🔑 Key Features

- **Multi-modal Risk Assessment**: Integration of clinical data, voice biomarkers, and sleep patterns
- **Hierarchical Security Protocol**: Role-based access control with NFC authentication
- **Interactive Voice Analysis**: Analysis of speech patterns for cognitive impairment indicators
- **Sleep Pattern Monitoring**: Sleep quality assessment and correlation with dementia risk
- **Automated PDF Reporting**: Comprehensive risk assessment reports with recommendations
- **Interactive Visualizations**: Risk breakdowns, feature importance, and temporal projections
- **HIPAA Compliant**: Adherence to healthcare privacy regulations

## 💻 Technologies Used

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Custom Naive Bayes implementation
- **Audio Analysis**: Librosa
- **Visualization**: Matplotlib, Plotly
- **Report Generation**: ReportLab
- **Security**: Custom authentication system

## 📊 Components

### Voice Analysis System
- Extracts speech features like speech rate, pause frequency, pitch variance
- Identifies word-finding difficulty, articulation precision, and voice tremor
- Generates spectrograms for visual analysis of vocal biomarkers

### Sleep Pattern Analysis
- Monitors sleep duration, quality, and consistency
- Tracks REM sleep percentage and sleep disruptions
- Correlates sleep metrics with cognitive health indicators

### Machine Learning Models
- Implements Naive Bayes classifier for dementia risk prediction
- Utilizes Random Forest for improved classification accuracy
- Compares model performance using accuracy, precision, recall, F1-score, and AUC

### Security System
- Role-based access control
- NFC-based authentication
- Hierarchical data protection

## 📊 Results

Our framework demonstrates significant accuracy in early dementia detection by combining multiple data modalities:

- **Voice analysis**: Identifies subtle changes in speech patterns
- **Sleep monitoring**: Detects disruptions associated with cognitive decline
- **Clinical assessment**: Integrates with traditional evaluation methods

The system provides an overall dementia risk score with component breakdowns, highlighting the most significant contributing factors for each individual.

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

## 📄 Future Work

- **Multimodal Integration**: Incorporating facial expression analysis, gait monitoring, and digital biomarkers
- **Federated Learning**: Maintaining data privacy while learning from diverse populations
- **Explainable AI**: Providing transparent reasoning behind classifications
- **Longitudinal Studies**: Tracking subtle changes over extended periods
- **Personalized Interventions**: Automatically generating recommendations based on detected patterns
- **Cross-cultural Validation**: Ensuring applicability across diverse linguistic and cultural contexts
- **Edge Computing Solutions**: Reducing latency and network dependencies

## 👥 Team

- **Dinesh S** - M.Tech student in Artificial Intelligence and Machine Learning at Vellore Institute of Technology
- **Divyadharshini J** - M.Tech student in Artificial Intelligence and Machine Learning at Vellore Institute of Technology
- **Srinidhi S** - M.Tech student in Artificial Intelligence and Machine Learning at Vellore Institute of Technology

## 📝 Citation

If you use this framework in your research, please cite our paper:
```
S. Dinesh, J. Divyadharshini, and S. Srinidhi "An Integrated Technological Framework for Early Dementia Detection: Combining Communication Analysis, Security Protocols, and Machine Learning," Vellore Institute of Technology, 2025.
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This application is intended as a screening tool and does not replace professional medical diagnosis. Always consult healthcare professionals for medical advice.
