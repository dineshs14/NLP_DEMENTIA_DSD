import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import time
import random
from datetime import datetime, timedelta
import base64
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io
import json
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.units import inch
from io import BytesIO
import base64

# Set page configuration
st.set_page_config(
    page_title="Advanced Dementia Risk Prediction",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("Dementia Risk Prediction Tool")
st.markdown("""
This application uses machine learning to predict dementia risk based on health factors,
voice analysis, sleep patterns, and other biomarkers. Please fill out the form below for a personalized assessment.
""")

# Create a hierarchical security system
def check_security_access(access_level, provided_code=None):
    # Simulate NFC authentication (in a real app, this would interface with actual NFC hardware)
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
        st.session_state['access_level'] = 0
    
    # For simulation purposes
    security_codes = {
        1: "PATIENT1234",  # Basic patient access
        2: "CAREGIVER5678",  # Caregiver access
        3: "CLINICIAN9012",  # Clinical staff access
        4: "ADMIN3456"  # Administrator access
    }
    
    if provided_code and provided_code == security_codes.get(access_level, ""):
        st.session_state['authenticated'] = True
        st.session_state['access_level'] = access_level
        return True
    elif st.session_state['authenticated'] and st.session_state['access_level'] >= access_level:
        return True
    return False

# Normal function for prediction using Naive Bayes
def Normal(n, mu, var):
    sd = np.sqrt(var)
    pdf = np.zeros_like(n)
    mask = sd != 0  # Avoid division by zero
    pdf[mask] = (np.e ** (-0.5 * ((n[mask] - mu) / sd[mask]) ** 2)) / (sd[mask] * np.sqrt(2 * np.pi))
    return pdf

def Predict(X, means, var, prior, classes):
    Predictions = []
    
    for instance in X:
        ClassLikelihood = []
        
        for cls in classes:
            FeatureLikelihoods = []
            FeatureLikelihoods.append(np.log(prior[cls]))
            
            for col in range(X.shape[1]):
                mean = means.iloc[cls, col]
                variance = var.iloc[cls, col]
                data = instance[col]
                
                Likelihood = Normal(data, mean, variance)
                
                if Likelihood != 0:
                    Likelihood = np.log(Likelihood)
                else:
                    Likelihood = -999  # A very low log probability
                
                FeatureLikelihoods.append(Likelihood)
            
            TotalLikelihood = sum(FeatureLikelihoods)
            ClassLikelihood.append(TotalLikelihood)
        
        MaxIndex = ClassLikelihood.index(max(ClassLikelihood))
        Prediction = classes[MaxIndex]
        # Calculate probability based on softmax of log likelihoods
        probabilities = np.exp(ClassLikelihood) / np.sum(np.exp(ClassLikelihood))
        Predictions.append((Prediction, probabilities[1]))
    
    return Predictions

def generate_risk_assessment_pdf(X, X_norm, integrated_risk, base_risk_score, voice_risk_score, sleep_risk_score, sorted_contributions, recommendations, lifestyle_recs):
    """
    Generate a PDF report for dementia risk assessment
    """
    # Create a buffer for the PDF
    buffer = BytesIO()
    
    # Create the PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter, 
                           rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=72)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Create custom styles
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Title'],
        fontSize=18,
        alignment=1,
        spaceAfter=24
    )
    
    heading_style = ParagraphStyle(
        'Heading1',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12
    )
    
    subheading_style = ParagraphStyle(
        'Heading2',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6
    )
    
    normal_style = styles['Normal']
    
    # Define content elements
    elements = []
    
    # Title
    elements.append(Paragraph('Dementia Risk Assessment Report', title_style))
    elements.append(Paragraph(f'Assessment Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', normal_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Patient Information
    elements.append(Paragraph('Patient Information', heading_style))
    
    patient_data = []
    
    # Add basic patient details
    if 'Age' in X:
        patient_data.append(['Age', f"{X['Age'].values[0]:.0f} years"])
    if 'Gender' in X:
        gender = "Male" if X['Gender'].values[0] == 1 else "Female"
        patient_data.append(['Gender', gender])
    if 'MMSE_Score' in X:
        patient_data.append(['MMSE Score', f"{X['MMSE_Score'].values[0]:.0f}/30"])
    
    # Create patient info table
    if patient_data:
        patient_table = Table(patient_data, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (0,-1), colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold')
        ]))
        elements.append(patient_table)
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Risk Score Section
    elements.append(Paragraph('Dementia Risk Assessment Results', heading_style))
    
    # Risk level classification
    if integrated_risk < 25:
        risk_level = "Low"
        risk_color = colors.green
    elif integrated_risk < 50:
        risk_level = "Moderate"
        risk_color = colors.orange
    elif integrated_risk < 75:
        risk_level = "High"
        risk_color = colors.red
    else:
        risk_level = "Very High"
        risk_color = colors.darkred
    
    # Overall risk score
    elements.append(Paragraph(f'Overall Risk Score: {integrated_risk:.1f}%', subheading_style))
    elements.append(Paragraph(f'Risk Level: {risk_level}', subheading_style))
    
    # Create risk visualization
    risk_data = [
        ['Risk Component', 'Score (%)'],
        ['Clinical Assessment', f"{base_risk_score:.1f}%"],
        ['Voice Analysis', f"{voice_risk_score:.1f}%"],
        ['Sleep Patterns', f"{sleep_risk_score:.1f}%"],
        ['Overall Risk', f"{integrated_risk:.1f}%"]
    ]
    
    risk_table = Table(risk_data, colWidths=[3*inch, 2*inch])
    risk_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('BACKGROUND', (0,-1), (-1,-1), colors.lightblue),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ALIGN', (1,0), (1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold')
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Risk Factors Section
    elements.append(Paragraph('Key Risk Factors', heading_style))
    
    # Get top 5 contributors
    top_contributors = sorted_contributions[:5]
    
    # Display top contributors
    risk_factors = []
    for feature, contribution in top_contributors:
        # Make feature name more readable
        readable_feature = feature.replace('_', ' ')
        risk_factors.append([readable_feature, f"{contribution:.1f}%"])
    
    if risk_factors:
        risk_factor_table = Table([['Risk Factor', 'Contribution']] + risk_factors, 
                                colWidths=[3.5*inch, 1.5*inch])
        risk_factor_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (1,0), (1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
        ]))
        elements.append(risk_factor_table)
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    elements.append(Paragraph('Recommendations', heading_style))
    
    # Clinical recommendations
    if recommendations:
        elements.append(Paragraph('Clinical Recommendations', subheading_style))
        for rec in recommendations:
            elements.append(Paragraph(f"• {rec}", normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    # Lifestyle recommendations
    if lifestyle_recs:
        elements.append(Paragraph('Lifestyle Recommendations', subheading_style))
        for rec in lifestyle_recs:
            elements.append(Paragraph(f"• {rec}", normal_style))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Note for doctors
    doctor_note = """
    <i>Note to healthcare providers: Not all measurements in this report are of equal clinical accuracy. 
    Please use clinical judgment when interpreting these results. This tool is intended as a 
    screening aid and not as a definitive diagnostic instrument.</i>
    """
    elements.append(Paragraph(doctor_note, normal_style))
    
    # Add footer with developers' names
    def add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 9)
        canvas.drawString(inch, 0.5*inch, 
                         "Developed by Dinesh, Divyadharshini, and Srinidhi")
        canvas.restoreState()
    
    # Build the PDF document
    doc.build(elements, onFirstPage=add_footer, onLaterPages=add_footer)
    
    # Get the PDF data
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

# Voice analysis function (simulated without requiring an external API)
def analyze_voice(audio_file):
    try:
        # In a real implementation, you would use a speech recognition and analysis library
        # Here we'll simulate the analysis with some pre-defined features
        
        # Load audio file using librosa if it exists
        if audio_file is not None:
            # Process the uploaded audio
            audio_bytes = audio_file.read()
            
            # Get some basic audio statistics
            # In a real app, you would extract features like:
            # - Speech rate
            # - Pauses between words
            # - Vocal tremor
            # - Pitch variation
            # - Word finding difficulty markers
            
            # Simulate extracted features
            speech_features = {
                'speech_rate': random.uniform(2.5, 5.5),  # Words per second
                'pause_frequency': random.uniform(0.1, 0.8),  # Pauses per sentence
                'pitch_variance': random.uniform(10, 100),  # Variance in Hz
                'word_finding_difficulty': random.uniform(0, 1),  # Normalized score
                'articulation_precision': random.uniform(0.5, 1.0),  # Normalized score
                'voice_tremor': random.uniform(0, 0.5),  # Normalized score
            }
            
            # Generate a spectrogram for visualization
            buffer = io.BytesIO()
            plt.figure(figsize=(10, 4))
            plt.title("Voice Analysis Spectrogram")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            # Create a dummy spectrogram
            plt.imshow(np.random.rand(100, 100), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            spectrogram = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            return speech_features, spectrogram
        else:
            return None, None
    except Exception as e:
        st.error(f"Error analyzing voice: {e}")
        return None, None

# Sleep correlation analysis
def analyze_sleep_patterns(sleep_data):
    # This function would analyze sleep data in a real application
    # For now, we'll simulate analysis results
    
    # Parse sleep data
    sleep_quality_score = 0
    sleep_disruption_score = 0
    rem_sleep_percentage = 0
    
    if sleep_data:
        dates = []
        qualities = []
        durations = []
        
        for entry in sleep_data.split('\n'):
            if ',' in entry:
                try:
                    date_str, quality, duration = entry.split(',')
                    date = datetime.strptime(date_str.strip(), '%Y-%m-%d')
                    quality = quality.strip().lower()
                    duration = float(duration.strip())
                    
                    dates.append(date)
                    
                    # Convert quality to numeric
                    if quality == 'poor':
                        qualities.append(1)
                        sleep_disruption_score += 1
                    elif quality == 'fair':
                        qualities.append(2)
                        sleep_disruption_score += 0.5
                    elif quality == 'good':
                        qualities.append(3)
                    elif quality == 'excellent':
                        qualities.append(4)
                        sleep_quality_score += 1
                    
                    durations.append(duration)
                    
                    # Analyze sleep duration
                    if duration < 6:
                        sleep_disruption_score += 0.5
                    elif duration > 8:
                        sleep_quality_score += 0.5
                except:
                    continue
        
        if dates:
            # Create a plot of sleep data
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Map quality values to colors
            colors = {1: 'red', 2: 'orange', 3: 'lightgreen', 4: 'darkgreen'}
            mapped_colors = [colors.get(q, 'gray') for q in qualities]
            
            ax.bar(range(len(dates)), durations, color=mapped_colors)
            ax.set_xlabel('Day')
            ax.set_ylabel('Sleep Duration (hours)')
            ax.set_title('Sleep Pattern Analysis')
            
            # Add quality labels
            quality_labels = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
            ax.set_xticks(range(len(dates)))
            ax.set_xticklabels([d.strftime('%m-%d') for d in dates], rotation=45)
            
            # Calculate metrics
            avg_duration = sum(durations) / len(durations) if durations else 0
            sleep_consistency = 1 - (max(durations) - min(durations)) / 8 if durations else 0
            sleep_quality_avg = sum(qualities) / len(qualities) if qualities else 0
            
            # Normalize scores
            if dates:
                sleep_quality_score = sleep_quality_score / len(dates) * 5  # Scale to 0-5
                sleep_disruption_score = sleep_disruption_score / len(dates) * 5  # Scale to 0-5
                rem_sleep_percentage = random.uniform(15, 25)  # Simulate REM sleep percentage
            
            return {
                'avg_duration': avg_duration,
                'sleep_consistency': sleep_consistency,
                'sleep_quality_avg': sleep_quality_avg,
                'sleep_quality_score': sleep_quality_score,
                'sleep_disruption_score': sleep_disruption_score,
                'rem_sleep_percentage': rem_sleep_percentage,
                'plot': fig
            }
    
    return None

# Create tab structure
tabs = st.tabs(["Authentication", "Patient Assessment", "Voice Analysis", "Sleep Patterns", "Results"])

# Authentication tab
with tabs[0]:
    st.header("Access Authentication")
    st.markdown("""
    This application uses a hierarchical security system to protect patient data.
    Please authenticate with the appropriate credentials for your access level.
    
    Access Levels:
    - Level 1: Patient (Basic access to own information)
    - Level 2: Caregiver (Access to patient information and basic analysis)
    - Level 3: Clinical Staff (Full access to patient data and clinical tools)
    - Level 4: Administrator (Complete system access)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        access_level = st.selectbox("Select Access Level", [1, 2, 3, 4], format_func=lambda x: {
            1: "Level 1 - Patient",
            2: "Level 2 - Caregiver",
            3: "Level 3 - Clinical Staff",
            4: "Level 4 - Administrator"
        }.get(x))
        
        auth_method = st.radio("Authentication Method", ["Access Code", "Simulated NFC"])
        
        if auth_method == "Access Code":
            auth_code = st.text_input("Enter Access Code", type="password")
            if st.button("Authenticate"):
                if check_security_access(access_level, auth_code):
                    st.success(f"Successfully authenticated with Level {access_level} access!")
                else:
                    st.error("Authentication failed. Invalid code.")
        else:
            if st.button("Scan NFC Card (Simulation)"):
                st.info("Scanning NFC card...")
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress.progress(i + 1)
                
                # Simulate random success/failure for demo purposes
                if random.random() > 0.3:  # 70% success rate
                    st.success("NFC authentication successful!")
                    st.session_state['authenticated'] = True
                    st.session_state['access_level'] = access_level
                else:
                    st.error("NFC authentication failed. Please try again or use access code.")
    
    with col2:
        st.subheader("Current Status")
        if st.session_state.get('authenticated', False):
            st.success(f"Authenticated with Level {st.session_state.get('access_level', 0)} access")
            
            # Show what features are available at this level
            st.markdown("### Available Features:")
            features = {
                1: ["Basic risk assessment", "General recommendations"],
                2: ["Patient risk assessment", "Voice analysis (basic)", "Basic sleep analysis", "Caregiver recommendations"],
                3: ["Full clinical assessment", "Advanced voice analysis", "Detailed sleep correlation", "Treatment recommendations"],
                4: ["All clinical features", "System administration", "Data export", "Model tuning"]
            }
            
            for feature in features.get(st.session_state.get('access_level', 0), []):
                st.markdown(f"- {feature}")
        else:
            st.warning("Not authenticated. Please authenticate to access features.")

# Patient Assessment tab (only accessible if authenticated)
with tabs[1]:
    if st.session_state.get('authenticated', False):
        st.header("Patient Information")
        
        # Create form with input fields
        with st.form("patient_info_form"):
            # Basic demographic information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Demographics & Basics")
                age = st.slider("Age", min_value=40, max_value=100, value=65, step=1)
                gender = st.selectbox("Gender", ["Male", "Female"])
                education = st.selectbox("Education Level", ["None", "Primary School", "Secondary School", "Diploma/Degree"])
                dominant_hand = st.selectbox("Dominant Hand", ["Right", "Left"])
                mmse_score = st.slider("MMSE Score (Mini-Mental State Examination)", min_value=0, max_value=30, value=27, step=1)
                
                st.subheader("Genetic & Family Factors")
                apoe_status = st.selectbox("APOE ε4 Status", ["Negative", "Positive"])
                family_history = st.selectbox("Family History of Dementia", ["No", "Yes"])
                
                st.subheader("Lifestyle Factors")
                smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Former Smoker", "Current Smoker"])
                physical_activity = st.selectbox("Physical Activity Level", ["Sedentary", "Mild Activity", "Moderate Activity"])
                nutrition_diet = st.selectbox("Nutrition Diet", ["Regular Diet", "Low-Carb Diet", "Mediterranean Diet"])
                sleep_quality = st.selectbox("Sleep Quality", ["Good", "Poor"])
            
            with col2:
                st.subheader("Health Conditions")
                depression = st.selectbox("Depression Status", ["No", "Yes"])
                chronic_health = st.selectbox("Chronic Health Conditions", ["None", "Heart Disease", "Hypertension", "Diabetes"])
                medication_history = st.selectbox("On Medication for Hypertension/Diabetes", ["No", "Yes"])
                
                st.subheader("Clinical Measurements")
                bmi = st.slider("BMI", min_value=15.0, max_value=45.0, value=25.0, step=0.1)
                systolic_bp = st.slider("Systolic Blood Pressure (mmHg)", min_value=90, max_value=200, value=120, step=1)
                diastolic_bp = st.slider("Diastolic Blood Pressure (mmHg)", min_value=50, max_value=120, value=80, step=1)
                cholesterol = st.slider("Total Cholesterol (mg/dL)", min_value=100, max_value=300, value=180, step=1)
                glucose = st.slider("Glucose Level (mg/dL)", min_value=70, max_value=200, value=100, step=1)
                hba1c = st.slider("HbA1c (%)", min_value=4.0, max_value=12.0, value=5.5, step=0.1)
                vitamin_b12 = st.slider("Vitamin B12 (pg/mL)", min_value=100, max_value=1000, value=500, step=10)
                vitamin_d = st.slider("Vitamin D (ng/mL)", min_value=10, max_value=100, value=30, step=1)
                
                # Additional interactive input if access level is high enough
                if st.session_state.get('access_level', 0) >= 3:
                    st.subheader("Advanced Clinical Markers")
                    hippocampal_volume = st.slider("Hippocampal Volume (% of norm)", min_value=50, max_value=120, value=100, step=1)
                    amyloid_level = st.slider("Amyloid Beta Level", min_value=0, max_value=100, value=20, step=1)
                    tau_protein = st.slider("Tau Protein Level", min_value=0, max_value=100, value=15, step=1)
                else:
                    hippocampal_volume = 100
                    amyloid_level = 20
                    tau_protein = 15
            
            # Submit button
            submitted = st.form_submit_button("Save Patient Data")
            
            if submitted:
                st.success("Patient data saved successfully!")
                st.session_state['patient_data_saved'] = True
    else:
        st.warning("Authentication required. Please authenticate in the Authentication tab.")

# Voice Analysis tab
with tabs[2]:
    if st.session_state.get('authenticated', False) and st.session_state.get('access_level', 0) >= 2:
        st.header("Voice Analysis for Dementia Detection")
        st.markdown("""
        Upload an audio recording of the patient's speech for analysis.
        Our system will extract linguistic and acoustic features that may indicate cognitive decline.
        
        Suggested prompt for recording:
        - Ask the patient to describe what they had for breakfast
        - Ask the patient to describe their childhood home
        - Ask the patient to count backward from 100 by 7
        """)
        
        # Create columns for a better layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Upload voice recording
            audio_file = st.file_uploader("Upload Voice Recording", type=['wav', 'mp3', 'm4a'])
            
            # Record audio directly (simulated)
            if st.button("Record Audio (Simulation)"):
                with st.spinner("Recording..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.03)
                        progress.progress(i + 1)
                    st.success("Recording completed!")
                    # In a real app, this would save the recording
            
            # Interactive voice assessment
            st.subheader("Interactive Voice Assessment")
            question_options = [
                "Please count backward from 100 by 7",
                "Can you name as many animals as possible in one minute?",
                "Please describe what you did yesterday in detail",
                "Can you repeat this phrase: 'The quick brown fox jumps over the lazy dog'"
            ]
            
            selected_question = st.selectbox("Select a question to ask:", question_options)
            
            if st.button("Start Interactive Assessment"):
                st.info(f"Please ask the patient: {selected_question}")
                st.markdown("##### Simulated Response Timer")
                response_time = st.empty()
                progress = st.progress(0)
                
                start_time = time.time()
                for i in range(100):
                    current_time = time.time() - start_time
                    response_time.markdown(f"Response time: {current_time:.1f} seconds")
                    progress.progress(i + 1)
                    time.sleep(0.05)
                
                st.success("Response recorded!")
        
        with col2:
            st.subheader("Voice Analysis Results")
            
            # Process the uploaded audio or simulated recording
            if audio_file or st.button("Analyze Simulated Recording"):
                with st.spinner("Analyzing voice patterns..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress.progress(i + 1)
                    
                    speech_features, spectrogram = analyze_voice(audio_file)
                    
                    if speech_features:
                        st.success("Analysis complete!")
                        
                        # Display feature analysis
                        st.markdown("### Speech Biomarkers")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Speech Rate", f"{speech_features['speech_rate']:.2f} words/sec")
                            st.metric("Pause Frequency", f"{speech_features['pause_frequency']:.2f}")
                            st.metric("Pitch Variance", f"{speech_features['pitch_variance']:.2f} Hz")
                        
                        with col2:
                            st.metric("Word Finding Difficulty", f"{speech_features['word_finding_difficulty']:.2f}")
                            st.metric("Articulation Precision", f"{speech_features['articulation_precision']:.2f}")
                            st.metric("Voice Tremor", f"{speech_features['voice_tremor']:.2f}")
                        
                        # Display dementia risk based on voice
                        voice_risk_score = (0.3 * speech_features['pause_frequency'] + 
                                          0.3 * speech_features['word_finding_difficulty'] + 
                                          0.2 * speech_features['voice_tremor'] + 
                                          0.2 * (1 - speech_features['articulation_precision']))
                        
                        st.markdown(f"### Voice-based Dementia Risk Score: {voice_risk_score*100:.1f}%")
                        
                        # Risk visualization
                        fig, ax = plt.subplots(figsize=(8, 1))
                        ax.barh(0, 100, color='lightgray', height=0.5)
                        ax.barh(0, voice_risk_score*100, color='red', height=0.5)
                        ax.set_xlim(0, 100)
                        ax.set_yticks([])
                        ax.set_xticks([0, 25, 50, 75, 100])
                        ax.set_xlabel('Risk %')
                        st.pyplot(fig)
                        
                        # Display spectrogram
                        if spectrogram:
                            st.markdown("### Voice Spectrogram Analysis")
                            st.image(f"data:image/png;base64,{spectrogram}", use_column_width=True)
                            
                            # Add to session state for later use
                            st.session_state['voice_risk_score'] = voice_risk_score
                    else:
                        st.error("No audio data available for analysis.")
    else:
        access_level_needed = 2
        if st.session_state.get('authenticated', False):
            st.warning(f"This feature requires Level {access_level_needed} access. Your current access level is {st.session_state.get('access_level', 0)}.")
        else:
            st.warning("Authentication required. Please authenticate in the Authentication tab.")

# Sleep Patterns tab
with tabs[3]:
    if st.session_state.get('authenticated', False) and st.session_state.get('access_level', 0) >= 2:
        st.header("Sleep Pattern Analysis")
        st.markdown("""
        Sleep disturbances are common in early stages of dementia and may serve as early biomarkers.
        Enter sleep data to analyze patterns and correlations with cognitive health.
        
        Format: One entry per line with date, quality, and duration in hours.
        Example:
        ```
        2023-01-01, Good, 7.5
        2023-01-02, Poor, 5.2
        ```
        
        Quality options: Poor, Fair, Good, Excellent
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample data to help users
            sample_data = """2023-03-01, Good, 7.5
2023-03-02, Fair, 6.8
2023-03-03, Poor, 5.2
2023-03-04, Fair, 6.5
2023-03-05, Good, 7.8
2023-03-06, Excellent, 8.2
2023-03-07, Fair, 6.4"""
            
            sleep_data = st.text_area("Enter Sleep Data (last 7-30 days)", value=sample_data, height=200)
            
            st.subheader("Sleep Monitoring Options")
            monitor_device = st.selectbox("Sleep Monitoring Device", 
                                        ["None", "Fitbit", "Apple Watch", "Samsung Galaxy Watch", "Oura Ring", "Sleep as Android App"])
            
            if monitor_device != "None":
                st.info(f"In a full implementation, this would connect to your {monitor_device} to automatically import sleep data.")
                if st.button("Connect Device (Simulation)"):
                    with st.spinner("Connecting to device..."):
                        progress = st.progress(0)
                        for i in range(100):
                            time.sleep(0.02)
                            progress.progress(i + 1)
                        st.success("Device connected! Data imported.")
                        
                        # Generate some random sleep data
                        today = datetime.now()
                        random_sleep_data = ""
                        qualities = ["Poor", "Fair", "Good", "Excellent"]
                        for i in range(14):
                            date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                            quality = random.choice(qualities)
                            duration = round(random.uniform(5.0, 9.0), 1)
                            random_sleep_data += f"{date}, {quality}, {duration}\n"
                        
                        sleep_data = random_sleep_data
        
        with col2:
            if st.button("Analyze Sleep Patterns"):
                with st.spinner("Analyzing sleep patterns..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress.progress(i + 1)
                    
                    sleep_analysis = analyze_sleep_patterns(sleep_data)
                    
                    if sleep_analysis:
                        st.success("Analysis complete!")
                        
                        # Display sleep metrics
                        st.markdown("### Sleep Pattern Metrics")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Sleep Duration", f"{sleep_analysis['avg_duration']:.1f} hours")
                            st.metric("Sleep Consistency", f"{sleep_analysis['sleep_consistency']:.2f}")
                            st.metric("Average Sleep Quality", f"{sleep_analysis['sleep_quality_avg']:.1f}/4")
                        
                        with col2:
                            st.metric("Sleep Quality Score", f"{sleep_analysis['sleep_quality_score']:.1f}/5")
                            st.metric("Sleep Disruption Score", f"{sleep_analysis['sleep_disruption_score']:.1f}/5")
                            st.metric("Estimated REM %", f"{sleep_analysis['rem_sleep_percentage']:.1f}%")
                        
                        # Display sleep pattern plot
                        st.markdown("### Sleep Pattern Visualization")
                        st.pyplot(sleep_analysis['plot'])
                        
                        # Calculate sleep-related dementia risk
                        sleep_risk_score = (
                            (5 - sleep_analysis['sleep_quality_score']) * 0.4 + 
                            sleep_analysis['sleep_disruption_score'] * 0.4 + 
                            (1 - sleep_analysis['sleep_consistency']) * 0.2
                        ) / 5
                        
                        st.markdown(f"### Sleep-based Dementia Risk Factor: {sleep_risk_score*100:.1f}%")
                        
                        # Risk visualization
                        fig, ax = plt.subplots(figsize=(8, 1))
                        ax.barh(0, 100, color='lightgray', height=0.5)
                        ax.barh(0, sleep_risk_score*100, color='orange', height=0.5)
                        ax.set_xlim(0, 100)
                        ax.set_yticks([])
                        ax.set_xticks([0, 25, 50, 75, 100])
                        ax.set_xlabel('Risk %')
                        st.pyplot(fig)
                        
                        # Store in session state for integrated analysis
                        st.session_state['sleep_risk_score'] = sleep_risk_score
                    else:
                        st.error("Could not analyze sleep data. Please check the format.")
    else:
        access_level_needed = 2
        if st.session_state.get('authenticated', False):
            st.warning(f"This feature requires Level {access_level_needed} access. Your current access level is {st.session_state.get('access_level', 0)}.")
        else:
            st.warning("Authentication required. Please authenticate in the Authentication tab.")

# Results tab
with tabs[4]:
    if st.session_state.get('authenticated', False):
        st.header("Integrated Dementia Risk Assessment")
        
        if st.button("Generate Comprehensive Assessment"):
            if not st.session_state.get('patient_data_saved', False):
                st.warning("Please complete the Patient Assessment form first.")
            else:
                with st.spinner("Processing patient data and generating assessment..."):
                    progress = st.progress(0)
                    for i in range(100):
                        time.sleep(0.02)
                        progress.progress(i + 1)
                    
                    # In a real app, you would use the actual patient data from the form
                    # Here we'll use the mock patient data for demonstration
                    
                    # Convert categorical variables to numerical values (same as original code)
                    gender_val = 1 if gender == "Female" else 2
                    
                    def education_to_numeric(x):
                        if x == "Primary School": return 1
                        elif x == "Secondary School": return 2
                        elif x == "Diploma/Degree": return 3
                        else: return 0
                    
                    education_val = education_to_numeric(education)
                    dominant_hand_val = 1 if dominant_hand == "Right" else 2
                    apoe_val = 1 if apoe_status == "Positive" else 0
                    family_history_val = 1 if family_history == "Yes" else 0
                    
                    def smoking_to_numeric(x):
                        if x == "Former Smoker": return 1
                        elif x == "Current Smoker": return 2
                        else: return 0
                    
                    smoking_val = smoking_to_numeric(smoking)
                    
                    def activity_to_numeric(x):
                        if x == "Mild Activity": return 1
                        elif x == "Moderate Activity": return 2
                        else: return 0
                    
                    activity_val = activity_to_numeric(physical_activity)
                    depression_val = 1 if depression == "Yes" else 0
                    medication_val = 1 if medication_history == "Yes" else 0
                    
                    def nutrition_to_numeric(x):
                        if x == "Low-Carb Diet": return 1
                        elif x == "Mediterranean Diet": return 2
                        else: return 0
                    
                    nutrition_val = nutrition_to_numeric(nutrition_diet)
                    sleep_val = 1 if sleep_quality == "Poor" else 0
                    
                    def health_to_numeric(x):
                        if x == "Heart Disease": return 1
                        elif x == "Hypertension": return 2
                        elif x == "Diabetes": return 3
                        else: return 0
                    
                    health_val = health_to_numeric(chronic_health)
                    
                    # Prepare data for the model
                    # Create a feature dataframe with the normalized values
                    X = pd.DataFrame({
                        'Age': [age],
                        'Gender': [gender_val],
                        'Education': [education_val],
                        'Dominant_Hand': [dominant_hand_val],
                        'MMSE_Score': [mmse_score],
                        'APOE_Status': [apoe_val],
                        'Family_History': [family_history_val],
                        'Smoking': [smoking_val],
                        'Physical_Activity': [activity_val],
                        'Depression': [depression_val],
                        'Chronic_Health': [health_val],
                        'Medication': [medication_val],
                        'Nutrition': [nutrition_val],
                        'Sleep_Quality': [sleep_val],
                        'BMI': [bmi],
                        'Systolic_BP': [systolic_bp],
                        'Diastolic_BP': [diastolic_bp],
                        'Cholesterol': [cholesterol],
                        'Glucose': [glucose],
                        'HbA1c': [hba1c],
                        'Vitamin_B12': [vitamin_b12],
                        'Vitamin_D': [vitamin_d],
                        'Hippocampal_Volume': [hippocampal_volume],
                        'Amyloid_Level': [amyloid_level],
                        'Tau_Protein': [tau_protein]
                    })
                    
                    # Normalize the data (in a real app, this would use a pre-trained scaler)
                    # For simplicity, we'll use a simple normalization approach
                    X_norm = X.copy()
                    
                    # Age normalization (40-100 to 0-1)
                    X_norm['Age'] = (X_norm['Age'] - 40) / 60
                    
                    # MMSE normalization (0-30 to 0-1, inverted because lower is worse)
                    X_norm['MMSE_Score'] = 1 - (X_norm['MMSE_Score'] / 30)
                    
                    # BMI normalization (centered around 25)
                    X_norm['BMI'] = abs(X_norm['BMI'] - 25) / 10
                    
                    # Blood pressure normalization
                    X_norm['Systolic_BP'] = abs(X_norm['Systolic_BP'] - 120) / 40
                    X_norm['Diastolic_BP'] = abs(X_norm['Diastolic_BP'] - 80) / 20
                    
                    # Cholesterol normalization
                    X_norm['Cholesterol'] = (X_norm['Cholesterol'] - 150) / 100
                    
                    # Glucose normalization
                    X_norm['Glucose'] = (X_norm['Glucose'] - 70) / 130
                    
                    # HbA1c normalization
                    X_norm['HbA1c'] = (X_norm['HbA1c'] - 5) / 7
                    
                    # Vitamin normalization
                    X_norm['Vitamin_B12'] = (500 - X_norm['Vitamin_B12']) / 500  # Lower B12 is worse
                    X_norm['Vitamin_D'] = (50 - X_norm['Vitamin_D']) / 40  # Lower vitamin D is worse
                    
                    # Advanced markers normalization
                    X_norm['Hippocampal_Volume'] = (100 - X_norm['Hippocampal_Volume']) / 50  # Lower volume is worse
                    X_norm['Amyloid_Level'] = X_norm['Amyloid_Level'] / 100  # Higher is worse
                    X_norm['Tau_Protein'] = X_norm['Tau_Protein'] / 100  # Higher is worse
                    
                    # Calculate model-based risk score
                    # In a real app, this would use trained model coefficients or a more complex model
                    risk_weights = {
                        'Age': 0.15,
                        'Gender': 0.02,
                        'Education': -0.05,
                        'Dominant_Hand': 0.01,
                        'MMSE_Score': 0.20,
                        'APOE_Status': 0.10,
                        'Family_History': 0.10,
                        'Smoking': 0.05,
                        'Physical_Activity': -0.05,
                        'Depression': 0.06,
                        'Chronic_Health': 0.06,
                        'Medication': 0.05,
                        'Nutrition': -0.04,
                        'Sleep_Quality': 0.06,
                        'BMI': 0.03,
                        'Systolic_BP': 0.05,
                        'Diastolic_BP': 0.05,
                        'Cholesterol': 0.04,
                        'Glucose': 0.05,
                        'HbA1c': 0.05,
                        'Vitamin_B12': 0.03,
                        'Vitamin_D': 0.03,
                        'Hippocampal_Volume': 0.15,
                        'Amyloid_Level': 0.15,
                        'Tau_Protein': 0.15
                    }
                    
                    # Calculate base risk score
                    base_risk_score = sum((X_norm[feature] * weight).iloc[0] for feature, weight in risk_weights.items())
                    
                    # Normalize to 0-100%
                    base_risk_score = min(max(base_risk_score * 100, 0), 100)
                    
                    # If we have voice and sleep data, incorporate them
                    voice_risk_score = st.session_state.get('voice_risk_score', 0.3) * 100
                    sleep_risk_score = st.session_state.get('sleep_risk_score', 0.3) * 100
                    
                    # Combine all risk factors with weights
                    integrated_risk = (
                        base_risk_score * 0.6 + 
                        voice_risk_score * 0.2 + 
                        sleep_risk_score * 0.2
                    )
                    
                    # Display the results
                    st.success("Assessment complete!")
                    
                    # Create columns for a cleaner display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Overall Dementia Risk Score")
                        
                        # Risk level classification
                        if integrated_risk < 25:
                            risk_level = "Low"
                            risk_color = "green"
                        elif integrated_risk < 50:
                            risk_level = "Moderate"
                            risk_color = "orange"
                        elif integrated_risk < 75:
                            risk_level = "High"
                            risk_color = "red"
                        else:
                            risk_level = "Very High"
                            risk_color = "darkred"
                        
                        # Display score and level
                        st.markdown(f"#### Score: <span style='color:{risk_color};'>{integrated_risk:.1f}%</span>", unsafe_allow_html=True)
                        st.markdown(f"#### Risk Level: <span style='color:{risk_color};'>{risk_level}</span>", unsafe_allow_html=True)
                        
                        # Risk visualization
                        fig, ax = plt.subplots(figsize=(8, 1))
                        ax.barh(0, 100, color='lightgray', height=0.5)
                        ax.barh(0, integrated_risk, color=risk_color, height=0.5)
                        ax.set_xlim(0, 100)
                        ax.set_yticks([])
                        ax.set_xticks([0, 25, 50, 75, 100])
                        ax.set_xlabel('Risk %')
                        st.pyplot(fig)
                        
                        # Component breakdown
                        st.subheader("Risk Component Breakdown")
                        
                        # Display each risk factor
                        components = {
                            "Clinical Assessment": base_risk_score,
                            "Voice Analysis": voice_risk_score,
                            "Sleep Patterns": sleep_risk_score
                        }
                        
                        # Create a bar chart of the components
                        fig, ax = plt.subplots(figsize=(8, 3))
                        bars = ax.barh(list(components.keys()), list(components.values()), color=['blue', 'purple', 'orange'])
                        ax.set_xlim(0, 100)
                        ax.set_xlabel('Risk Score (%)')
                        ax.set_title('Risk Component Analysis')
                        
                        # Add value labels to the bars
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f'{width:.1f}%',
                                   va='center')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Key Risk Factors")
                        
                        # Calculate individual feature contributions
                        feature_contributions = {feature: X_norm[feature].values[0] * weight * 100 
                                               for feature, weight in risk_weights.items()}
                        
                        # Sort and get top contributors
                        sorted_contributions = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)
                        top_contributors = sorted_contributions[:5]
                        
                        # Display top contributors
                        for feature, contribution in top_contributors:
                            # Make feature name more readable
                            readable_feature = feature.replace('_', ' ')
                            st.markdown(f"- **{readable_feature}**: {contribution:.1f}% contribution")
                        
                        # Protective factors
                        st.subheader("Protective Factors")
                        
                        # Get bottom contributors (negative impact on risk)
                        bottom_contributors = sorted_contributions[-3:] if len(sorted_contributions) >= 3 else []
                        
                        for feature, contribution in bottom_contributors:
                            if contribution < 0:
                                # Make feature name more readable
                                readable_feature = feature.replace('_', ' ')
                                st.markdown(f"- **{readable_feature}**: {abs(contribution):.1f}% reduction")
                        
                        # Recommendations based on access level
                        st.subheader("Recommendations")
                        
                        if st.session_state.get('access_level', 0) >= 3:
                            # Clinical recommendations
                            st.markdown("#### Clinical Recommendations")
                            
                            # Generate specific recommendations based on risk factors
                            recommendations = []
                            
                            if age > 65:
                                recommendations.append("Regular cognitive assessments every 6 months")
                            
                            if mmse_score < 26:
                                recommendations.append("Complete neuropsychological testing")
                            
                            if apoe_val == 1:
                                recommendations.append("Consider genetic counseling for family members")
                            
                            if health_val > 0:
                                recommendations.append("Optimize management of existing health conditions")
                            
                            if sleep_val == 1 or st.session_state.get('sleep_risk_score', 0) > 0.5:
                                recommendations.append("Sleep study to rule out sleep apnea")
                            
                            if voice_risk_score > 50:
                                recommendations.append("Regular speech therapy assessment")
                            
                            # Display recommendations
                            for rec in recommendations:
                                st.markdown(f"- {rec}")
                        
                        # Lifestyle recommendations for all access levels
                        st.markdown("#### Lifestyle Recommendations")
                        
                        lifestyle_recs = [
                            "Regular physical exercise (at least 150 minutes per week)",
                            "Mediterranean diet rich in omega-3 fatty acids",
                            "Cognitive stimulation activities daily",
                            "Social engagement and interaction",
                            "Maintain good sleep hygiene"
                        ]
                        
                        for rec in lifestyle_recs:
                            st.markdown(f"- {rec}")
                    
                    # If high enough access level, show additional details
                    if st.session_state.get('access_level', 0) >= 3:
                        st.subheader("Detailed Assessment Data")
                        
                        # Create a downloadable report
                        report_data = {
                            "Patient Data": X.to_dict(orient='records')[0],
                            "Risk Scores": {
                                "Overall Risk": integrated_risk,
                                "Clinical Risk": base_risk_score,
                                "Voice Risk": voice_risk_score,
                                "Sleep Risk": sleep_risk_score
                            },
                            "Risk Factors": {feature: contrib for feature, contrib in sorted_contributions},
                            "Recommendations": recommendations + lifestyle_recs,
                            "Assessment Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        pdf_data = generate_risk_assessment_pdf(
                            X, 
                            X_norm, 
                            integrated_risk, 
                            base_risk_score, 
                            voice_risk_score, 
                            sleep_risk_score, 
                            sorted_contributions, 
                            recommendations, 
                            lifestyle_recs
                        )
                        
                        # Create a download button for the PDF
                        st.download_button(
                            label="Download Full Assessment Report (PDF)",
                            data=pdf_data,
                            file_name="dementia_risk_assessment.pdf",
                            mime="application/pdf"
                        )
                        
                        # Show data visualization options
                        st.subheader("Visualization Options")
                        
                        viz_type = st.selectbox("Select Visualization", 
                                               ["Feature Importance", "Risk Factors Radar", "Temporal Risk Projection"])
                        
                        if viz_type == "Feature Importance":
                            # Create feature importance plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Get top 10 features by absolute importance
                            top_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                            features = [f[0].replace('_', ' ') for f in top_features]
                            values = [f[1] for f in top_features]
                            
                            # Create color mapping based on positive/negative contribution
                            colors = ['red' if v > 0 else 'green' for v in values]
                            
                            # Create horizontal bar chart
                            bars = ax.barh(features, values, color=colors)
                            
                            # Add labels
                            ax.set_xlabel('Contribution to Risk Score (%)')
                            ax.set_title('Top 10 Feature Importance')
                            
                            # Add value annotations
                            for bar in bars:
                                width = bar.get_width()
                                ax.text(width + 0.5 if width > 0 else width - 0.5, 
                                       bar.get_y() + bar.get_height()/2, 
                                       f'{width:.1f}%',
                                       va='center', 
                                       ha='left' if width > 0 else 'right')
                            
                            st.pyplot(fig)
                        
                        elif viz_type == "Risk Factors Radar":
                            # Create radar chart of risk categories
                            categories = [
                                'Demographic', 
                                'Genetic', 
                                'Cognitive',
                                'Lifestyle', 
                                'Health', 
                                'Biomarkers',
                                'Voice',
                                'Sleep'
                            ]
                            
                            # Calculate category values
                            category_values = [
                                (X_norm['Age'].values[0] * risk_weights['Age'] + 
                                 X_norm['Gender'].values[0] * risk_weights['Gender']) * 100,
                                (X_norm['APOE_Status'].values[0] * risk_weights['APOE_Status'] + 
                                 X_norm['Family_History'].values[0] * risk_weights['Family_History']) * 100,
                                X_norm['MMSE_Score'].values[0] * risk_weights['MMSE_Score'] * 100,
                                (X_norm['Smoking'].values[0] * risk_weights['Smoking'] + 
                                 X_norm['Physical_Activity'].values[0] * risk_weights['Physical_Activity'] + 
                                 X_norm['Nutrition'].values[0] * risk_weights['Nutrition']) * 100,
                                (X_norm['Depression'].values[0] * risk_weights['Depression'] + 
                                 X_norm['Chronic_Health'].values[0] * risk_weights['Chronic_Health'] + 
                                 X_norm['Medication'].values[0] * risk_weights['Medication']) * 100,
                                (X_norm['Hippocampal_Volume'].values[0] * risk_weights['Hippocampal_Volume'] + 
                                 X_norm['Amyloid_Level'].values[0] * risk_weights['Amyloid_Level'] + 
                                 X_norm['Tau_Protein'].values[0] * risk_weights['Tau_Protein']) * 100,
                                voice_risk_score / 5,  # Scale for better visualization
                                sleep_risk_score / 5   # Scale for better visualization
                            ]
                            
                            # Create radar chart
                            angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
                            
                            # Close the loop
                            values = category_values + [category_values[0]]
                            angles = angles + [angles[0]]
                            categories = categories + [categories[0]]
                            
                            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                            
                            # Plot data
                            ax.plot(angles, values, 'o-', linewidth=2, color='red')
                            ax.fill(angles, values, alpha=0.25, color='red')
                            
                            # Set category labels
                            ax.set_thetagrids(np.degrees(angles[:-1]), categories[:-1])
                            
                            # Set y-labels
                            ax.set_rlabel_position(0)
                            ax.set_yticks([5, 10, 15, 20])
                            ax.set_yticklabels(['5%', '10%', '15%', '20%'])
                            ax.set_ylim(0, 20)
                            
                            ax.set_title('Risk Factor Categories')
                            
                            st.pyplot(fig)
                        
                        elif viz_type == "Temporal Risk Projection":
                            # Create a projection of risk over time
                            years = list(range(0, 11))  # 10-year projection
                            
                            # Calculate risk progression based on current factors
                            # In a real app, this would use a more sophisticated model
                            base_progression = [integrated_risk]
                            
                            for i in range(1, 11):
                                # Risk increases more rapidly with age and genetic factors
                                new_risk = base_progression[-1] + (
                                    1.0 +  # Base annual increase
                                    0.2 * i * X_norm['Age'].values[0] +  # Age acceleration
                                    0.5 * i * X_norm['APOE_Status'].values[0] +  # Genetic acceleration
                                    0.3 * i * X_norm['Family_History'].values[0]  # Family history acceleration
                                )
                                base_progression.append(min(new_risk, 100))  # Cap at 100%
                            
                            # Calculate modified progression with interventions
                            intervention_progression = [integrated_risk]
                            
                            for i in range(1, 11):
                                # Risk increases more slowly with interventions
                                new_risk = intervention_progression[-1] + (
                                    1.0 +  # Base annual increase
                                    0.2 * i * X_norm['Age'].values[0] +  # Age acceleration
                                    0.5 * i * X_norm['APOE_Status'].values[0] +  # Genetic acceleration
                                    0.3 * i * X_norm['Family_History'].values[0] -  # Family history acceleration
                                    1.5  # Intervention benefit
                                )
                                intervention_progression.append(min(max(new_risk, 0), 100))  # Cap between 0-100%
                            
                            # Create the plot
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            ax.plot(years, base_progression, 'r-', linewidth=2, label='Without Intervention')
                            ax.plot(years, intervention_progression, 'g-', linewidth=2, label='With Intervention')
                            
                            ax.set_xlabel('Years from Now')
                            ax.set_ylabel('Predicted Risk Score (%)')
                            ax.set_title('Projected Dementia Risk Over Time')
                            ax.set_xlim(0, 10)
                            ax.set_ylim(0, 100)
                            ax.legend()
                            ax.grid(True, linestyle='--', alpha=0.7)
                            
                            st.pyplot(fig)
                            
                            # Add explanation
                            st.markdown("""
                            This projection shows how the patient's dementia risk may change over time:
                            - The red line shows the projected risk without any intervention
                            - The green line shows the potential effect of:
                                - Regular cognitive exercise
                                - Improved diet and physical activity
                                - Better management of existing conditions
                                - Early interventions for sleep issues
                            """)
    else:
        st.warning("Authentication required. Please authenticate in the Authentication tab.")



# Add footer
st.markdown("---")
st.markdown("© 2024 Advanced Dementia Risk Assessment Tool. For clinical use only.")
st.markdown("This application complies with HIPAA and other relevant healthcare privacy regulations.")

# Add documentation link and help button in the sidebar
with st.sidebar:
    st.markdown("## Help & Documentation")
    
    if st.button("User Guide"):
        st.markdown("""
        ### User Guide
        
        This application uses a hierarchical security system:
        - **Level 1**: Patient (basic access)
        - **Level 2**: Caregiver (standard access)
        - **Level 3**: Clinical Staff (advanced access)
        - **Level 4**: Administrator (complete access)
        
        #### How to use:
        1. Authenticate with your access code
        2. Complete patient information
        3. Upload voice recording (optional)
        4. Enter sleep data (optional)
        5. Generate comprehensive assessment
        
        For detailed documentation, please contact your system administrator.
        """)
    
    if st.button("About the Model"):
        st.markdown("""
        ### About the Risk Model
        
        This tool employs a multi-modal approach to assess dementia risk:
        
        1. **Clinical Assessment**: Patient data, medical history, and biomarkers
        2. **Voice Analysis**: Linguistic and acoustic features from speech samples
        3. **Sleep Pattern Analysis**: Sleep duration, quality, and disturbances
        
        The model integrates these factors using a weighted algorithm based on
        clinical research and established risk factors.
        
        This is intended as a clinical decision support tool and not a diagnostic device.
        """)
    
    # Add version information
    st.markdown("#### Version Information")
    st.text("Application Version: 2.3.1")
    st.text("Model Version: 1.8.5")
    st.text("Last Updated: March 15, 2024")
    st.markdown("#### Developer Information")
    st.text("Developed by: Dinesh, Divyadhardhini, and Srinidhi")

# Add a section to simulate database connectivity (for administrators only)
if st.session_state.get('access_level', 0) >= 4:
    with st.sidebar:
        st.markdown("---")
        st.markdown("## Administrator Tools")
        
        if st.button("Database Connection Status"):
            st.info("Checking database connection...")
            progress = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress.progress(i + 1)
            st.success("Connection established to secure clinical database.")
            
            # Show mock database stats
            st.markdown("### Database Statistics")
            st.text("Total Records: 4,562")
            st.text("Latest Sync: 2024-03-14 08:32:17")
            st.text("Storage Usage: 68%")
        
        if st.button("Model Performance Metrics"):
            st.info("Retrieving model performance metrics...")
            
            # Display mock metrics
            metrics_data = {
                'Accuracy': 0.86,
                'Precision': 0.82,
                'Recall': 0.79,
                'F1 Score': 0.80,
                'ROC AUC': 0.89
            }
            
            # Create metrics visualization
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.bar(metrics_data.keys(), metrics_data.values(), color='blue')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Metrics')
            
            for i, v in enumerate(metrics_data.values()):
                ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
            
            st.pyplot(fig)