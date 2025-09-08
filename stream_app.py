import joblib
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

@st.cache_resource
def load_model():
    model_file = 'model/diabetes_class.pkl'
    model_dict = joblib.load(model_file)
    return model_dict['model'], model_dict['features'], model_dict['threshold']

model, features, threshold = load_model()

st.set_page_config(page_title="Diabetes Prescreening", 
                   layout="wide", 
                   initial_sidebar_state="expanded"
                   )

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", ["Diabetes Prediction", "Model Information"])
st.sidebar.header("About :")
st.sidebar.info(
    "This app predicts diabetes risk using **Logistic Regression**, "
    "selected over Random Forest and LightGBM after rigorous evaluation.\n\n"
    "The dataset comes from the CDC (253,680 participants). "
    "We simplified the survey into a few demographic and lifestyle questions."
)


if page == "Diabetes Prediction":
    st.title("Diabetes Risk Assessment from Lifestyle Factors")

    st.markdown("""
    <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
    <h4 style='color: #1e3a8a; margin: 0;'>Healthcare Disclaimer</h4>
    <p style='margin-bottom: 0;'>This tool is for educational purposes only and should not replace professional medical advice. 
    Always consult healthcare professionals for medical decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create two columns for better layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Demographic")

        Sex_place = st.radio("Gender", ["Male", "Female"], horizontal=True)
        Sex = 1 if Sex_place == 'Male' else 0 # convert from placeholder to real variable

        Age_place = st.number_input("Age", min_value=18, max_value=99,value=30)
        # Mapping Age
        a = [18,25,29,34,39,44,49,54,59,64,69,74,79,100]
        idx = list(range(1,len(a)))
        Age = None
        for i in range(len(a)-1):
            if (Age_place >= a[i]) and (Age_place < a[i+1]):
              Age = idx[i]
              break
        if Age is None:
            Age = len(a) - 1

        Edu_place = st.selectbox("Education", [
            "Never attended school or only kindergarten", 
            "Grades 1 through 8 (Elementary)",
            "Grades 9 through 11 (Some high school)",
            "Grade 12 or GED (High school graduate)",
            "College 1 year to 3 years (or Technical school)",
            "College 4 years or more (College graduate)"
            ], index=3)

        edu_mapping = {
            "Never attended school or only kindergarten": 1,
            "Grades 1 through 8 (Elementary)": 2,
            "Grades 9 through 11 (Some high school)": 3,
            "Grade 12 or GED (High school graduate)": 4,
            "College 1 year to 3 years (or Technical school)": 5,
            "College 4 years or more (College graduate)": 6
        }
        Education = edu_mapping[Edu_place]

        Inc_place = st.selectbox("Annual Income", [
            "Less than $10,000", 
            "$10,000 - $14,999",
            "$15,000 - $19,999",
            "$20,000 - $24,999",
            "$25,000 - $34,999",
            "$35,000 - $49,999",
            "$50,000 - $74,999",
            "$75,000 or more"
            ], index=4)

        inc_mapping = {
            "Less than $10,000": 1,
            "$10,000 - $14,999": 2,
            "$15,000 - $19,999": 3,
            "$20,000 - $24,999": 4,
            "$25,000 - $34,999": 5,
            "$35,000 - $49,999": 6,
            "$50,000 - $74,999": 7,
            "$75,000 or more": 8
        }
        Income = inc_mapping[Inc_place]

        Weight = st.number_input("Weight (kg)", min_value=2, max_value=200, value=70)
        Height = st.number_input("Height (cm)", min_value=10, max_value=250, value=170)
        # Calculating BMI
        BMI = Weight / ((Height*0.01)**2)
        b = [0, 18.5, 25, 30]
        b_st = ['Underweight', 'Healthy', 'Overweight', 'Obese']
        BMI_status = 'Obese'  # Default for BMI >= 30
        for i in range(len(b)-1):
            if (BMI > b[i]) and (BMI <= b[i+1]):
                BMI_status = b_st[i]
                break

    with col_right:
        st.subheader("Lifestyle Factors")

        GenHlth = st.select_slider(
            "How would you rate your general health?", 
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}[x],
            value=3
        )

        Smoker_place = st.selectbox("Have you smoked at least 100 cigarettes in your entire life? [5 packs = 100 cigarettes]", ["Yes", "No"], index = 0)
        Smoker = 1 if Smoker_place == 'Yes' else 0

        Phys_place = st.selectbox("Do you have physical activities in past 30 days - not including job?", ["Yes", "No"], index = 0)
        PhysActivity = 1 if Phys_place == 'Yes' else 0

        Fruit_place = st.selectbox("Do you consume any fruits at least one per day?", ["Yes", "No"], index = 0)
        Fruits = 1 if Fruit_place == 'Yes' else 0

        Veggie_place = st.selectbox("Do you consume any vegetables at least one per day?", ["Yes", "No"], index = 0)
        Veggies = 1 if Veggie_place == 'Yes' else 0

        Alco_place = st.selectbox("Heavy alcohol drinker? (Men: >14 drinks/week, Women: >7 drinks/week)", ["Yes", "No"], index = 0)
        HvyAlcoholConsump = 1 if Alco_place == 'Yes' else 0

        Health_place = st.selectbox("Do you have any kind of healthcare coverage, including health insurance, prepaid plans such as HMO, etc?", ["Yes", "No"], index = 0)
        AnyHealthcare = 1 if Health_place == 'Yes' else 0

        Nodo_place = st.selectbox("In past 12 months, was there a time you needed to see a doctor but couldn't because of cost?", ["No", "Yes"], index=0)
        NoDocbcCost = 1 if Nodo_place == 'Yes' else 0

        Walk_place = st.selectbox("Do you have serious difficulty walking or climbing stairs?", ["Yes", "No"], index = 0)
        DiffWalk = 1 if Walk_place == 'Yes' else 0


    # Prediction section
    st.markdown("---")

    input_data = {
        "Sex": Sex,
        "Age" : Age,
        "Education": Education,
        "Income": Income,
        "Smoker": Smoker,
        "PhysActivity": PhysActivity,
        "Fruits": Fruits,
        "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump,
        "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost,
        "GenHlth": GenHlth,
        "DiffWalk": DiffWalk,
        "BMI": BMI,
        }
    
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Assess Diabetes Risk", type="primary", use_container_width=True):
            # Make prediction
            X = pd.DataFrame([input_data])[features]
            y_proba = model.predict_proba(X)[0, 1]
            y_pred = int(y_proba >= threshold)

            st.markdown("## Risk Assessment Results")
            st.write(f"**Your BMI**: {BMI:.3f}")
            st.write(f"**Your BMI Status**: {BMI_status}")

            st.write(f"**Risk Probability**: {y_proba:.3f}")
            st.write(f"**Predicted Class**: {y_pred} (1 = Diabetes, 0 = No Diabetes)")

            # Visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = y_proba * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Diabetes Risk Probability (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 55], 'color': "yellow"},
                        {'range': [55, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold * 100
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.metric("Risk Probability", f"{y_proba:.1%}")
                st.metric("Model Threshold", f"{threshold:.1%}")
            
            with col_res2:
                if y_pred == 1:
                    st.error("⚠️ **Elevated Risk Detected**")
                    st.warning("Consider consulting with a healthcare professional for proper evaluation.")
                else:
                    st.success("✅ **Lower Risk Profile**")
                    st.info("Maintain your healthy lifestyle habits!")
            
            # Risk factors contribution (simplified visualization)
            st.markdown("### 🎯 Key Risk Factors in Your Profile")
            
            risk_factors = []
            if GenHlth >= 4:
                risk_factors.append("Poor/Fair general health")
            if Smoker:
                risk_factors.append("Smoking history")
            if not PhysActivity:
                risk_factors.append("Limited physical activity")
            if Age >= 6:
                risk_factors.append("Age factor")
            if not Fruits or not Veggies:
                risk_factors.append("Limited fruit/vegetable intake")
            if HvyAlcoholConsump:
                risk_factors.append("Heavy alcohol consumption")
            if DiffWalk:
                risk_factors.append("Mobility difficulties")
            
            if risk_factors:
                st.warning("**Identified risk factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.success("**Your profile shows generally healthy lifestyle choices!**")

elif page == "Model Information":
    st.title("📈 Model Information & Performance")
    
    # Model Overview
    st.markdown("Model Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Logistic Regression Model
        
        Our diabetes prediction model uses **Logistic Regression** with L1 regularization, 
        selected after comprehensive evaluation against Random Forest and LightGBM models.
        
        **Why Logistic Regression?**
        - **Interpretable**: Clear understanding of how each factor affects risk
        - **Reliable**: Consistent performance across different data splits  
        - **Efficient**: Fast predictions with minimal computational requirements
        - **Probabilistic**: Provides meaningful probability estimates
        """)
    
    with col2:
        # Performance metrics
        st.markdown("### Model Performance")
        st.metric("ROC-AUC Score", "0.759 ± 0.001")
        st.metric("Features Used", "13 lifestyle factors")
        st.metric("Optimal Threshold", "0.547")
        st.metric("Regularization", "L1 (C=0.01)")
    
    # Dataset Information
    st.markdown("---")
    st.markdown("## About the Dataset")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Diabetes Health Indicators Dataset
        
        **Source**: Centers for Disease Control and Prevention (CDC)
        
        **Study Population**: 253,680 participants across the United States
        
        **Purpose**: Understanding the relationship between lifestyle factors and diabetes risk
        
        **Data Type**: Cross-sectional survey including demographics, lifestyle behaviors, and health indicators
                    
        **Sources** :
        """)
        st.link_button("UCI repository", "https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators")
    
    with col2:
        # Dataset statistics
        dataset_stats = pd.DataFrame({
            'Metric': ['Total Participants', 'Features Available', 'Lifestyle Features', 'Medical Features', 'Demographics'],
            'Value': ['253,680', '35', '9', '6', '4']
        })
        st.dataframe(dataset_stats, hide_index=True)
    
    # Feature Analysis
    st.markdown("---")
    st.markdown("## Feature Analysis")
    
    # Feature comparison chart
    performance_data = {
        'Feature Set': ['Lifestyle Only', 'Medical Only', 'All Features'],
        'ROC-AUC': [0.744, 0.783, 0.787],
        'F1-Score': [0.376, 0.400, 0.407],
        'Features Count': [13, 10, 19]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(perf_df, x='Feature Set', y='ROC-AUC', 
                     title='Model Performance by Feature Set',
                     color='ROC-AUC', color_continuous_scale='viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(perf_df, x='Features Count', y='ROC-AUC', 
                        size='F1-Score', hover_name='Feature Set',
                        title='Performance vs Complexity Trade-off')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature categories
    st.markdown("## Feature Categories")
    
    tab1, tab2, tab3 = st.tabs(["Lifestyle Features (Used)", "Medical Features", "All Features"])
    
    with tab1:
        lifestyle_features = [
            "Smoking history", "Physical activity", "Daily fruit consumption", 
            "Daily vegetable consumption", "Heavy alcohol consumption",
            "Healthcare coverage", "Cost barriers to healthcare", 
            "General health rating", "Walking difficulties",
            "Age", "Sex", "Education level", "Income level"
        ]
        
        st.markdown("**These 13 lifestyle and demographic factors are used in our current model:**")
        for i, feature in enumerate(lifestyle_features, 1):
            st.write(f"{i}. {feature}")
    
    with tab2:
        medical_features = [
            "High blood pressure", "High cholesterol", "Cholesterol screening",
            "BMI", "Stroke history", "Heart disease/heart attack history"
        ]
        
        st.markdown("**Medical examination features (not used in current model):**")
        for i, feature in enumerate(medical_features, 1):
            st.write(f"{i}. {feature}")
        
        st.info("💡 **Note**: Medical features show higher predictive power but require clinical testing. Our model focuses on easily obtainable lifestyle information.")
    
    with tab3:
        st.markdown("**Complete feature set combines both categories:**")
        st.write("- **Lifestyle + Demographics**: 13 features")  
        st.write("- **Medical Examination**: 6 features")
        st.write("- **Total**: 19 unique features")
        
        st.markdown("**Model Selection Rationale:**")
        st.write("**Lifestyle Model Selected** - Accessible screening without medical tests")
        st.write("**Medical Model** - Higher accuracy but requires clinical visit")
        st.write("**Combined Model** - Balanced approach but more complex")
    
    # Model Validation
    st.markdown("---")
    st.markdown("## Model Validation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Cross-Validation Results
        
        **Method**: 5-fold Stratified Cross-Validation
        
        **Performance Consistency**: 
        - Mean ROC-AUC: 0.759 ± 0.001
        - Low standard deviation indicates stable performance
        
        **Model Selection**: 
        - Compared 3 algorithms (RF, LR, LGBM)
        - Logistic Regression selected for interpretability and reliability
        """)
    
    with col2:
        st.markdown("""
        ### Model Limitations
        
        **Important Considerations**:
        - Screening tool only, not diagnostic
        - Based on self-reported lifestyle data
        - Population-level patterns may not apply individually
        - Medical evaluation still recommended for definitive diagnosis
        
        **Recommended Use**: 
        Initial risk assessment and lifestyle awareness
                    
        **More information about how we train the model: 
        """)
        st.link_button("Visit Repository", "https://github.com/ignsagita/class-diabetes-lifestyle")
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #856404;'>
    <h4 style='color: #856404; margin-top: 0;'>Important Medical Disclaimer</h4>
    <p style='margin-bottom: 0;'>
    This prediction model is intended for educational and screening purposes only. It should not be used as a substitute 
    for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers 
    with questions about your medical conditions. The model's predictions are based on population-level patterns and 
    may not accurately reflect individual risk factors or health status.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8em;'>"
    "Diabetes Risk Assessment Tool | Based on CDC Health Survey Data | For Educational Purposes Only"
    "</div>", 
    unsafe_allow_html=True
)
