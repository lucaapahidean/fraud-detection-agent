import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import sys
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import BAFDataLoader
from models import FraudDetectionModels
from fraud_agents import fraud_detection_agent, data_exploration_agent, set_models_manager
from agents import Runner

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Bank Fraud Detection AI",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = BAFDataLoader()
if 'models_manager' not in st.session_state:
    st.session_state.models_manager = FraudDetectionModels()
    set_models_manager(st.session_state.models_manager)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("🏦 Fraud Detection AI")
    st.markdown("---")
    
    # Step 1: Data Loading
    st.subheader("1️⃣ Load Dataset")
    
    dataset_variant = st.selectbox(
        "Select Dataset Variant",
        ["Base", "variant_I", "variant_II", "variant_III", "variant_IV", "variant_V"],
        help="Base has no induced bias. Variants have different types of bias."
    )
    
    if st.button("📂 Load Data", type="primary"):
        with st.spinner("Loading dataset..."):
            try:
                df = st.session_state.data_loader.extract_and_load(dataset_variant)
                st.session_state.data_loaded = True
                st.success(f"✅ Loaded {len(df):,} records!")
            except Exception as e:
                st.error(f"❌ Error loading data: {e}")
    
    # Step 2: Model Training
    st.markdown("---")
    st.subheader("2️⃣ Train Models")
    
    if st.session_state.data_loaded:
        use_smote = st.checkbox("Use SMOTE (balance classes)", value=True)
        
        if st.button("🤖 Train Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes"):
                try:
                    X_train, X_test, y_train, y_test = st.session_state.data_loader.get_train_test_split()
                    
                    results = st.session_state.models_manager.train_models(
                        X_train, y_train, X_test, y_test, use_smote=use_smote
                    )
                    
                    st.session_state.models_trained = True
                    st.session_state.training_results = results
                    st.success("✅ Models trained successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error training models: {e}")
    else:
        st.info("Load data first to train models")
    
    # Check for cached models on startup
    if not st.session_state.models_trained:
        if st.session_state.models_manager.load_models():
            st.session_state.models_trained = True
            st.info("✅ Models loaded from cache")
    
    # Step 3: Dataset Info
    if st.session_state.data_loaded:
        st.markdown("---")
        st.subheader("📊 Dataset Info")
        
        feature_info = st.session_state.data_loader.get_feature_info()
        st.metric("Total Records", f"{feature_info['total_records']:,}")
        st.metric("Fraud Rate", feature_info['fraud_rate'])
        st.metric("Features", feature_info['total_features'])

# Main content
st.title("🏦 Bank Account Fraud Detection with AI Agents")
st.markdown("### Powered by OpenAI Agents SDK, scikit-learn & Streamlit")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["🤖 AI Assistant", "📊 Dataset Explorer", "📈 Model Performance", "🔍 Quick Predict"])

# Tab 1: AI Assistant (Chat Interface)
with tab1:
    st.header("Chat with Fraud Detection Agent")
    st.markdown("Ask questions about fraud detection, model predictions, or dataset insights!")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train models first using the sidebar to enable fraud predictions.")
    
    # Chat input at the top
    if prompt := st.chat_input("Ask me anything about fraud detection..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get agent response
        with st.spinner("Thinking..."):
            try:
                # Run agent
                result = asyncio.run(Runner.run(fraud_detection_agent, prompt))
                response = result.final_output
                
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        
        st.rerun()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Tab 2: Dataset Explorer
with tab2:
    st.header("Explore the BAF Dataset")
    
    if st.session_state.data_loaded:
        df = st.session_state.data_loader.df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications", f"{len(df):,}")
        with col2:
            fraud_count = df['fraud_bool'].sum()
            st.metric("Fraudulent", f"{fraud_count:,}")
        with col3:
            legitimate_count = len(df) - fraud_count
            st.metric("Legitimate", f"{legitimate_count:,}")
        with col4:
            fraud_rate = (fraud_count / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        st.markdown("---")
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(100), width='stretch')
        
        # Feature statistics
        st.subheader("📊 Feature Statistics")
        st.dataframe(df.describe(), width='stretch')
        
    else:
        st.info("Load the dataset using the sidebar to explore it.")

# Tab 3: Model Performance
with tab3:
    st.header("Model Performance Metrics")
    
    if st.session_state.models_trained and 'training_results' in st.session_state:
        results = st.session_state.training_results
        
        # Display metrics for each model
        for model_name, metrics in results.items():
            st.subheader(f"📊 {metrics['model_name']}")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            with col5:
                st.metric("AUC-ROC", f"{metrics['auc_roc']:.4f}")
            
            # Confusion Matrix
            st.write("**Confusion Matrix:**")
            cm_df = pd.DataFrame(
                metrics['confusion_matrix'],
                columns=['Predicted Legitimate', 'Predicted Fraud'],
                index=['Actual Legitimate', 'Actual Fraud']
            )
            st.dataframe(cm_df, width='stretch')
            
            st.markdown("---")
        
        # Feature Importance
        st.subheader("🎯 Top Fraud Indicators (XGBoost)")
        try:
            importance = st.session_state.models_manager.get_feature_importance('xgboost', top_n=15)
            importance_df = pd.DataFrame(
                list(importance.items()),
                columns=['Feature', 'Importance']
            ).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature'))
        except Exception as e:
            st.error(f"Could not load feature importance: {e}")
    
    else:
        st.info("Train models first to view performance metrics.")

# Tab 4: Quick Predict
with tab4:
    st.header("🔍 Quick Fraud Prediction")
    
    if st.session_state.models_trained:
        st.markdown("Enter application details to get an instant fraud prediction:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            income = st.slider("Income (decile)", 0.1, 0.9, 0.5, 0.1)
            customer_age = st.slider("Customer Age", 10, 90, 40, 10)
            credit_risk_score = st.slider("Credit Risk Score", -191, 389, 100, 10)
        
        with col2:
            proposed_credit_limit = st.slider("Proposed Credit Limit", 200, 2000, 1000, 100)
            employment_status = st.selectbox(
                "Employment Status",
                ["CA", "CB", "CC", "CD", "CE", "CF", "CG"]
            )
            model_choice = st.selectbox(
                "Model",
                ["xgboost", "random_forest", "logistic_regression"]
            )
        
        if st.button("🔮 Predict Fraud Risk", type="primary"):
            with st.spinner("Analyzing application..."):
                try:
                    application_data = {
                        "income": income,
                        "customer_age": customer_age,
                        "credit_risk_score": credit_risk_score,
                        "proposed_credit_limit": proposed_credit_limit,
                        "employment_status": employment_status,
                        "intended_balcon_amount": 50,
                        "name_email_similarity": 0.5,
                        "prev_address_months_count": 12,
                        "current_address_months_count": 24,
                        "days_since_request": 0,
                        "payment_type": "AB",
                        "zip_count_4w": 10,
                        "velocity_6h": 1,
                        "velocity_24h": 2000,
                        "velocity_4w": 4000,
                        "bank_branch_count_8w": 100,
                        "date_of_birth_distinct_emails_4w": 1,
                        "email_is_free": 1,
                        "housing_status": "BA",
                        "phone_home_valid": 1,
                        "phone_mobile_valid": 1,
                        "bank_months_count": 0,
                        "has_other_cards": 0,
                        "foreign_request": 0,
                        "source": "INTERNET",
                        "session_length_in_minutes": 5,
                        "device_os": "Windows",
                        "keep_alive_session": 1,
                        "device_distinct_emails": 1,
                        "device_fraud_count": 0,
                    }
                    
                    result = st.session_state.models_manager.predict_fraud(
                        application_data, model_choice
                    )
                    
                    # Display result
                    st.markdown("---")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        prediction = "🚨 FRAUD" if result['is_fraud'] else "✅ LEGITIMATE"
                        st.metric("Prediction", prediction)
                    
                    with col2:
                        st.metric("Fraud Probability", f"{result['fraud_probability']:.2%}")
                    
                    with col3:
                        st.metric("Risk Level", result['risk_level'])
                    
                    # Recommendation
                    if result['is_fraud']:
                        st.error("⛔ **Recommendation:** REJECT APPLICATION")
                    elif result['risk_level'] == "MEDIUM":
                        st.warning("⚠️ **Recommendation:** MANUAL REVIEW REQUIRED")
                    else:
                        st.success("✅ **Recommendation:** APPROVE APPLICATION")
                
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
    
    else:
        st.info("Train models first to make predictions.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using OpenAI Agents SDK, scikit-learn, and Streamlit</p>
    <p><small>Bank Account Fraud (BAF) Dataset by Feedzai</small></p>
</div>
""", unsafe_allow_html=True)