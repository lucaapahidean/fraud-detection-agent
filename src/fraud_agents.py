from agents import Agent, function_tool, RunContextWrapper
from typing import Dict, Any, Annotated
import json
from models import FraudDetectionModels

# Initialize the models manager (will be set from app)
models_manager: FraudDetectionModels = None

def set_models_manager(manager: FraudDetectionModels):
    """Set the global models manager."""
    global models_manager
    models_manager = manager

@function_tool
def predict_fraud_risk(
    income: Annotated[float, "Annual income in decile form (0.1-0.9)"],
    customer_age: Annotated[int, "Customer age in years, rounded to decade (10-90)"],
    credit_risk_score: Annotated[int, "Internal credit risk score (-191 to 389)"],
    intended_balcon_amount: Annotated[float, "Initial balance amount (-16 to 114)"],
    employment_status: Annotated[str, "Employment status code (e.g., CA, CB, CC, CD, CE, CF, CG)"],
    proposed_credit_limit: Annotated[int, "Proposed credit limit (200-2000)"],
    model_name: Annotated[str, "Model to use: logistic_regression, random_forest, or xgboost"] = "xgboost"
) -> str:
    """
    Predict fraud risk for a bank account application using machine learning models.
    Returns fraud probability, risk level, and recommendation.
    """
    if models_manager is None:
        return json.dumps({"error": "Models not initialized"})
    
    # Create application data
    application_data = {
        "income": income,
        "customer_age": customer_age,
        "credit_risk_score": credit_risk_score,
        "intended_balcon_amount": intended_balcon_amount,
        "employment_status": employment_status,
        "proposed_credit_limit": proposed_credit_limit,
        # Set defaults for other features
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
    
    try:
        result = models_manager.predict_fraud(application_data, model_name)
        
        recommendation = "REJECT" if result['is_fraud'] else "APPROVE"
        if result['risk_level'] == "MEDIUM":
            recommendation = "MANUAL_REVIEW"
        
        response = {
            "fraud_prediction": "FRAUD" if result['is_fraud'] else "LEGITIMATE",
            "fraud_probability": f"{result['fraud_probability']:.2%}",
            "risk_level": result['risk_level'],
            "recommendation": recommendation,
            "model_used": result['model_used'],
            "key_factors": {
                "income_decile": income,
                "age": customer_age,
                "credit_score": credit_risk_score,
                "proposed_limit": proposed_credit_limit
            }
        }
        
        return json.dumps(response, indent=2)
    
    except Exception as e:
        return json.dumps({"error": str(e)})

@function_tool
def compare_models(
    income: Annotated[float, "Annual income in decile form (0.1-0.9)"],
    customer_age: Annotated[int, "Customer age in years"],
    credit_risk_score: Annotated[int, "Internal credit risk score"],
    proposed_credit_limit: Annotated[int, "Proposed credit limit"]
) -> str:
    """
    Compare predictions from all available models for the same application.
    Returns predictions from Logistic Regression, Random Forest, and XGBoost.
    """
    if models_manager is None:
        return json.dumps({"error": "Models not initialized"})
    
    models = ["logistic_regression", "random_forest", "xgboost"]
    comparison = {}
    
    application_data = {
        "income": income,
        "customer_age": customer_age,
        "credit_risk_score": credit_risk_score,
        "proposed_credit_limit": proposed_credit_limit,
        "employment_status": "CA",
        "intended_balcon_amount": 50,
        # Defaults for other features
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
    
    for model_name in models:
        try:
            result = models_manager.predict_fraud(application_data, model_name)
            comparison[model_name] = {
                "prediction": "FRAUD" if result['is_fraud'] else "LEGITIMATE",
                "probability": f"{result['fraud_probability']:.2%}",
                "risk_level": result['risk_level']
            }
        except Exception as e:
            comparison[model_name] = {"error": str(e)}
    
    return json.dumps(comparison, indent=2)

@function_tool
def get_model_performance() -> str:
    """
    Get performance metrics for all trained models.
    Returns accuracy, precision, recall, F1-score, and AUC-ROC for each model.
    """
    if models_manager is None or not models_manager.models:
        return json.dumps({"error": "Models not trained yet"})
    
    performance = {}
    
    # Note: This would need access to stored metrics
    # For now, return model names
    performance = {
        "available_models": list(models_manager.models.keys()),
        "note": "Run model training to see detailed performance metrics"
    }
    
    return json.dumps(performance, indent=2)

@function_tool
def get_top_fraud_indicators(model_name: Annotated[str, "Model name"] = "xgboost") -> str:
    """
    Get the top features that indicate fraud risk according to a specific model.
    Returns feature importance rankings.
    """
    if models_manager is None:
        return json.dumps({"error": "Models not initialized"})
    
    try:
        feature_importance = models_manager.get_feature_importance(model_name, top_n=10)
        
        # Format for readability
        formatted = {
            "model": model_name,
            "top_fraud_indicators": [
                {"feature": feature, "importance": f"{importance:.4f}"}
                for feature, importance in feature_importance.items()
            ]
        }
        
        return json.dumps(formatted, indent=2)
    
    except Exception as e:
        return json.dumps({"error": str(e)})


# Define the fraud detection agent
fraud_detection_agent = Agent(
    name="Fraud Detection Analyst",
    instructions="""You are an expert fraud detection analyst with access to machine learning models 
    trained on bank account fraud data. You help users:
    
    1. Predict fraud risk for new bank account applications
    2. Compare predictions across different ML models
    3. Explain model performance and important fraud indicators
    4. Provide recommendations on whether to approve, reject, or manually review applications
    
    When analyzing applications, consider factors like:
    - Customer age and income level
    - Credit risk score
    - Proposed credit limit
    - Employment status
    
    Always provide clear explanations and actionable recommendations based on the model predictions.
    Be transparent about prediction confidence and limitations.""",
    
    tools=[
        predict_fraud_risk,
        compare_models,
        get_model_performance,
        get_top_fraud_indicators
    ],
    
    model="gpt-4o"
)

# Data exploration agent
data_exploration_agent = Agent(
    name="Data Exploration Assistant",
    instructions="""You are a data science assistant helping users understand the Bank Account 
    Fraud (BAF) dataset. You provide insights about:
    
    1. Dataset composition and statistics
    2. Feature distributions and correlations
    3. Fraud patterns and trends
    4. Data quality and preprocessing steps
    
    Explain complex concepts in simple terms and provide actionable insights.""",
    
    model="gpt-4o"
)