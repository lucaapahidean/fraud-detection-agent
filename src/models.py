import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import json

class FraudDetectionModels:
    """Train and manage fraud detection models."""
    
    def __init__(self, cache_dir: str = "models_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = None
        self.categorical_features = ['payment_type', 'employment_status', 
                                     'housing_status', 'source', 'device_os']
        
    def preprocess_data(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Preprocess features: handle categoricals, scale numerics."""
        X = X.copy()
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in X.columns:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    X[col] = self.encoders[col].fit_transform(X[col].astype(str))
                else:
                    # Handle unseen categories
                    le = self.encoders[col]
                    X[col] = X[col].astype(str).apply(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        # Scale numerical features
        if fit:
            self.scalers['standard'] = StandardScaler()
            X_scaled = self.scalers['standard'].fit_transform(X)
        else:
            X_scaled = self.scalers['standard'].transform(X)
        
        return pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    use_smote: bool = True) -> Dict[str, Any]:
        """Train multiple models and return performance metrics."""
        
        # Preprocess
        X_train_processed = self.preprocess_data(X_train, fit=True)
        X_test_processed = self.preprocess_data(X_test, fit=False)
        
        # Apply SMOTE for class imbalance
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_processed, y_train = smote.fit_resample(X_train_processed, y_train)
            print(f"After SMOTE: {len(X_train_processed)} samples")
        
        results = {}
        
        # Model 1: Logistic Regression
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        lr.fit(X_train_processed, y_train)
        self.models['logistic_regression'] = lr
        results['logistic_regression'] = self._evaluate_model(
            lr, X_test_processed, y_test, "Logistic Regression"
        )
        
        # Model 2: Random Forest
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                    random_state=42, n_jobs=-1, class_weight='balanced')
        rf.fit(X_train_processed, y_train)
        self.models['random_forest'] = rf
        results['random_forest'] = self._evaluate_model(
            rf, X_test_processed, y_test, "Random Forest"
        )
        
        # Model 3: XGBoost
        print("\nTraining XGBoost...")
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train_processed, y_train)
        self.models['xgboost'] = xgb_model
        results['xgboost'] = self._evaluate_model(
            xgb_model, X_test_processed, y_test, "XGBoost"
        )
        
        # Save models
        self.save_models()
        
        return results
    
    def _evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict[str, Any]:
        """Evaluate a single model."""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Calculate precision at different recall levels
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        
        metrics = {
            "model_name": model_name,
            "accuracy": report['accuracy'],
            "precision": report['1']['precision'],
            "recall": report['1']['recall'],
            "f1_score": report['1']['f1-score'],
            "auc_roc": auc_score,
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")
        
        return metrics
    
    def predict_fraud(self, application_data: Dict[str, Any], 
                     model_name: str = 'xgboost') -> Dict[str, Any]:
        """Predict fraud for a single application."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train models first.")
        
        # Convert to DataFrame
        df = pd.DataFrame([application_data])
        
        # Ensure all features are present
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_names]
        
        # Preprocess
        df_processed = self.preprocess_data(df, fit=False)
        
        # Predict
        model = self.models[model_name]
        prediction = model.predict(df_processed)[0]
        probability = model.predict_proba(df_processed)[0][1]
        
        return {
            "is_fraud": bool(prediction),
            "fraud_probability": float(probability),
            "risk_level": "HIGH" if probability > 0.7 else "MEDIUM" if probability > 0.4 else "LOW",
            "model_used": model_name
        }
    
    def get_feature_importance(self, model_name: str = 'xgboost', top_n: int = 10) -> Dict[str, float]:
        """Get top feature importances from a model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            # Sort and get top N
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:top_n]
            return dict(sorted_features)
        else:
            return {}
    
    def save_models(self):
        """Save trained models to disk."""
        for name, model in self.models.items():
            joblib.dump(model, self.cache_dir / f"{name}.pkl")
        
        joblib.dump(self.scalers, self.cache_dir / "scalers.pkl")
        joblib.dump(self.encoders, self.cache_dir / "encoders.pkl")
        joblib.dump(self.feature_names, self.cache_dir / "feature_names.pkl")
        
        print(f"\nModels saved to {self.cache_dir}")
    
    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            model_files = list(self.cache_dir.glob("*.pkl"))
            if not model_files:
                return False
            
            for model_file in model_files:
                if model_file.name == "scalers.pkl":
                    self.scalers = joblib.load(model_file)
                elif model_file.name == "encoders.pkl":
                    self.encoders = joblib.load(model_file)
                elif model_file.name == "feature_names.pkl":
                    self.feature_names = joblib.load(model_file)
                else:
                    model_name = model_file.stem
                    self.models[model_name] = joblib.load(model_file)
            
            print(f"Loaded {len(self.models)} models from cache")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False