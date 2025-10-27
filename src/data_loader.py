import pandas as pd
import zipfile
import os
from pathlib import Path
from typing import Tuple
import numpy as np

class BAFDataLoader:
    """Load and preprocess Bank Account Fraud (BAF) dataset."""
    
    def __init__(self, zip_path: str = "data/archive.zip"):
        self.zip_path = zip_path
        self.data_dir = Path("data")
        self.df = None
        
    def extract_and_load(self, variant: str = "Base") -> pd.DataFrame:
        """
        Extract data from zip and load a specific variant.
        
        Args:
            variant: Dataset variant to load. Options:
                - "Base" (no induced bias)
                - "variant_I" (group size disparity)
                - "variant_II" (prevalence disparity)
                - "variant_III" (separability disparity)
                - "variant_IV" (temporal prevalence disparity)
                - "variant_V" (temporal separability disparity)
        """
        # Extract if not already extracted
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            print(f"Extracting {self.zip_path}...")
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            print("Extraction complete!")
        
        # Map variant names to actual file names
        variant_map = {
            "Base": "Base.csv",
            "variant_I": "Variant I.csv",
            "variant_II": "Variant II.csv",
            "variant_III": "Variant III.csv",
            "variant_IV": "Variant IV.csv",
            "variant_V": "Variant V.csv"
        }
        
        # Get the actual filename
        filename = variant_map.get(variant, f"{variant}.csv")
        csv_file = self.data_dir / filename
        
        if not csv_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {csv_file}")
        
        print(f"Loading {csv_file}...")
        self.df = pd.read_csv(csv_file)
        print(f"Loaded {len(self.df):,} records with {len(self.df.columns)} features")
        
        return self.df
    
    def get_train_test_split(self, test_months: list = [6, 7]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data based on temporal information (month column).
        
        Args:
            test_months: List of months to use for testing (validation set)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call extract_and_load() first.")
        
        # Separate features and target
        X = self.df.drop(['fraud_bool'], axis=1)
        y = self.df['fraud_bool']
        
        # Temporal split
        train_mask = ~X['month'].isin(test_months)
        test_mask = X['month'].isin(test_months)
        
        X_train = X[train_mask].drop('month', axis=1)
        X_test = X[test_mask].drop('month', axis=1)
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        print(f"Training set: {len(X_train):,} samples ({y_train.sum():,} frauds, {(y_train.sum()/len(y_train)*100):.2f}%)")
        print(f"Test set: {len(X_test):,} samples ({y_test.sum():,} frauds, {(y_test.sum()/len(y_test)*100):.2f}%)")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_info(self) -> dict:
        """Get information about features in the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call extract_and_load() first.")
        
        feature_info = {
            "total_features": len(self.df.columns),
            "numeric_features": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_features": list(self.df.select_dtypes(include=['object']).columns),
            "total_records": len(self.df),
            "fraud_rate": f"{(self.df['fraud_bool'].sum() / len(self.df) * 100):.2f}%",
            "missing_values": self.df.isnull().sum().to_dict()
        }
        
        return feature_info