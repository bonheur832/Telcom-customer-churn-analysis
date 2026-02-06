#!/usr/bin/env python3
"""
Main script to orchestrate the entire telco churn analysis pipeline
"""

import sys
import os
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.model_training import ChurnPredictor


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("TELCO CUSTOMER CHURN ANALYSIS PIPELINE")
    print("=" * 60)

    # Step 1: Data Cleaning
    print("\nSTEP 1: DATA CLEANING")
    print("-" * 40)

    cleaner = DataCleaner('data/Telco_customer_churn.csv')
    df_clean = cleaner.clean_data()
    cleaner.save_cleaned_data(df_clean, 'data/cleaned_customer_churn.csv')

    # Step 2: Feature Engineering
    print("\n\nSTEP 2: FEATURE ENGINEERING")
    print("-" * 40)

    engineer = FeatureEngineer('data/cleaned_customer_churn.csv')
    X, y = engineer.prepare_features()
    X_selected = engineer.feature_selection(k=20)
    engineer.save_artifacts()
    engineer.save_processed_data('data/processed_features.csv')

    # Step 3: Model Training
    print("\n\nSTEP 3: MODEL TRAINING")
    print("-" * 40)

    predictor = ChurnPredictor('data/processed_features.csv')
    predictor.split_data()
    results, models = predictor.train_models()
    predictor.evaluate_best_model()
    predictor.save_model()
    predictor.generate_summary_report()

    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETE!")
    print("=" * 60)
    print("\nAll outputs have been saved in their respective folders.")
    print("Check the 'reports/' folder for analysis results.")
    print("Check the 'models/' folder for trained models.")


if __name__ == "__main__":
    # Create necessary directories
    directories = ['data', 'src', 'reports', 'models', 'notebooks']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    main()