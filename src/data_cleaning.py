import pandas as pd
import numpy as np
import os
from datetime import datetime


class DataCleaner:
    def __init__(self, filepath):
        """
        Initialize DataCleaner with the dataset path
        """
        self.df = pd.read_csv(filepath)
        self.original_shape = self.df.shape
        print(f"Original dataset shape: {self.original_shape}")

    def explore_data(self):
        """
        Initial data exploration and summary
        """
        print("=" * 50)
        print("DATA EXPLORATION SUMMARY")
        print("=" * 50)

        # Basic info
        print("\n1. DATASET INFORMATION:")
        print(f"Total records: {len(self.df)}")
        print(f"Total columns: {len(self.df.columns)}")

        # Data types
        print("\n2. DATA TYPES:")
        print(self.df.dtypes.value_counts())

        # Missing values
        print("\n3. MISSING VALUES:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])

        # Duplicates
        duplicates = self.df.duplicated().sum()
        print(f"\n4. DUPLICATE RECORDS: {duplicates}")

        return self.df

    def clean_data(self):
        """
        Perform data cleaning operations
        """
        print("\n" + "=" * 50)
        print("DATA CLEANING PROCESS")
        print("=" * 50)

        # Make a copy
        df_clean = self.df.copy()

        # 1. Handle Total Charges (should be numeric)
        print("1. Converting 'Total Charges' to numeric...")
        df_clean['Total Charges'] = pd.to_numeric(df_clean['Total Charges'], errors='coerce')

        # 2. Check for numeric columns that should be categorical
        print("2. Converting appropriate columns to categorical...")
        categorical_cols = [
            'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
            'Multiple Lines', 'Internet Service', 'Online Security',
            'Online Backup', 'Device Protection', 'Tech Support',
            'Streaming TV', 'Streaming Movies', 'Contract',
            'Paperless Billing', 'Payment Method', 'Churn Label',
            'Gender'
        ]

        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype('category')

        # 3. Handle missing values in Total Charges
        print("3. Handling missing values in 'Total Charges'...")
        # Fill with median based on tenure and monthly charges
        mask = df_clean['Total Charges'].isnull()
        if mask.any():
            df_clean.loc[mask, 'Total Charges'] = df_clean.loc[mask, 'Monthly Charges'] * df_clean.loc[
                mask, 'Tenure Months']

        # 4. Drop unnecessary columns
        print("4. Removing redundant columns...")
        columns_to_drop = ['Count', 'Country', 'Lat Long', 'Zip Code', 'Latitude', 'Longitude']
        df_clean = df_clean.drop(columns=[col for col in columns_to_drop if col in df_clean.columns])

        # 5. Create additional features
        print("5. Creating new features...")
        # Average monthly charge per tenure
        df_clean['Avg_Monthly_Charge'] = df_clean['Total Charges'] / df_clean['Tenure Months'].replace(0, 1)

        # Tenure groups
        df_clean['Tenure_Group'] = pd.cut(
            df_clean['Tenure Months'],
            bins=[0, 12, 24, 36, 48, 60, 72],
            labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr']
        )

        # 6. Handle categorical columns with too many unique values
        print("6. Checking categorical variables...")
        categorical_stats = {}
        for col in df_clean.select_dtypes(include=['category', 'object']).columns:
            unique_count = df_clean[col].nunique()
            categorical_stats[col] = unique_count
            if unique_count > 20:
                print(f"Warning: {col} has {unique_count} unique values")

        # 7. Save cleaning report
        self.save_cleaning_report(df_clean)

        print(f"\nCleaning complete!")
        print(f"Original shape: {self.original_shape}")
        print(f"Cleaned shape: {df_clean.shape}")
        print(f"Columns removed: {len(self.df.columns) - len(df_clean.columns)}")

        return df_clean

    def save_cleaning_report(self, df_clean):
        """
        Save cleaning report
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_records': self.original_shape[0],
            'original_columns': self.original_shape[1],
            'cleaned_records': df_clean.shape[0],
            'cleaned_columns': df_clean.shape[1],
            'missing_values_after': df_clean.isnull().sum().sum(),
            'duplicates_after': df_clean.duplicated().sum(),
            'data_types': str(dict(df_clean.dtypes.value_counts())),
            'columns_removed': list(set(self.df.columns) - set(df_clean.columns))
        }

        # Create reports directory if it doesn't exist
        os.makedirs('./reports', exist_ok=True)

        with open('./reports/data_cleaning_report.txt', 'w') as f:
            for key, value in report.items():
                f.write(f"{key}: {value}\n")

        print("Cleaning report saved to 'reports/data_cleaning_report.txt'")

    def save_cleaned_data(self, df_clean, output_path):
        """
        Save cleaned data to CSV
        """
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df_clean.to_csv(output_path, index=False)
        print(f"\nCleaned data saved to: {output_path}")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


# Main execution
if __name__ == "__main__":
    # Initialize cleaner
    cleaner = DataCleaner('./data/Telco_customer_churn.csv')

    # Explore data
    df = cleaner.explore_data()

    # Clean data
    df_clean = cleaner.clean_data()

    # Save cleaned data
    cleaner.save_cleaned_data(df_clean, './data/cleaned_customer_churn.csv')