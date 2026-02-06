import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os


class FeatureEngineer:
    def __init__(self, data_path):
        """
        Initialize FeatureEngineer with cleaned data
        """
        self.df = pd.read_csv(data_path)
        self.features = None
        self.target = None
        self.encoders = {}
        self.scaler = StandardScaler()

    def prepare_features(self):
        """
        Prepare features for modeling
        """
        print("=" * 50)
        print("FEATURE ENGINEERING")
        print("=" * 50)

        df_encoded = self.df.copy()

        # 1. Separate target variable
        if 'Churn Value' in df_encoded.columns:
            self.target = df_encoded['Churn Value']
            df_encoded = df_encoded.drop('Churn Value', axis=1)

        # 2. Drop identifier columns
        id_cols = ['CustomerID', 'Churn Label', 'Churn Reason', 'Churn Score']
        df_encoded = df_encoded.drop(columns=[col for col in id_cols if col in df_encoded.columns])

        # 3. Handle categorical variables
        print("1. Encoding categorical variables...")
        categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            # For binary categorical variables
            if df_encoded[col].nunique() == 2:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
                self.encoders[col] = le
            # For multi-categorical variables (one-hot encoding)
            else:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)

        # 4. Handle numerical variables
        print("2. Scaling numerical variables...")
        numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns

        # Save column names before scaling
        self.feature_columns = df_encoded.columns.tolist()

        # Scale numerical features
        df_encoded[numerical_cols] = self.scaler.fit_transform(df_encoded[numerical_cols])

        self.features = df_encoded

        print(f"Total features created: {len(self.feature_columns)}")
        print(f"Feature shape: {self.features.shape}")

        return self.features, self.target

    def feature_selection(self, k=20):
        """
        Select top k features using ANOVA F-value
        """
        print(f"\n3. Selecting top {k} features...")

        # Use SelectKBest with ANOVA F-value
        selector = SelectKBest(score_func=f_classif, k=min(k, self.features.shape[1]))
        X_selected = selector.fit_transform(self.features, self.target)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = self.features.columns[selected_mask].tolist()

        print(f"Selected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features, 1):
            print(f"{i}. {feature}")

        # Create DataFrame with selected features
        self.features_selected = pd.DataFrame(X_selected, columns=selected_features)

        # Save feature scores
        feature_scores = pd.DataFrame({
            'Feature': self.features.columns,
            'Score': selector.scores_,
            'P-value': selector.pvalues_
        }).sort_values('Score', ascending=False)

        self.save_feature_importance(feature_scores)

        return self.features_selected

    def save_feature_importance(self, feature_scores):
        """
        Save feature importance report
        """
        os.makedirs('./reports', exist_ok=True)

        feature_scores.to_csv('./reports/feature_importance.csv', index=False)
        print("\nFeature importance scores saved to 'reports/feature_importance.csv'")

    def save_artifacts(self):
        """
        Save encoders and scaler for future use
        """
        os.makedirs('./models', exist_ok=True)

        # Save scaler
        joblib.dump(self.scaler, './models/scaler.pkl')

        # Save encoders if any
        if self.encoders:
            joblib.dump(self.encoders, './models/encoders.pkl')

        # Save feature columns
        joblib.dump(self.feature_columns, './models/feature_columns.pkl')

        print("\nFeature engineering artifacts saved to 'models/' folder")

    def save_processed_data(self, output_path):
        """
        Save processed features and target
        """
        # Create processed data
        processed_data = pd.concat([self.features_selected, self.target.reset_index(drop=True)], axis=1)

        # Save to CSV
        processed_data.to_csv(output_path, index=False)
        print(f"\nProcessed data saved to: {output_path}")


# Main execution
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = FeatureEngineer('./data/cleaned_customer_churn.csv')

    # Prepare features
    X, y = engineer.prepare_features()

    # Select features
    X_selected = engineer.feature_selection(k=20)

    # Save artifacts
    engineer.save_artifacts()

    # Save processed data
    engineer.save_processed_data('./data/processed_features.csv')