import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score, precision_recall_curve,
    f1_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime


class ChurnPredictor:
    def __init__(self, features_path, target_col='Churn Value'):
        """
        Initialize ChurnPredictor
        """
        data = pd.read_csv(features_path)

        if target_col in data.columns:
            self.X = data.drop(target_col, axis=1)
            self.y = data[target_col]
        else:
            # Assume last column is target
            self.X = data.iloc[:, :-1]
            self.y = data.iloc[:, -1]

        print(f"Dataset shape: {data.shape}")
        print(f"Features: {self.X.shape[1]}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        print(f"Churn rate: {self.y.mean():.2%}")

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )

        print(f"\nTrain set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Train churn rate: {self.y_train.mean():.2%}")
        print(f"Test churn rate: {self.y_test.mean():.2%}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def handle_imbalance(self):
        """
        Handle class imbalance using SMOTE
        """
        print("\nApplying SMOTE to handle class imbalance...")
        smote = SMOTE(random_state=42)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)

        print(f"After SMOTE - Train set: {self.X_train_resampled.shape}")
        print(f"Class distribution after SMOTE:\n{pd.Series(self.y_train_resampled).value_counts()}")

        return self.X_train_resampled, self.y_train_resampled

    def train_models(self):
        """
        Train multiple models and compare performance
        """
        print("\n" + "=" * 50)
        print("MODEL TRAINING")
        print("=" * 50)

        # Define models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }

        # Hyperparameter grids
        param_grids = {
            'Logistic Regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l2']
            },
            'Random Forest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear']
            }
        }

        results = {}
        best_models = {}

        for name, model in models.items():
            print(f"\nTraining {name}...")

            # Create pipeline with SMOTE
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=42)),
                ('classifier', model)
            ])

            # Grid search
            grid_search = GridSearchCV(
                pipeline,
                {'classifier__' + key: value for key, value in param_grids[name].items()},
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(self.X_train, self.y_train)

            # Best model
            best_model = grid_search.best_estimator_
            best_models[name] = best_model

            # Predictions
            y_pred = best_model.predict(self.X_test)
            y_pred_proba = best_model.predict_proba(self.X_test)[:, 1]

            # Metrics
            results[name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'accuracy': accuracy_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'f1_score': f1_score(self.y_test, y_pred)
            }

            print(f"Best params: {grid_search.best_params_}")
            print(f"CV Score: {grid_search.best_score_:.4f}")
            print(f"Test Accuracy: {results[name]['accuracy']:.4f}")
            print(f"Test ROC-AUC: {results[name]['roc_auc']:.4f}")

        # Compare models
        self.compare_models(results)
        self.best_models = best_models

        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        self.best_model = best_models[best_model_name]

        print(f"\nBest model: {best_model_name}")
        print(f"ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")

        return results, best_models

    def compare_models(self, results):
        """
        Compare model performance
        """
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)

        print("\n" + "=" * 50)
        print("MODEL COMPARISON")
        print("=" * 50)
        print(comparison_df[['accuracy', 'roc_auc', 'f1_score']])

        # Save comparison
        os.makedirs('./reports', exist_ok=True)
        comparison_df.to_csv('./reports/model_comparison.csv')

        # Plot comparison
        plt.figure(figsize=(12, 6))

        metrics = ['accuracy', 'roc_auc', 'f1_score']
        x = np.arange(len(comparison_df.index))
        width = 0.25

        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, comparison_df[metric], width, label=metric)

        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, comparison_df.index, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./reports/model_comparison.png', dpi=300)
        plt.close()

        print("\nModel comparison saved to 'reports/' folder")

    def evaluate_best_model(self):
        """
        Detailed evaluation of the best model
        """
        print("\n" + "=" * 50)
        print("DETAILED EVALUATION OF BEST MODEL")
        print("=" * 50)

        # Predictions
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('./reports/confusion_matrix.png', dpi=300)
        plt.close()

        # ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('./reports/roc_curve.png', dpi=300)
        plt.close()

        # Feature importance (if available)
        if hasattr(self.best_model.named_steps['classifier'], 'feature_importances_'):
            feature_importance = self.best_model.named_steps['classifier'].feature_importances_
            feature_names = self.X.columns

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False).head(15)

            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Top 15 Feature Importance')
            plt.tight_layout()
            plt.savefig('./reports/feature_importance_top15.png', dpi=300)
            plt.close()

            # Save feature importance
            importance_df.to_csv('./reports/feature_importance_detailed.csv', index=False)

    def save_model(self):
        """
        Save the best model and evaluation results
        """
        os.makedirs('./models', exist_ok=True)

        # Save model
        model_path = './models/best_churn_model.pkl'
        joblib.dump(self.best_model, model_path)

        # Save model metadata
        metadata = {
            'model_type': type(self.best_model.named_steps['classifier']).__name__,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_count': self.X.shape[1],
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'churn_rate': float(self.y.mean())
        }

        with open('./models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)

        # Save predictions for analysis
        predictions = pd.DataFrame({
            'actual': self.y_test.values,
            'predicted': self.best_model.predict(self.X_test),
            'probability': self.best_model.predict_proba(self.X_test)[:, 1]
        })
        predictions.to_csv('./reports/model_predictions.csv', index=False)

        print(f"\nBest model saved to: {model_path}")
        print(f"Model metadata saved to: ./models/model_metadata.json")
        print(f"Predictions saved to: ./reports/model_predictions.csv")

    def generate_summary_report(self):
        """
        Generate a summary report of the modeling process
        """
        report = f"""
        CHURN PREDICTION MODELING REPORT
        =================================

        Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        1. DATASET OVERVIEW:
           - Total samples: {len(self.X)}
           - Features: {self.X.shape[1]}
           - Churn rate: {self.y.mean():.2%}
           - Non-churn: {(self.y == 0).sum()}
           - Churn: {(self.y == 1).sum()}

        2. DATA SPLIT:
           - Training set: {len(self.X_train)} samples ({len(self.X_train) / len(self.X):.1%})
           - Test set: {len(self.X_test)} samples ({len(self.X_test) / len(self.X):.1%})

        3. BEST MODEL:
           - Model type: {type(self.best_model.named_steps['classifier']).__name__}
           - Best parameters: {self.best_model.named_steps['classifier'].get_params()}

        4. MODEL PERFORMANCE ON TEST SET:
        """

        # Add performance metrics
        y_pred = self.best_model.predict(self.X_test)
        y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        report += f"""
           - Accuracy: {accuracy_score(self.y_test, y_pred):.4f}
           - ROC-AUC: {roc_auc_score(self.y_test, y_pred_proba):.4f}
           - F1-Score: {f1_score(self.y_test, y_pred):.4f}

        5. BUSINESS IMPLICATIONS:
           - Based on the model, key drivers of churn have been identified
           - Feature importance plot shows the most influential factors
           - Recommendations for customer retention strategies can be derived

        6. FILES GENERATED:
           - Best model: models/best_churn_model.pkl
           - Model metadata: models/model_metadata.json
           - Model comparison: reports/model_comparison.csv
           - Feature importance: reports/feature_importance_detailed.csv
           - Visualizations: Various .png files in reports/ folder
        """

        # Save report
        with open('./reports/modeling_summary.txt', 'w') as f:
            f.write(report)

        print("\nSummary report saved to: reports/modeling_summary.txt")


# Main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor('./data/processed_features.csv')

    # Split data
    predictor.split_data()

    # Train models
    results, models = predictor.train_models()

    # Evaluate best model
    predictor.evaluate_best_model()

    # Save model and results
    predictor.save_model()

    # Generate summary report
    predictor.generate_summary_report()