import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    precision_recall_curve, 
    roc_curve, 
    roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

class CreditCardFraudDetector:
    def __init__(self, dataset_path):
        """
        Initialize the fraud detection project
        
        Args:
            dataset_path (str): Path to the credit card transactions dataset
        """
        self.dataset = pd.read_csv(dataset_path)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
    
    def explore_dataset(self):
        """
        Perform initial data exploration and visualization
        """
        print("Dataset Information:")
        print(self.dataset.info())
        
        print("\nClass Distribution:")
        print(self.dataset['Class'].value_counts(normalize=True))
        
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Class', data=self.dataset)
        plt.title('Distribution of Fraudulent vs. Non-Fraudulent Transactions')
        plt.show()
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the dataset for machine learning
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Controls the shuffling applied to the data
        """
        # Separate features and target
        self.X = self.dataset.drop('Class', axis=1)
        self.y = self.dataset['Class']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=self.y
        )
        
        # Scale the features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
    
    def train_models(self):
        """
        Train multiple machine learning models
        """
        # Logistic Regression
        lr = LogisticRegression(class_weight='balanced')
        lr.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = lr
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100, 
            class_weight='balanced', 
            random_state=42
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
    
    def evaluate_models(self):
        """
        Evaluate trained models using various metrics
        """
        for name, model in self.models.items():
            print(f"\n{name} Model Evaluation:")
            y_pred = model.predict(self.X_test)
            
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.show()
            
            # ROC Curve
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Receiver Operating Characteristic - {name}')
            plt.legend(loc="lower right")
            plt.show()
    
    def detect_fraud(self, transaction_data):
        """
        Predict fraud probability for new transactions
        
        Args:
            transaction_data (array-like): Features of new transaction
        
        Returns:
            dict: Fraud probabilities from different models
        """
        results = {}
        for name, model in self.models.items():
            fraud_prob = model.predict_proba(transaction_data.reshape(1, -1))[0][1]
            results[name] = fraud_prob
        return results

def main():
    # Replace with your actual dataset path
    dataset_path = 'credit_card_transactions.csv'
    
    # Initialize and run the fraud detection project
    fraud_detector = CreditCardFraudDetector(dataset_path)
    
    # Explore the dataset
    fraud_detector.explore_dataset()
    
    # Preprocess data
    fraud_detector.preprocess_data()
    
    # Train models
    fraud_detector.train_models()
    
    # Evaluate models
    fraud_detector.evaluate_models()
    
    # Example of fraud detection for a new transaction
    # Note: Replace with actual transaction features
    new_transaction = np.random.rand(29)  # Assuming 29 features
    fraud_probabilities = fraud_detector.detect_fraud(new_transaction)
    print("\nFraud Detection for New Transaction:")
    for model, prob in fraud_probabilities.items():
        print(f"{model}: {prob * 100:.2f}% probability of fraud")

if __name__ == "__main__":
    main()
