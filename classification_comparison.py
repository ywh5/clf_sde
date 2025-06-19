import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import jieba
import os
from tqdm import tqdm

class ClassificationComparison:
    def __init__(self):
        self.datasets = {
            'iris': self._load_iris,
            'breast_cancer': self._load_breast_cancer,
            'thucnews': self._load_thucnews
        }
        self.classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
        self.feature_selectors = {
            'SelectKBest': SelectKBest(f_classif, k=5),
            'RFE': RFE(LogisticRegression(max_iter=1000), n_features_to_select=5),
            'PCA': PCA(n_components=5)
        }
        
    def _load_iris(self):
        data = load_iris()
        return data.data, data.target, data.feature_names
        
    def _load_breast_cancer(self):
        data = load_breast_cancer()
        return data.data, data.target, data.feature_names
        
    def _load_thucnews(self):
        # Note: THUCNews dataset needs to be downloaded and processed separately
        # This is a placeholder for the actual implementation
        raise NotImplementedError("THUCNews dataset loading needs to be implemented")
        
    def preprocess_data(self, X, y, feature_names=None):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def evaluate_classifier(self, clf, X_train, X_test, y_train, y_test):
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'training_time': training_time
        }
        
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            metrics['roc_auc'] = auc(fpr, tpr)
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            
        return metrics
        
    def run_comparison(self, dataset_name):
        print(f"\nProcessing {dataset_name} dataset...")
        X, y, feature_names = self.datasets[dataset_name]()
        
        results = {}
        
        # Original features
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y, feature_names)
        
        print("\nEvaluating classifiers with original features:")
        for clf_name, clf in self.classifiers.items():
            metrics = self.evaluate_classifier(clf, X_train, X_test, y_train, y_test)
            results[f"{clf_name}_original"] = metrics
            print(f"\n{clf_name} Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1']:.4f}")
            print(f"Training Time: {metrics['training_time']:.4f} seconds")
            
        # Feature selection methods
        for selector_name, selector in self.feature_selectors.items():
            print(f"\nEvaluating with {selector_name} feature selection:")
            X_selected = selector.fit_transform(X, y)
            X_train, X_test, y_train, y_test = self.preprocess_data(X_selected, y)
            
            for clf_name, clf in self.classifiers.items():
                metrics = self.evaluate_classifier(clf, X_train, X_test, y_train, y_test)
                results[f"{clf_name}_{selector_name}"] = metrics
                print(f"\n{clf_name} with {selector_name} Results:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1 Score: {metrics['f1']:.4f}")
                print(f"Training Time: {metrics['training_time']:.4f} seconds")
                
        return results
        
    def plot_results(self, results, dataset_name):
        # Create comparison plots
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            values = [results[f"{clf}_{feat}"][metric] 
                     for clf in self.classifiers.keys() 
                     for feat in ['original'] + list(self.feature_selectors.keys())]
            labels = [f"{clf}_{feat}" 
                     for clf in self.classifiers.keys() 
                     for feat in ['original'] + list(self.feature_selectors.keys())]
            
            plt.bar(range(len(values)), values)
            plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
            plt.title(f'{metric.capitalize()} Comparison')
            plt.tight_layout()
            
        plt.savefig(f'{dataset_name}_metrics_comparison.png')
        plt.close()
        
        # Plot ROC curves for methods that support probability estimates
        plt.figure(figsize=(10, 8))
        for clf_name in self.classifiers.keys():
            if f"{clf_name}_original" in results and 'fpr' in results[f"{clf_name}_original"]:
                plt.plot(results[f"{clf_name}_original"]['fpr'],
                        results[f"{clf_name}_original"]['tpr'],
                        label=f'{clf_name} (Original)')
                
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig(f'{dataset_name}_roc_curves.png')
        plt.close()

def main():
    comparison = ClassificationComparison()
    
    # Run comparison for Iris and Breast Cancer datasets
    for dataset in ['iris', 'breast_cancer']:
        results = comparison.run_comparison(dataset)
        comparison.plot_results(results, dataset)
        
    print("\nNote: THUCNews dataset implementation is pending. Please implement the dataset loading function.")

if __name__ == "__main__":
    main() 