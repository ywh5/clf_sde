import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import seaborn as sns

class IrisClassification:
    def __init__(self):
        self.classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
        self.feature_selectors = {
            'SelectKBest': SelectKBest(f_classif, k=2),
            'RFE': RFE(LogisticRegression(max_iter=1000), n_features_to_select=2),
            'PCA': PCA(n_components=2)
        }
        
    def load_data(self):
        data = load_iris()
        return data.data, data.target, data.feature_names
        
    def preprocess_data(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def evaluate_classifier(self, clf, X_train, X_test, y_train, y_test):
        start_time = time.time()
        clf.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test) if hasattr(clf, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'training_time': training_time
        }
        
        if y_pred_proba is not None:
            # 将标签转换为one-hot编码
            n_classes = len(np.unique(y_test))
            y_test_bin = label_binarize(y_test, classes=range(n_classes))
            
            # 计算每个类别的ROC曲线
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # 计算微平均ROC曲线
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            metrics['roc_auc'] = roc_auc
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
            
        return metrics
        
    def run_comparison(self):
        print("\nProcessing Iris dataset...")
        X, y, feature_names = self.load_data()
        results = {}
        
        # Original features
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        
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
        
    def plot_results(self, results):
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
            plt.title(f'Iris Dataset - {metric.capitalize()} Comparison')
            plt.tight_layout()
            
        plt.savefig('iris_metrics_comparison.png')
        plt.close()
        
        # Plot ROC curves for each class and micro-average
        plt.figure(figsize=(10, 8))
        for clf_name in self.classifiers.keys():
            if f"{clf_name}_original" in results and 'fpr' in results[f"{clf_name}_original"]:
                # Plot ROC curve for each class
                for i in range(3):  # Iris has 3 classes
                    plt.plot(results[f"{clf_name}_original"]['fpr'][i],
                            results[f"{clf_name}_original"]['tpr'][i],
                            label=f'{clf_name} (Class {i})')
                
                # Plot micro-average ROC curve
                plt.plot(results[f"{clf_name}_original"]['fpr']["micro"],
                        results[f"{clf_name}_original"]['tpr']["micro"],
                        label=f'{clf_name} (Micro-average)',
                        linestyle='--')
                
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Iris Dataset - ROC Curves (One-vs-Rest)')
        plt.legend(loc="lower right")
        plt.savefig('iris_roc_curves.png')
        plt.close()

def main():
    iris_classifier = IrisClassification()
    results = iris_classifier.run_comparison()
    iris_classifier.plot_results(results)

if __name__ == "__main__":
    main() 