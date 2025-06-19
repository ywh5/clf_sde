import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class THUCNewsClassification:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.categories = [
            '体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐'
        ]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        
        self.classifiers = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100)
        }
        
        self.feature_selectors = {
            'SelectKBest': SelectKBest(f_classif, k=1000),
            'RFE': RFE(LogisticRegression(max_iter=1000), n_features_to_select=1000),
            'PCA': PCA(n_components=1000)
        }
        
    def load_data(self, max_samples_per_category=1000):
        texts = []
        labels = []
        
        print("Loading THUCNews dataset...")
        for category in tqdm(self.categories):
            category_dir = os.path.join(self.data_dir, category)
            if not os.path.exists(category_dir):
                print(f"Warning: Directory {category_dir} does not exist")
                continue
                
            files = os.listdir(category_dir)[:max_samples_per_category]
            for file in files:
                file_path = os.path.join(category_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if text:  # Only add non-empty texts
                            texts.append(self.preprocess_text(text))
                            labels.append(category)
                except Exception as e:
                    print(f"Error reading file {file_path}: {str(e)}")
                    
        # Convert texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        
        return X.toarray(), y, self.vectorizer.get_feature_names_out()
        
    def preprocess_text(self, text):
        # Remove special characters and numbers
        text = ''.join([char for char in text if char.isalpha() or char.isspace()])
        # Tokenize using jieba
        words = jieba.cut(text)
        # Join words back into a string
        return ' '.join(words)
        
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
        
    def run_comparison(self):
        print("\nProcessing THUCNews dataset...")
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
            plt.title(f'THUCNews Dataset - {metric.capitalize()} Comparison')
            plt.tight_layout()
            
        plt.savefig('thucnews_metrics_comparison.png')
        plt.close()
        
        # Plot ROC curves
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
        plt.title('THUCNews Dataset - ROC Curves')
        plt.legend(loc="lower right")
        plt.savefig('thucnews_roc_curves.png')
        plt.close()

def main():
    # 请替换为实际的THUCNews数据集路径
    data_dir = "path/to/thucnews/dataset"
    thucnews_classifier = THUCNewsClassification(data_dir)
    results = thucnews_classifier.run_comparison()
    thucnews_classifier.plot_results(results)

if __name__ == "__main__":
    main() 