import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from tqdm import tqdm

class THUCNewsProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.categories = [
            '体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐'
        ]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        
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
                            texts.append(text)
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

def main():
    # Example usage
    processor = THUCNewsProcessor("path/to/thucnews/dataset")
    X, y, feature_names = processor.load_data(max_samples_per_category=1000)
    print(f"Loaded {len(X)} samples with {len(feature_names)} features")
    print(f"Number of categories: {len(np.unique(y))}")

if __name__ == "__main__":
    main() 