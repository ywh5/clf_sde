# import os
# import zipfile
# import requests
# from tqdm import tqdm
#
#
# def download_thucnews(save_dir='thucnews', unzip=True):
#     os.makedirs(save_dir, exist_ok=True)
#     zip_path = os.path.join(save_dir, 'THUCNews.zip')
#
#     # 官方 GitHub 发布的下载链接（大约 129MB）
#     url = "https://github.com/gaussic/text-classification-cnn-rnn/raw/master/data/THUCNews.zip"
#
#     if not os.path.exists(zip_path):
#         print("Downloading THUCNews.zip...")
#         response = requests.get(url, stream=True)
#         total_size = int(response.headers.get('content-length', 0))
#         with open(zip_path, 'wb') as f, tqdm(
#                 desc="Downloading",
#                 total=total_size,
#                 unit='B',
#                 unit_scale=True,
#                 unit_divisor=1024,
#         ) as bar:
#             for data in response.iter_content(chunk_size=1024):
#                 f.write(data)
#                 bar.update(len(data))
#         print("Download complete.")
#
#     if unzip:
#         print("Unzipping...")
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall(save_dir)
#         print(f"Unzipped to {save_dir}")
#     else:
#         print(f"Saved zip to: {zip_path}")
#
#
# if __name__ == "__main__":
#     download_thucnews(save_dir='./thucnews')

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cnews_loader
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from pprint import pprint
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from cnews_loader import *
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

# 设置数据读取、模型、结果保存路径
base_dir = './input'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')


train_contents, train_labels = read_file(train_dir)
test_contents, test_labels = read_file(test_dir)

