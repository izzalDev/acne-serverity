import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil

train_files = [
    'data/raw/Classification/NNEW_trainval_0.txt',
    'data/raw/Classification/NNEW_trainval_1.txt',
    'data/raw/Classification/NNEW_trainval_2.txt',
    'data/raw/Classification/NNEW_trainval_3.txt',
    'data/raw/Classification/NNEW_trainval_4.txt']

test_files = [
    'data/raw/Classification/NNEW_test_0.txt',
    'data/raw/Classification/NNEW_test_1.txt',
    'data/raw/Classification/NNEW_test_2.txt',
    'data/raw/Classification/NNEW_test_3.txt',
    'data/raw/Classification/NNEW_test_4.txt']

path = 'data/raw/Classification/JPEGImagesv'

li_train = []
for file in train_files:
    train_df = pd.read_csv(train_files[0],names=['path','label','leisons'],sep='  ',engine='python')
    li_train.append(train_df)

x_train = pd.concat(li_train, axis=0, ignore_index=True)

li_test = []
for file in test_files:
    test_df = pd.read_csv(test_files[0],names=['path','label','leisons'],sep='  ',engine='python')
    li_test.append(test_df)
x_test = pd.concat(li_test, axis=0, ignore_index=True)

for label in x_train.label.unique():
    if not os.path.exists(f'data/processed/train/{label}'):
        os.makedirs(f'data/processed/train/{label}')
    if not os.path.exists(f'data/processed/test/{label}'):
        os.makedirs(f'data/processed/test/{label}')

for i in tqdm(x_train.index, desc='Saving train images'):
    src = f'data/raw/Classification/JPEGImages/{x_train.path[i]}'
    dst = f'data/processed/train/{x_train.label[i]}/{x_train.path[i]}'
    shutil.copy(src, dst)

for i in tqdm(x_test.index, desc='Saving test images'):
    src = f'data/raw/Classification/JPEGImages/{x_test.path[i]}'
    dst = f'data/processed/test/{x_test.label[i]}/{x_test.path[i]}'
    shutil.copy(src, dst)
