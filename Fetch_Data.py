import pandas as pd
from datasets import load_dataset

dataset = load_dataset("papluca/language-identification")

dataset['train'].to_pandas().to_csv("data/lang_train.csv", index=False)
dataset['validation'].to_pandas().to_csv("data/lang_val.csv", index=False)
dataset['test'].to_pandas().to_csv("data/lang_test.csv", index=False)

print("Data is ready")