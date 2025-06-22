import os, math
import pandas as pd
import numpy as np
import fitz
from PIL import Image
from collections import Counter

class DataInspector:
    def __init__(self, path: str):
        self.path = path
        self.type = None
        self.raw = None
        self.report = {}

    def load(self):
        if os.path.isdir(self.path):
            self.type = 'image';  return self._load_images()
        ext = os.path.splitext(self.path)[1].lower()
        if ext in ('.csv','.tsv'):
            self.type = 'table'; return pd.read_csv(self.path)
        if ext in ('.xls','.xlsx'):
            self.type = 'table'; return pd.read_excel(self.path)
        if ext == '.pdf':
            self.type = 'text'
            doc = fitz.open(self.path)
            return "".join(p.get_text() for p in doc)
        raise ValueError(f"Unsupported: {ext}")

    def analyze_and_repair(self):
        data = self.load()
        if self.type == 'table':
            df = data.copy()
            # Mixed‐type detector & repair
            mask = df.applymap(
                lambda x: isinstance(x,str) and any(c.isalpha() for c in x) and any(c.isdigit() for c in x)
            )
            self.report['mixed_cells'] = int(mask.sum().sum())
            # convert & fill
            for col in df.columns:
                if df[col].dtype == object:
                    num = pd.to_numeric(df[col], errors='coerce')
                    df[col] = np.where(num.isna(), num, num).fillna(df[col].median() if num.isna().mean()<0.3 else 0)
            self.report['shape'] = df.shape
            self.raw = df

        elif self.type == 'image':
            imgs, labels = [], []
            for lbl in os.listdir(self.path):
                p = os.path.join(self.path, lbl)
                if os.path.isdir(p):
                    for f in os.listdir(p):
                        if f.lower().endswith(('jpg','png')):
                            imgs.append(Image.open(os.path.join(p,f)))
                            labels.append(lbl)
            self.report['n_images'] = len(imgs)
            self.raw = (imgs, labels)

        else:  # text
            txt = data
            words = txt.split()
            self.report['word_count'] = len(words)
            self.raw = txt

        return self.raw, self.report
