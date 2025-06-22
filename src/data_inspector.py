import os, math
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
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
            self.type = 'image'
            return self._load_images()
        ext = os.path.splitext(self.path)[1].lower()
        if ext in ('.csv', '.tsv'):
            self.type = 'table'
            return pd.read_csv(self.path)
        if ext in ('.xls', '.xlsx'):
            self.type = 'table'
            return pd.read_excel(self.path)
        if ext == '.pdf':
            self.type = 'text'
            doc = fitz.open(self.path)
            return "".join(p.get_text() for p in doc)
        raise ValueError(f"Unsupported: {ext}")

    def _load_images(self):
        imgs, labels = [], []
        label_counts = Counter()
        for lbl in os.listdir(self.path):
            p = os.path.join(self.path, lbl)
            if os.path.isdir(p):
                for f in os.listdir(p):
                    if f.lower().endswith(('jpg','png','jpeg','bmp')):
                        img = Image.open(os.path.join(p, f))
                        imgs.append(img)
                        labels.append(lbl)
                        label_counts[lbl] += 1
        self.report['image_counts'] = dict(label_counts)
        self.report['class_imbalance'] = self._calculate_imbalance(label_counts)
        return imgs, labels

    def _calculate_imbalance(self, counts):
        total = sum(counts.values())
        probs = [c/total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs)
        max_entropy = math.log2(len(counts)) if counts else 1
        return (max_entropy - entropy) / max_entropy

    def analyze_and_repair(self):
        data = self.load()
        if self.type == 'table':
            df = data.copy()
            mask = df.applymap(
                lambda x: isinstance(x, str) and any(c.isalpha() for c in x) and any(c.isdigit() for c in x)
            )
            self.report['mixed_cells'] = int(mask.sum().sum())
            for col in df.columns:
                if df[col].dtype == object:
                    num = pd.to_numeric(df[col], errors='coerce')
                    if num.isna().mean() < 0.3:
                        df[col] = num.fillna(num.median())
                    else:
                        # categorical
                        df[col] = df[col].astype('category').cat.codes
            self.report['shape'] = df.shape
            self.raw = df

        elif self.type == 'image':
            imgs, labels = data
            widths, heights = zip(*(img.size for img in imgs))
            self.report['avg_width'] = np.mean(widths)
            self.report['avg_height'] = np.mean(heights)
            self.raw = imgs, labels

        else:  # text
            txt = data
            words = txt.split()
            wc = len(words)
            vocab = len(set(words))
            entropy = -sum((c/wc)*math.log2(c/wc) for c in Counter(words).values()) if wc else 0
            self.report.update({
                'word_count': wc,
                'vocab_size': vocab,
                'entropy': entropy
            })
            self.raw = txt

        return self.raw, self.report
