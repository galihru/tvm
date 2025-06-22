import pandas as pd
import numpy as np
import os
import fitz
from PIL import Image

class DataInspector:
    """
    Inspect various dataset types: tabular (CSV/Excel), image folders, PDF text.
    Repair mixed entries and invalid data.
    """
    def __init__(self, path: str):
        self.path = path
        self.data = None
        self.type = None
        self.report = {}

    def load(self):
        if os.path.isdir(self.path):
            # image folder
            self.type = 'image'
            self.data = [os.path.join(self.path, f) for f in os.listdir(self.path)
                         if f.lower().endswith(('jpg','png','jpeg','bmp'))]
        else:
            ext = os.path.splitext(self.path)[1].lower()
            if ext in ['.csv', '.tsv']:
                self.type = 'table'
                self.data = pd.read_csv(self.path)
            elif ext in ['.xls', '.xlsx']:
                self.type = 'table'
                self.data = pd.read_excel(self.path)
            elif ext == '.pdf':
                self.type = 'pdf'
                doc = fitz.open(self.path)
                text = "".join(page.get_text() for page in doc)
                self.data = text
            else:
                raise ValueError(f"Unsupported format: {ext}")
        return self.data

    def analyze_and_repair(self):
        if self.type == 'table':
            df = self.data
            # basic report
            self.report['shape'] = df.shape
            # detect mixed-type cells
            mask = df.applymap(lambda x: isinstance(x, str) and any(c.isalpha() for c in x) and any(c.isdigit() for c in x))
            # counts
            self.report['mixed_cells'] = int(mask.sum().sum())
            # repair: non-numeric->NaN->mean
            for col in df.select_dtypes(include='object'):
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.fillna(df.mean(numeric_only=True), inplace=True)
            self.data = df
        elif self.type == 'image':
            # report number of images and average dimensions
            dims = []
            for f in self.data:
                with Image.open(f) as img:
                    dims.append(img.size)
            self.report['n_images'] = len(dims)
            w,h = zip(*dims)
            self.report['avg_width'] = np.mean(w)
            self.report['avg_height'] = np.mean(h)
        elif self.type == 'pdf':
            self.report['n_chars'] = len(self.data)
        return self.report
