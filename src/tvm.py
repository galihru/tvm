import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal, GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping
import logging
import re
from PIL import Image
import PyPDF2
import textract

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TVMAdvisor")

class DatasetAnalyzer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.data_type = self.detect_data_type()
        self.analysis_report = {}
        self.load_data()
        self.clean_data()

    def detect_data_type(self):
        """Deteksi otomatis tipe data berdasarkan ekstensi file"""
        if os.path.isdir(self.data_path):
            return 'image_folder'
        elif self.data_path.lower().endswith(('.csv')):
            return 'csv'
        elif self.data_path.lower().endswith(('.xls', '.xlsx')):
            return 'excel'
        elif self.data_path.lower().endswith(('.pdf')):
            return 'pdf'
        elif self.data_path.lower().endswith(('.txt', '.doc', '.docx')):
            return 'text'
        else:
            raise ValueError("Format file tidak didukung")

    def load_data(self):
        """Memuat data berdasarkan tipe file"""
        logger.info(f"Memuat data dari {self.data_path} (Tipe: {self.data_type})")
        
        if self.data_type == 'csv':
            self.data = pd.read_csv(self.data_path)
            logger.info(f"Data CSV dimuat. {len(self.data)} baris, {len(self.data.columns)} kolom")
            
        elif self.data_type == 'excel':
            self.data = pd.read_excel(self.data_path)
            logger.info(f"Data Excel dimuat. {len(self.data)} baris")
            
        elif self.data_type == 'image_folder':
            self.data = []
            self.labels = []
            for label in os.listdir(self.data_path):
                label_path = os.path.join(self.data_path, label)
                if os.path.isdir(label_path):
                    for file in os.listdir(label_path):
                        if file.lower().endswith(('png', 'jpg', 'jpeg')):
                            img = Image.open(os.path.join(label_path, file))
                            self.data.append(np.array(img))
                            self.labels.append(label)
            self.data = np.array(self.data)
            self.labels = np.array(self.labels)
            logger.info(f"Data gambar dimuat. Bentuk: {self.data.shape}, Jumlah kelas: {len(np.unique(self.labels))}")
            
        elif self.data_type in ['text', 'pdf']:
            if self.data_type == 'pdf':
                text = textract.process(self.data_path).decode('utf-8')
            else:
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            self.data = text.split('\n')  # Split menjadi baris
            logger.info(f"Data teks dimuat. {len(self.data)} baris")

    def clean_data(self):
        """Pembersihan data dan perbaikan nilai tidak valid"""
        if isinstance(self.data, pd.DataFrame):
            # Deteksi kolom dengan masalah tipe data
            for col in self.data.columns:
                # Coba konversi ke numerik, jika gagal konversi ke kategorikal
                original_type = str(self.data[col].dtype)
                converted = pd.to_numeric(self.data[col], errors='coerce')
                
                if converted.isna().mean() > 0.3:  # >30% nilai invalid
                    # Pertahankan sebagai string
                    self.data[col] = self.data[col].astype(str)
                    logger.warning(f"Kolom {col} mengandung banyak nilai non-numerik. Dipertahankan sebagai string.")
                else:
                    # Konversi ke numerik dan isi missing value
                    self.data[col] = converted
                    if self.data[col].isna().any():
                        median_val = self.data[col].median()
                        self.data[col].fillna(median_val, inplace=True)
                        logger.info(f"Kolom {col}: {self.data[col].isna().sum()} nilai NaN diganti dengan median {median_val:.2f}")
            
            # Analisis distribusi
            self.analysis_report['distributions'] = {}
            numeric_cols = self.data.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                self.analysis_report['distributions'][col] = {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'skewness': self.data[col].skew()
                }

    def analyze_dataset(self):
        """Analisis komprehensif dataset"""
        analysis = {
            "samples": 0,
            "features": 0,
            "classes": 0,
            "feature_types": {},
            "issues": []
        }
        
        if isinstance(self.data, pd.DataFrame):
            analysis["samples"] = len(self.data)
            analysis["features"] = len(self.data.columns)
            
            # Analisis tipe fitur
            for col in self.data.columns:
                dtype = str(self.data[col].dtype)
                if dtype.startswith('int') or dtype.startswith('float'):
                    analysis["feature_types"][col] = 'numerical'
                else:
                    analysis["feature_types"][col] = 'categorical'
                    analysis["classes"] = len(self.data[col].unique())
            
            # Deteksi masalah data
            for col in self.data.columns:
                null_count = self.data[col].isnull().sum()
                if null_count > 0:
                    analysis["issues"].append(f"Kolom {col} memiliki {null_count} nilai kosong")
                
                if analysis["feature_types"][col] == 'numerical':
                    z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                    outlier_count = (z_scores > 3).sum()
                    if outlier_count > 0:
                        analysis["issues"].append(f"Kolom {col} memiliki {outlier_count} outlier")
        
        elif isinstance(self.data, np.ndarray) and self.data_type == 'image_folder':
            analysis["samples"] = self.data.shape[0]
            analysis["features"] = np.prod(self.data.shape[1:])
            analysis["classes"] = len(np.unique(self.labels))
            
            # Analisis distribusi kelas
            class_counts = {label: np.sum(self.labels == label) for label in np.unique(self.labels)}
            imbalance_ratio = max(class_counts.values()) / min(class_counts.values())
            if imbalance_ratio > 5:
                analysis["issues"].append(f"Ketidakseimbangan kelas terdeteksi (rasio: {imbalance_ratio:.1f}x)")
        
        logger.info("\n=== Laporan Analisis Dataset ===")
        logger.info(f"Jumlah sampel: {analysis['samples']}")
        logger.info(f"Jumlah fitur: {analysis['features']}")
        logger.info(f"Jumlah kelas: {analysis['classes']}")
        logger.info(f"Tipe fitur: {analysis['feature_types']}")
        
        if analysis["issues"]:
            logger.warning("Masalah terdeteksi:")
            for issue in analysis["issues"]:
                logger.warning(f" - {issue}")
        else:
            logger.info("Tidak ditemukan masalah signifikan")
        
        self.analysis_report = analysis
        return analysis

class TVMConfigAdvisor:
    def __init__(self, analysis_report):
        self.analysis = analysis_report
    
    def suggest_split_ratio(self):
        """Saran pembagian data berdasarkan penelitian (arXiv:2002.11328)"""
        n_samples = self.analysis["samples"]
        
        if n_samples < 1000:
            return 0.70  # 70% training
        elif 1000 <= n_samples < 10000:
            return 0.75  # 75% training
        elif 10000 <= n_samples < 100000:
            return 0.80  # 80% training
        else:
            return 0.85  # 85% training
    
    def suggest_learning_rate(self):
        """Saran learning rate berdasarkan penelitian (arXiv:1506.01186)"""
        n_features = self.analysis["features"]
        
        if n_features < 50:
            return 0.01
        elif 50 <= n_features < 200:
            return 0.005
        elif 200 <= n_features < 1000:
            return 0.001
        else:
            return 0.0005
    
    def suggest_weight_init(self):
        """Saran inisialisasi bobot berdasarkan penelitian (arXiv:1502.01852)"""
        has_categorical = any(t == 'categorical' for t in self.analysis["feature_types"].values())
        return "He Normal" if has_categorical else "Glorot Uniform"
    
    def suggest_architecture(self):
        """Saran arsitektur berdasarkan tipe data"""
        if "image" in self.analysis.get("data_type", ""):
            return "CNN"
        elif "text" in self.analysis.get("data_type", ""):
            return "LSTM"
        else:
            return "Dense Network"
    
    def generate_config_report(self):
        """Hasilkan laporan konfigurasi lengkap"""
        config = {
            "split_ratio": self.suggest_split_ratio(),
            "learning_rate": self.suggest_learning_rate(),
            "weight_init": self.suggest_weight_init(),
            "architecture": self.suggest_architecture(),
            "epochs": self.suggest_epochs()
        }
        
        logger.info("\n=== Rekomendasi Konfigurasi ===")
        logger.info(f"Arsitektur: {config['architecture']}")
        logger.info(f"Pembagian Data: Training {config['split_ratio']*100:.0f}%, Validasi {100-config['split_ratio']*100:.0f}%")
        logger.info(f"Learning Rate: {config['learning_rate']}")
        logger.info(f"Inisialisasi Bobot: {config['weight_init']}")
        logger.info(f"Epochs: {config['epochs']}")
        
        return config
    
    def suggest_epochs(self):
        """Saran jumlah epochs berdasarkan kompleksitas data"""
        complexity = self.analysis['samples'] * self.analysis['features'] / 1000
        return max(10, min(100, int(complexity ** 0.5)))

class TVMModelTrainer:
    def __init__(self, data, config):
        self.data = data
        self.config = config
        self.model = None
        self.history = None
    
    def prepare_data(self):
        """Persiapkan data berdasarkan tipe"""
        if isinstance(self.data, pd.DataFrame):
            # Pisahkan fitur dan label
            X = self.data.iloc[:, :-1].values
            y = self.data.iloc[:, -1].values
            
            # Encode label jika kategorikal
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            # Normalisasi
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
            # Pembagian data
            return train_test_split(
                X, y, 
                train_size=self.config['split_ratio'],
                stratify=y,
                random_state=42
            )
        
        elif isinstance(self.data, np.ndarray):  # Data gambar
            X = self.data / 255.0  # Normalisasi
            y = self.labels
            
            # Encode label
            le = LabelEncoder()
            y = le.fit_transform(y)
            
            return train_test_split(
                X, y, 
                train_size=self.config['split_ratio'],
                stratify=y,
                random_state=42
            )
        
        return None

    def build_model(self, input_shape, n_classes):
        """Bangun model berdasarkan konfigurasi"""
        initializer = HeNormal() if "He" in self.config['weight_init'] else GlorotUniform()
        
        model = Sequential()
        
        if self.config['architecture'] == "CNN":
            model.add(Conv2D(32, (3,3), activation='relu', 
                          input_shape=input_shape,
                          kernel_initializer=initializer))
            model.add(MaxPooling2D((2,2)))
            model.add(Conv2D(64, (3,3), activation='relu',
                          kernel_initializer=initializer))
            model.add(MaxPooling2D((2,2)))
            model.add(Flatten())
        elif self.config['architecture'] == "LSTM":
            model.add(LSTM(128, input_shape=input_shape,
                        kernel_initializer=initializer))
        else:  # Dense Network
            model.add(Dense(128, activation='relu', 
                         input_dim=input_shape[0],
                         kernel_initializer=initializer))
            model.add(Dense(64, activation='relu',
                         kernel_initializer=initializer))
        
        model.add(Dense(n_classes, activation='softmax'))
        
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self):
        """Latih model dengan konfigurasi yang diberikan"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        if X_train.ndim > 2:  # Data gambar
            input_shape = X_train.shape[1:]
        else:
            input_shape = (X_train.shape[1],)
        
        n_classes = len(np.unique(y_train))
        
        self.model = self.build_model(input_shape, n_classes)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=self.config['epochs'],
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )
        
        return self.history

    def evaluate_model(self):
        """Evaluasi model dan hasilkan metrik"""
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, np.argmax(train_pred, axis=1))
        test_acc = accuracy_score(y_test, np.argmax(test_pred, axis=1))
        
        # Untuk regresi
        try:
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
        except:
            train_r2, test_r2 = None, None
        
        return {
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "final_epoch": len(self.history.history['loss'])
        }

    def plot_training_history(self):
        """Visualisasi proses training"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss Metrics')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy Metrics')
        plt.xlabel('Epochs')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
    def plot_feature_importance(self):
        """Visualisasi pentingnya fitur (untuk data tabular)"""
        if not isinstance(self.data, pd.DataFrame):
            return
            
        importance = np.abs(self.model.layers[0].get_weights()[0]).mean(axis=1)
        features = self.data.columns[:-1]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=features, palette="viridis")
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

def main(data_path):
    # Langkah 1: Analisis dataset
    analyzer = DatasetAnalyzer(data_path)
    analysis_report = analyzer.analyze_dataset()
    
    # Langkah 2: Dapatkan saran konfigurasi
    advisor = TVMConfigAdvisor(analysis_report)
    config = advisor.generate_config_report()
    
    # Langkah 3: Training model
    trainer = TVMModelTrainer(analyzer.data, config)
    history = trainer.train_model()
    
    # Langkah 4: Evaluasi
    metrics = trainer.evaluate_model()
    logger.info("\n=== Hasil Evaluasi ===")
    logger.info(f"Akurasi Training: {metrics['train_accuracy']:.4f}")
    logger.info(f"Akurasi Testing: {metrics['test_accuracy']:.4f}")
    if metrics['train_r2']:
        logger.info(f"R² Training: {metrics['train_r2']:.4f}")
        logger.info(f"R² Testing: {metrics['test_r2']:.4f}")
    logger.info(f"Epoch Aktual: {metrics['final_epoch']}")
    
    # Langkah 5: Visualisasi
    trainer.plot_training_history()
    if isinstance(analyzer.data, pd.DataFrame):
        trainer.plot_feature_importance()
    
    return {
        "analysis": analysis_report,
        "config": config,
        "metrics": metrics
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Penggunaan: python TVM_advisor.py <path_ke_dataset>")
        sys.exit(1)
    
    results = main(sys.argv[1])
    print("\nRingkasan Hasil:")
    print(f"Rekomendasi Pembagian Data: {results['config']['split_ratio']*100:.0f}% training")
    print(f"Learning Rate: {results['config']['learning_rate']}")
    print(f"Akurasi Testing: {results['metrics']['test_accuracy']:.4f}")
