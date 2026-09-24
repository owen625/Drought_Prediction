import os
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple, List

class DroughtPredictionService:
    """Service untuk prediksi risiko kekeringan"""
    
    def __init__(self, model_path: str, encoder_path: str):
        self.model = None
        self.encoder = None
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.load_model()
    
    def load_model(self):
        """Load model dan encoder dari file"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                print(f"✓ Model loaded dari {self.model_path}")
            else:
                print(f"✗ Model file tidak ditemukan: {self.model_path}")
                raise FileNotFoundError(f"Model tidak ditemukan di {self.model_path}")
            
            if os.path.exists(self.encoder_path):
                self.encoder = joblib.load(self.encoder_path)
                print(f"✓ Encoder loaded dari {self.encoder_path}")
            else:
                print(f"✗ Encoder file tidak ditemukan: {self.encoder_path}")
                raise FileNotFoundError(f"Encoder tidak ditemukan di {self.encoder_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def is_ready(self) -> bool:
        """Check apakah model sudah ready"""
        return self.model is not None and self.encoder is not None
    
    def encode_wind_direction(self, ddd_car: str) -> int:
        """Encode wind direction menggunakan encoder yang sama dengan training"""
        try:
            encoded = self.encoder.transform([ddd_car])[0]
            return int(encoded)
        except Exception as e:
            # Jika nilai tidak ada dalam training data, gunakan default
            print(f"Warning: Encoding {ddd_car} gagal, menggunakan default")
            return 0
    
    def prepare_features(self, 
                        tn: float, tx: float, tavg: float,
                        rh_avg: float, precipitation: float,
                        ss: float, ff_x: float, ddd_car: str) -> Tuple[pd.DataFrame, int]:
        """Siapkan features untuk prediksi"""
        
        ddd_car_code = self.encode_wind_direction(ddd_car)
        
        # Features harus dalam urutan YANG SAMA dengan training
        features_dict = {
            'Tn': tn,
            'Tx': tx,
            'Tavg': tavg,
            'RH_avg': rh_avg,
            'precipitation': precipitation,
            'ss': ss,
            'ff_x': ff_x,
            'ddd_car_code': ddd_car_code
        }
        
        return features_list, ddd_car_code
    
    def calculate_spi(self, precipitation: float, 
                     ref_mean: float = 75.0, ref_std: float = 45.0) -> float:
        """Hitung SPI (Standardized Precipitation Index)"""
        spi = (precipitation - ref_mean) / ref_std if ref_std != 0 else 0
        return round(spi, 4)
    
    def predict(self, 
               tn: float, tx: float, tavg: float,
               rh_avg: float, precipitation: float,
               ss: float, ff_x: float, ddd_car: str) -> dict:
        """
        Buat prediksi risiko kekeringan
        
        Returns:
            dict dengan keys: spi, risk (label), risk_score (probabilitas)
        """
        if not self.is_ready():
            raise RuntimeError("Model belum di-load. Panggil load_model() terlebih dahulu")
        
        # Siapkan features
        X, ddd_code = self.prepare_features(tn, tx, tavg, rh_avg, precipitation, ss, ff_x, ddd_car)
        
        # Hitung SPI
        spi = self.calculate_spi(precipitation)
        
        # Prediksi dengan model (output: 0=low risk, 1=medium/high risk)
        risk_pred = self.model.predict(X)[0]
        
        # Dapatkan probability scores
        risk_proba = self.model.predict_proba(X)[0]
        risk_score = max(risk_proba)  # Confidence score
        
        # Map prediksi ke label risiko
        # Model trained dengan 2 classes: 0=low, 1=medium
        # Tapi SPI bisa >1 untuk risiko tinggi
        if spi <= -1.0:
            risk_label = 'high'
        elif spi <= 0:
            risk_label = 'medium'
        else:
            risk_label = 'low'
        
        return {
            'spi': spi,
            'risk': risk_label,
            'risk_score': round(float(risk_score), 4),
            'ddd_car_code': int(ddd_code)
        }


# Initialize service (akan di-import di main.py)
def init_prediction_service(project_dir: str = ".") -> DroughtPredictionService:
    """Inisialisasi prediction service"""
    model_path = os.path.join(project_dir, 'models', 'rf_drought_model.pkl')
    encoder_path = os.path.join(project_dir, 'models', 'label_encoder_ddd_car.pkl')
    
    return DroughtPredictionService(model_path, encoder_path)
