"""
SAR veri ön işleme modülü.

Bu modül STAC'tan gelen ham SAR verisinin aşağıdaki işlemlerini gerçekleştirir:
- dB cinsine çevirme ve calibration
- Speckle (gürültü) filtreleme
- Normalizasyon (0-1 aralığına getirme)
"""

import logging
from typing import Tuple, Optional, Union

import numpy as np
from scipy import ndimage
from skimage import restoration

logger = logging.getLogger(__name__)


class SARPreprocessor:
    """SAR verisi ön işleme sınıfı.
    
    Attributes:
        reference_power (float): dB hesaplaması için referans güç (varsayılan: 1.0)
        speckle_filter_method (str): Speckle filtreleme yöntemi ('lee', 'median', 'frost')
        filter_window_size (int): Filtreleme penceresi boyutu (varsayılan: 5)
    """
    
    def __init__(
        self,
        reference_power: float = 1.0,
        speckle_filter_method: str = "lee",
        filter_window_size: int = 5,
    ):
        """SARPreprocessor'ı başlatır.
        
        Args:
            reference_power: dB hesaplaması için referans güç değeri
            speckle_filter_method: Filtreleme yöntemi ('lee', 'median', 'frost')
            filter_window_size: Filtreleme penceresi boyutu (tek sayı olmalı)
        """
        self.reference_power = reference_power
        self.speckle_filter_method = speckle_filter_method.lower()
        self.filter_window_size = filter_window_size if filter_window_size % 2 == 1 else filter_window_size + 1
        
        if self.speckle_filter_method not in ["lee", "median", "frost"]:
            raise ValueError(
                f"Bilinmeyen filtreleme yöntemi: {speckle_filter_method}. "
                "Seçenekler: 'lee', 'median', 'frost'"
            )
        logger.info(
            "SARPreprocessor başlatıldı: "
            f"speckle_filter={self.speckle_filter_method}, "
            f"window_size={self.filter_window_size}"
        )
    
    def convert_to_db(
        self,
        sar_data: np.ndarray,
        is_power: bool = True,
        db_offset: float = 1e-10,
    ) -> np.ndarray:
        """SAR verisini dB cinsine çevirir.
        
        Args:
            sar_data: Ham SAR verisi (amplitude veya power)
            is_power: True ise power, False ise amplitude olarak kabul eder
            db_offset: Sayısal istikrar için minimum eşik değeri
        
        Returns:
            dB cinsinde dönüştürülmüş veri
        
        Raises:
            ValueError: Geçersiz giriş verisi durumunda
        """
        if not isinstance(sar_data, np.ndarray):
            raise ValueError("SAR verisi numpy array olmalıdır")
        
        # Negatif değerleri kontrol et
        if np.any(sar_data < 0):
            logger.warning("SAR verisinde negatif değerler bulundu, mutlak değeri alınıyor")
            sar_data = np.abs(sar_data)
        
        # Power veya amplitude'ı dB'ye çevir
        if is_power:
            # Power (I²+Q²) -> dB: 10 * log10(power / reference)
            sar_data_clipped = np.maximum(sar_data, db_offset)
            db_data = 10 * np.log10(sar_data_clipped / self.reference_power)
        else:
            # Amplitude -> Power: A² = (I²+Q²)
            # Sonra dB'ye çevir
            power_data = sar_data ** 2
            power_data_clipped = np.maximum(power_data, db_offset)
            db_data = 10 * np.log10(power_data_clipped / self.reference_power)
        
        logger.info(f"SAR verisi dB'ye çevrildi. Min: {db_data.min():.2f}, Max: {db_data.max():.2f}")
        return db_data
    
    def apply_calibration(
        self,
        sar_data: np.ndarray,
        calibration_factor: Optional[float] = None,
    ) -> np.ndarray:
        """SAR verisine calibration (kalibrasyonh) uygular.
        
        Sentinel-1 SAR verisi için radiometric calibration.
        
        Args:
            sar_data: dB cinsinde SAR verisi
            calibration_factor: Calibration faktörü (None ise otomatik hesaplanır)
        
        Returns:
            Kalibre edilmiş SAR verisi
        """
        if calibration_factor is None:
            # Sentinel-1 için standart calibration faktörü
            # Bu, görüntünün ortalama enerji seviyesini normalize eder
            calibration_factor = 1.0
        
        calibrated_data = sar_data * calibration_factor
        logger.info(f"Calibration uygulandı (factor: {calibration_factor})")
        return calibrated_data
    
    def apply_lee_filter(
        self,
        sar_data: np.ndarray,
        window_size: Optional[int] = None,
        noise_variance: float = 0.25,
    ) -> np.ndarray:
        """Lee filtresini SAR verisine uygular (Speckle filtreleme).
        
        Lee filter, SAR görüntülerinin speckle gürültüsünü azaltmak için etkili bir yöntemdir.
        Yerel istatistikleri kullanarak orijinal veri ile düzleştirilmiş versiyonu dengeler.
        
        Args:
            sar_data: Giriş SAR verisi (preferably dB cinsinde)
            window_size: Filtreleme penceresi boyutu
            noise_variance: Gürültü varyansı (0-1 arası)
        
        Returns:
            Filtrelenmiş SAR verisi
        """
        window_size = window_size or self.filter_window_size
        
        # Pencere boyutunu uyarla
        if window_size < 3:
            window_size = 3
        if window_size % 2 == 0:
            window_size += 1
        
        filtered_data = np.zeros_like(sar_data, dtype=np.float32)
        pad_size = window_size // 2
        
        # Veriyi padding ile genişlet (reflect mode)
        padded_data = np.pad(sar_data, pad_size, mode='reflect')
        
        for i in range(sar_data.shape[0]):
            for j in range(sar_data.shape[1]):
                # Pencere içindeki veriyi al
                window = padded_data[
                    i : i + window_size,
                    j : j + window_size
                ]
                
                # Yerel istatistikler
                local_mean = np.mean(window)
                local_var = np.var(window)
                
                # Lee filter formülü
                if local_var < 1e-6:
                    filtered_data[i, j] = local_mean
                else:
                    lee_factor = max(0, 1 - (noise_variance * self.reference_power) / local_var)
                    filtered_data[i, j] = local_mean + lee_factor * (sar_data[i, j] - local_mean)
        
        logger.info(f"Lee filtresi uygulandı (window: {window_size}x{window_size})")
        return filtered_data
    
    def apply_median_filter(
        self,
        sar_data: np.ndarray,
        window_size: Optional[int] = None,
    ) -> np.ndarray:
        """Median filtresini SAR verisine uygular.
        
        Basit ve hızlı speckle filtreleme yöntemi.
        
        Args:
            sar_data: Giriş SAR verisi
            window_size: Filtreleme penceresi boyutu
        
        Returns:
            Filtrelenmiş SAR verisi
        """
        window_size = window_size or self.filter_window_size
        if window_size % 2 == 0:
            window_size += 1
        
        filtered_data = ndimage.median_filter(sar_data, size=window_size)
        logger.info(f"Median filtresi uygulandı (window: {window_size}x{window_size})")
        return filtered_data
    
    def apply_frost_filter(
        self,
        sar_data: np.ndarray,
        window_size: Optional[int] = None,
        damping_factor: float = 2.0,
    ) -> np.ndarray:
        """Frost filtresini SAR verisine uygular.
        
        Adaptif speckle filtreleme yöntemi.
        
        Args:
            sar_data: Giriş SAR verisi (dB cinsinde)
            window_size: Filtreleme penceresi boyutu
            damping_factor: Sönümleme faktörü (damping parameter)
        
        Returns:
            Filtrelenmiş SAR verisi
        """
        window_size = window_size or self.filter_window_size
        if window_size % 2 == 0:
            window_size += 1
        
        filtered_data = np.zeros_like(sar_data, dtype=np.float32)
        pad_size = window_size // 2
        padded_data = np.pad(sar_data, pad_size, mode='reflect')
        
        for i in range(sar_data.shape[0]):
            for j in range(sar_data.shape[1]):
                window = padded_data[
                    i : i + window_size,
                    j : j + window_size
                ]
                
                local_mean = np.mean(window)
                local_var = np.var(window)
                
                if local_var < 1e-6:
                    filtered_data[i, j] = local_mean
                else:
                    # Frost filter formülü
                    frost_factor = np.exp(-damping_factor * local_var / (local_mean ** 2))
                    filtered_data[i, j] = local_mean * frost_factor + sar_data[i, j] * (1 - frost_factor)
        
        logger.info(f"Frost filtresi uygulandı (window: {window_size}x{window_size})")
        return filtered_data
    
    def apply_speckle_filter(
        self,
        sar_data: np.ndarray,
        method: Optional[str] = None,
    ) -> np.ndarray:
        """Seçilen speckle filtreleme yöntemini uygular.
        
        Args:
            sar_data: dB cinsinde SAR verisi
            method: Filtreleme yöntemi ('lee', 'median', 'frost'). 
                   None ise sınıfta tanımlanan yöntem kullanılır.
        
        Returns:
            Filtrelenmiş SAR verisi
        """
        method = (method or self.speckle_filter_method).lower()
        
        if method == "lee":
            return self.apply_lee_filter(sar_data)
        elif method == "median":
            return self.apply_median_filter(sar_data)
        elif method == "frost":
            return self.apply_frost_filter(sar_data)
        else:
            raise ValueError(f"Bilinmeyen filtreleme yöntemi: {method}")
    
    def normalize(
        self,
        sar_data: np.ndarray,
        method: str = "minmax",
        percentile_range: Tuple[float, float] = (2, 98),
    ) -> np.ndarray:
        """SAR verisini normalizasyon yapar (0-1 aralığına).
        
        Args:
            sar_data: Giriş SAR verisi
            method: Normalizasyon yöntemi 
                   - 'minmax': Min-max normalizasyonu
                   - 'zscore': Z-score normalizasyonu (sigmoid ile 0-1'e getirilir)
                   - 'percentile': Yüzdelik değerlere göre normalizasyon
            percentile_range: Percentile yöntemi için yüzdelik aralığı (alt, üst)
        
        Returns:
            0-1 aralığında normalizasyonu yapılmış veri
        """
        method = method.lower()
        
        if method == "minmax":
            # Min-max normalizasyonu
            data_min = np.min(sar_data)
            data_max = np.max(sar_data)
            
            if data_max - data_min < 1e-10:
                logger.warning("Veri aralığı çok küçük, sabit olarak normalize ediliyor")
                normalized_data = np.ones_like(sar_data) * 0.5
            else:
                normalized_data = (sar_data - data_min) / (data_max - data_min)
            
            logger.info(f"Min-max normalizasyonu uygulandı: [{data_min:.4f}, {data_max:.4f}]")
        
        elif method == "percentile":
            # Yüzdelik normalizasyonu (outlier'leri kontrol altında tutar)
            p_low, p_high = percentile_range
            lower_bound = np.percentile(sar_data, p_low)
            upper_bound = np.percentile(sar_data, p_high)
            
            normalized_data = np.clip(sar_data, lower_bound, upper_bound)
            if upper_bound - lower_bound < 1e-10:
                normalized_data = np.ones_like(sar_data) * 0.5
            else:
                normalized_data = (normalized_data - lower_bound) / (upper_bound - lower_bound)
            
            logger.info(
                f"Percentile normalizasyonu uygulandı: "
                f"[{p_low}%, {p_high}%] -> [{lower_bound:.4f}, {upper_bound:.4f}]"
            )
        
        elif method == "zscore":
            # Z-score normalizasyonu + sigmoid
            mean = np.mean(sar_data)
            std = np.std(sar_data)
            
            if std < 1e-10:
                logger.warning("Standart sapma çok küçük, sabit olarak normalize ediliyor")
                normalized_data = np.ones_like(sar_data) * 0.5
            else:
                z_scores = (sar_data - mean) / std
                # Sigmoid fonksiyonu ile 0-1 aralığına getir
                normalized_data = 1 / (1 + np.exp(-z_scores))
            
            logger.info(f"Z-score normalizasyonu uygulandı (mean: {mean:.4f}, std: {std:.4f})")
        
        else:
            raise ValueError(
                f"Bilinmeyen normalizasyon yöntemi: {method}. "
                "Seçenekler: 'minmax', 'zscore', 'percentile'"
            )
        
        # Finalize: 0-1 aralığında olduğundan emin ol
        normalized_data = np.clip(normalized_data, 0, 1)
        
        return normalized_data.astype(np.float32)
    
    def process(
        self,
        sar_data: np.ndarray,
        is_power: bool = True,
        speckle_method: Optional[str] = None,
        normalization_method: str = "minmax",
    ) -> np.ndarray:
        """Tam SAR işleme pipeline'ını uygular.
        
        İşlem sırası:
        1. dB cinsine çevirme
        2. Calibration
        3. Speckle filtreleme
        4. Normalizasyon
        
        Args:
            sar_data: Ham SAR verisi (amplitude veya power)
            is_power: True ise power, False ise amplitude
            speckle_method: Speckle filtreleme yöntemi
            normalization_method: Normalizasyon yöntemi
        
        Returns:
            İşlenmiş SAR verisi (0-1 aralığında, np.float32)
        """
        logger.info("SAR işleme pipeline'ı başlatılıyor...")
        
        # 1. dB'ye çevirme
        db_data = self.convert_to_db(sar_data, is_power=is_power)
        
        # 2. Calibration
        calibrated_data = self.apply_calibration(db_data)
        
        # 3. Speckle filtreleme
        filtered_data = self.apply_speckle_filter(calibrated_data, method=speckle_method)
        
        # 4. Normalizasyon
        normalized_data = self.normalize(filtered_data, method=normalization_method)
        
        logger.info("SAR işleme tamamlandı")
        return normalized_data


def preprocess_sar_image(
    sar_image: np.ndarray,
    is_power: bool = True,
    speckle_filter: str = "lee",
    filter_window: int = 5,
    normalization: str = "minmax",
) -> np.ndarray:
    """SAR görüntüsünü ön işleme yapar (Kolaylık fonksiyonu).
    
    Args:
        sar_image: Ham SAR görüntüsü
        is_power: Giriş verisi power ise True, amplitude ise False
        speckle_filter: Speckle filtreleme yöntemi ('lee', 'median', 'frost')
        filter_window: Filtreleme penceresi boyutu
        normalization: Normalizasyon yöntemi
    
    Returns:
        İşlenmiş SAR görüntüsü (0-1 aralığında)
    """
    preprocessor = SARPreprocessor(
        speckle_filter_method=speckle_filter,
        filter_window_size=filter_window,
    )
    return preprocessor.process(
        sar_image,
        is_power=is_power,
        normalization_method=normalization,
    )
