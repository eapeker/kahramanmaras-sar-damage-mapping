"""
SAR preprocessing modülünün unit testleri.

Her çalıştırıldığında testler dinamik olarak oluşturulan test verisiyle çalışır.
Sonuçlar cache'lenmez; her test çalıştırmasında yeni veriler işlenir.
"""

import pytest
import numpy as np
from src.preprocessing.preprocessing import SARPreprocessor, preprocess_sar_image


class TestSARPreprocessor:
    """SARPreprocessor sınıfının testleri."""
    
    @pytest.fixture
    def preprocessor(self):
        """Test için SARPreprocessor instance'ı."""
        return SARPreprocessor(
            reference_power=1.0,
            speckle_filter_method="lee",
            filter_window_size=5,
        )
    
    @pytest.fixture
    def sample_sar_data(self):
        """Örnek SAR verisi (power cinsinde, 100x100 piksel).
        
        Her test çalıştırması için yeni veriler üretilir (cache değil).
        """
        np.random.seed(42)
        # Gerçekçi SAR power verisi: 0.001 - 0.1 aralığında
        return np.random.uniform(0.001, 0.1, (100, 100)).astype(np.float32)
    
    @pytest.fixture
    def sample_sar_amplitude(self):
        """Örnek SAR amplitude verisi."""
        np.random.seed(42)
        return np.random.uniform(0.01, 0.3, (100, 100)).astype(np.float32)
    
    # ============== dB Çevirme Testleri ==============
    def test_convert_to_db_power(self, preprocessor, sample_sar_data):
        """Power verisinin dB'ye doğru çevrildiğini test et."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        
        # Çıktı float array olmalı
        assert isinstance(db_data, np.ndarray)
        assert db_data.dtype == np.ndarray.dtype or db_data.dtype == np.float64
        
        # Boyut değişmemeli
        assert db_data.shape == sample_sar_data.shape
        
        # dB değerleri negatif olabilir (gücü çünkü 1'den küçük)
        assert np.all(db_data < 0)  # Power < 1 ise dB < 0
        
        print(f"✓ Power to dB: {db_data.min():.2f} - {db_data.max():.2f} dB")
    
    def test_convert_to_db_amplitude(self, preprocessor, sample_sar_amplitude):
        """Amplitude verisinin dB'ye doğru çevrildiğini test et."""
        db_data = preprocessor.convert_to_db(sample_sar_amplitude, is_power=False)
        
        assert isinstance(db_data, np.ndarray)
        assert db_data.shape == sample_sar_amplitude.shape
        
        # Amplitude² -> Power -> dB
        print(f"✓ Amplitude to dB: {db_data.min():.2f} - {db_data.max():.2f} dB")
    
    def test_convert_to_db_negative_handling(self, preprocessor):
        """Negatif değerlerin işlenmesini test et."""
        data_with_negatives = np.array([[-0.5, 0.1], [0.05, -0.02]], dtype=np.float32)
        
        # Negatif değerlere karşı uyarı vermeli ama çalışmalı
        db_data = preprocessor.convert_to_db(data_with_negatives, is_power=True)
        assert db_data.shape == data_with_negatives.shape
        assert np.all(np.isfinite(db_data))  # NaN veya Inf olmamalı
        
        print("✓ Negative values handled correctly")
    
    # ============== Calibration Testleri ==============
    def test_apply_calibration(self, preprocessor, sample_sar_data):
        """Calibration uygulanmasını test et."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        calibrated = preprocessor.apply_calibration(db_data)
        
        # Calibration factor 1.0 ise veri aynı kalmalı (scale yapılmamış)
        assert calibrated.shape == db_data.shape
        print("✓ Calibration applied")
    
    # ============== Speckle Filtreleme Testleri ==============
    def test_lee_filter(self, preprocessor, sample_sar_data):
        """Lee filtresi test."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        filtered = preprocessor.apply_lee_filter(db_data, window_size=5)
        
        assert filtered.shape == db_data.shape
        assert filtered.dtype == np.float32
        
        # Filtreleme gürültüyü azaltmalı (varyans düşmeli)
        original_var = np.var(db_data)
        filtered_var = np.var(filtered)
        assert filtered_var <= original_var
        
        print(f"✓ Lee filter: variance {original_var:.4f} -> {filtered_var:.4f}")
    
    def test_median_filter(self, preprocessor, sample_sar_data):
        """Median filtresi test."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        filtered = preprocessor.apply_median_filter(db_data, window_size=5)
        
        assert filtered.shape == db_data.shape
        original_var = np.var(db_data)
        filtered_var = np.var(filtered)
        assert filtered_var <= original_var
        
        print(f"✓ Median filter: variance {original_var:.4f} -> {filtered_var:.4f}")
    
    def test_frost_filter(self, preprocessor, sample_sar_data):
        """Frost filtresi test."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        filtered = preprocessor.apply_frost_filter(db_data, window_size=5)
        
        assert filtered.shape == db_data.shape
        print("✓ Frost filter applied")
    
    def test_speckle_filter_method_selection(self, preprocessor, sample_sar_data):
        """Farklı speckle filtreleme yöntemlerini test et."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        
        for method in ["lee", "median", "frost"]:
            filtered = preprocessor.apply_speckle_filter(db_data, method=method)
            assert filtered.shape == db_data.shape
            assert np.all(np.isfinite(filtered))
        
        print("✓ All speckle filter methods work")
    
    def test_invalid_filter_method(self, preprocessor, sample_sar_data):
        """Geçersiz filtreleme yöntemi test."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        
        with pytest.raises(ValueError):
            preprocessor.apply_speckle_filter(db_data, method="invalid_filter")
    
    # ============== Normalizasyon Testleri ==============
    def test_normalize_minmax(self, preprocessor, sample_sar_data):
        """Min-Max normalizasyon test."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        normalized = preprocessor.normalize(db_data, method="minmax")
        
        assert normalized.shape == db_data.shape
        assert normalized.dtype == np.float32
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        assert np.isclose(normalized.min(), 0.0)
        assert np.isclose(normalized.max(), 1.0)
        
        print(f"✓ Min-Max normalization: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    def test_normalize_percentile(self, preprocessor, sample_sar_data):
        """Percentile normalizasyon test."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        normalized = preprocessor.normalize(
            db_data, 
            method="percentile",
            percentile_range=(2, 98)
        )
        
        assert normalized.shape == db_data.shape
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        print(f"✓ Percentile normalization: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    def test_normalize_zscore(self, preprocessor, sample_sar_data):
        """Z-Score normalizasyon test."""
        db_data = preprocessor.convert_to_db(sample_sar_data, is_power=True)
        normalized = preprocessor.normalize(db_data, method="zscore")
        
        assert normalized.shape == db_data.shape
        assert np.all(normalized >= 0) and np.all(normalized <= 1)
        print(f"✓ Z-Score normalization: [{normalized.min():.4f}, {normalized.max():.4f}]")
    
    def test_invalid_normalization_method(self, preprocessor, sample_sar_data):
        """Geçersiz normalizasyon yöntemi test."""
        with pytest.raises(ValueError):
            preprocessor.normalize(sample_sar_data, method="invalid")
    
    # ============== Full Pipeline Testleri ==============
    def test_full_pipeline(self, preprocessor, sample_sar_data):
        """Tam işleme pipeline'ını test et."""
        result = preprocessor.process(
            sample_sar_data,
            is_power=True,
            speckle_method="lee",
            normalization_method="minmax"
        )
        
        # Çıktı kontrolleri
        assert result.shape == sample_sar_data.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)
        assert np.all(np.isfinite(result))
        
        print(f"✓ Full pipeline: [{result.min():.4f}, {result.max():.4f}]")
    
    def test_pipeline_consistency(self, sample_sar_data):
        """Pipeline'ın tutarlılığını test et.
        
        Aynı parametrelerle çalıştırılması aynı sonucu vermeli.
        """
        preproc = SARPreprocessor(speckle_filter_method="median", filter_window_size=3)
        
        result1 = preproc.process(sample_sar_data.copy(), is_power=True)
        result2 = preproc.process(sample_sar_data.copy(), is_power=True)
        
        # Sonuçlar neredeyse aynı olmalı (floating point hassasiyeti nedeniyle)
        assert np.allclose(result1, result2, rtol=1e-5)
        print("✓ Pipeline consistency verified")
    
    # ============== Convenience Function Testleri ==============
    def test_convenience_function(self, sample_sar_data):
        """Kolaylık fonksiyonunu test et."""
        result = preprocess_sar_image(
            sample_sar_data,
            is_power=True,
            speckle_filter="median",
            filter_window=5,
            normalization="minmax"
        )
        
        assert result.shape == sample_sar_data.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0) and np.all(result <= 1)
        print("✓ Convenience function works")
    
    # ============== Edge Case Testleri ==============
    def test_small_input(self, preprocessor):
        """Küçük input (3x3) test."""
        small_data = np.array([[0.01, 0.02, 0.03],
                               [0.04, 0.05, 0.06],
                               [0.07, 0.08, 0.09]], dtype=np.float32)
        
        result = preprocessor.process(small_data)
        assert result.shape == small_data.shape
        assert np.all(result >= 0) and np.all(result <= 1)
        print("✓ Small input handled")
    
    def test_uniform_input(self, preprocessor):
        """Tek değerli input test (tüm piksel aynı)."""
        uniform_data = np.ones((50, 50), dtype=np.float32) * 0.05
        
        result = preprocessor.process(uniform_data)
        assert result.shape == uniform_data.shape
        # Tek değer olsa bile işlem tamamlanmalı
        assert np.all(np.isfinite(result))
        print("✓ Uniform input handled")
    
    def test_large_dynamic_range(self, preprocessor):
        """Geniş dinamik aralık test."""
        np.random.seed(42)
        # 1e-4 ile 1.0 arasında geniş aralık
        large_range = np.random.uniform(1e-4, 1.0, (100, 100)).astype(np.float32)
        
        result = preprocessor.process(large_range)
        assert result.shape == large_range.shape
        assert np.all(result >= 0) and np.all(result <= 1)
        print("✓ Large dynamic range handled")


if __name__ == "__main__":
    # Doğrudan çalıştırma: python -m pytest tests/test_preprocessing.py -v
    # veya sadece: pytest tests/test_preprocessing.py -v
    pytest.main([__file__, "-v", "-s"])
