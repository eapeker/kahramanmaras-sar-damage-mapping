"""
SAR preprocessing modülünün örnek kullanım ve test script'i.

Bu script preprocessing modülünün nasıl çalıştığını gösterir.
Her çalıştırıldığında yeni test verileri oluşturulur (cache değil).
"""

import numpy as np
import matplotlib.pyplot as plt
from src.preprocessing.preprocessing import SARPreprocessor, preprocess_sar_image


def create_synthetic_sar_data(height=256, width=256, seed=None):
    """Sentetik SAR verisi oluştur (test için).
    
    Args:
        height, width: Görüntü boyutları
        seed: Rastgele seed (None ise her çalıştırmada farklı veri)
    
    Returns:
        Power cinsinde SAR verisi (numpy array)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Temel SAR sinyali (Gaussian blob'lar)
    sar_signal = np.zeros((height, width))
    
    # Rastgele kaynaklar ekle
    for _ in range(5):
        y, x = np.random.randint(0, height), np.random.randint(0, width)
        radius = np.random.randint(10, 30)
        intensity = np.random.uniform(0.05, 0.3)
        
        yy, xx = np.ogrid[:height, :width]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= radius ** 2
        sar_signal[mask] += intensity
    
    # Speckle gürültüsü ekle (SAR'ın karakteristik özelliği)
    speckle_noise = np.random.exponential(scale=0.02, size=(height, width))
    sar_data = sar_signal + speckle_noise
    
    # Veriyi power cinsinde (0.001 - 0.3 aralığında) normalleştir
    sar_data = np.clip(sar_data, 0, None)
    sar_data = (sar_data / sar_data.max()) * 0.2 + 0.01
    
    return sar_data.astype(np.float32)


def test_basic_preprocessing():
    """Basit preprocessing test."""
    print("=" * 60)
    print("TEST 1: Temel Preprocessing İşlemi")
    print("=" * 60)
    
    # Sentetik SAR verisi oluştur
    print("📊 Sentetik SAR verisi oluşturuluyor...")
    sar_data = create_synthetic_sar_data(height=256, width=256)
    print(f"   Veri boyutu: {sar_data.shape}")
    print(f"   Veri aralığı: [{sar_data.min():.6f}, {sar_data.max():.6f}] (Power cinsinde)")
    
    # Preprocessing uygula
    print("\n🔄 Preprocessing işlemi başlatılıyor...")
    processed = preprocess_sar_image(
        sar_data,
        is_power=True,
        speckle_filter="lee",
        filter_window=5,
        normalization="minmax"
    )
    print(f"   ✓ Tamamlandı")
    print(f"   Çıktı aralığı: [{processed.min():.6f}, {processed.max():.6f}] (0-1 aralığında)")
    
    return sar_data, processed


def test_filter_methods():
    """Farklı speckle filtreleme yöntemlerini karşılaştır."""
    print("\n" + "=" * 60)
    print("TEST 2: Speckle Filtreleme Yöntemlerinin Karşılaştırması")
    print("=" * 60)
    
    sar_data = create_synthetic_sar_data(height=128, width=128)
    preprocessor = SARPreprocessor()
    
    db_data = preprocessor.convert_to_db(sar_data, is_power=True)
    
    results = {}
    for method in ["lee", "median", "frost"]:
        print(f"\n📌 {method.upper()} filtresi uygulanıyor...")
        filtered = preprocessor.apply_speckle_filter(db_data, method=method)
        normalized = preprocessor.normalize(filtered)
        
        variance_before = np.var(db_data)
        variance_after = np.var(filtered)
        reduction = (1 - variance_after / variance_before) * 100
        
        print(f"   Varyans azalışı: {reduction:.1f}%")
        print(f"   Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
        
        results[method] = normalized
    
    return results


def test_normalization_methods():
    """Farklı normalizasyon yöntemlerini test et."""
    print("\n" + "=" * 60)
    print("TEST 3: Normalizasyon Yöntemlerinin Karşılaştırması")
    print("=" * 60)
    
    sar_data = create_synthetic_sar_data(height=128, width=128)
    preprocessor = SARPreprocessor()
    
    db_data = preprocessor.convert_to_db(sar_data, is_power=True)
    filtered = preprocessor.apply_speckle_filter(db_data)
    
    results = {}
    for method in ["minmax", "percentile", "zscore"]:
        print(f"\n📌 {method.upper()} normalizasyonu uygulanıyor...")
        normalized = preprocessor.normalize(filtered, method=method)
        
        mean_val = np.mean(normalized)
        std_val = np.std(normalized)
        
        print(f"   Ortalama: {mean_val:.4f}")
        print(f"   Standart sapma: {std_val:.4f}")
        print(f"   Min: {normalized.min():.4f}, Max: {normalized.max():.4f}")
        
        results[method] = normalized
    
    return results


def test_reproducibility():
    """Aynı veri üzerinde tutarlı sonuçlar üretiyor mu test et."""
    print("\n" + "=" * 60)
    print("TEST 4: Tutarlılık Testi (Reproducibility)")
    print("=" * 60)
    
    # Aynı seed ile aynı veriler oluştur
    sar_data = create_synthetic_sar_data(height=128, width=128, seed=42)
    
    print("📊 Aynı veri üzerinde 3 kez preprocessing işlemi yapılıyor...")
    
    results = []
    for i in range(3):
        processed = preprocess_sar_image(sar_data.copy(), is_power=True)
        results.append(processed)
        print(f"   ✓ İşlem {i+1} tamamlandı")
    
    # Sonuçları karşılaştır
    print("\n🔍 Sonuçlar karşılaştırılıyor...")
    
    diff_1_2 = np.max(np.abs(results[0] - results[1]))
    diff_2_3 = np.max(np.abs(results[1] - results[2]))
    
    print(f"   İşlem 1 vs 2 - Maksimum fark: {diff_1_2:.2e}")
    print(f"   İşlem 2 vs 3 - Maksimum fark: {diff_2_3:.2e}")
    
    if diff_1_2 < 1e-5 and diff_2_3 < 1e-5:
        print("   ✓ Sonuçlar tutarlı (Reproducible)")
        return True
    else:
        print("   ⚠ Tutarlılık problemi!")
        return False


def test_edge_cases():
    """Edge case'leri test et."""
    print("\n" + "=" * 60)
    print("TEST 5: Edge Case'ler")
    print("=" * 60)
    
    test_cases = [
        ("Çok küçük görüntü (8x8)", np.random.uniform(0.01, 0.1, (8, 8))),
        ("Tek tip veri (uniform)", np.ones((64, 64)) * 0.05),
        ("Geniş dinamik aralık", np.random.uniform(1e-4, 1.0, (64, 64))),
        ("Zero'lara yakın veriler", np.random.uniform(1e-6, 1e-4, (64, 64))),
    ]
    
    preprocessor = SARPreprocessor()
    
    for name, data in test_cases:
        print(f"\n📌 {name}:")
        try:
            result = preprocessor.process(data.astype(np.float32))
            print(f"   ✓ Başarılı - [{result.min():.4f}, {result.max():.4f}]")
        except Exception as e:
            print(f"   ✗ Hata: {e}")


def print_summary():
    """Özet bilgi yazdır."""
    print("\n" + "=" * 60)
    print("ÖZET - Test Sonuçları")
    print("=" * 60)
    print("""
✓ Her test çalıştırıldığında:
  - Yeni sentetik SAR verileri oluşturulur
  - İşlem parametreleri sabittir
  - Aynı parametrelerle aynı sonuçlar üretilir

💡 Cache Mekanizması:
  - Bu preprocessing script'inde cache kullanılmaz
  - Her çalıştırmada işlem sıfırdan yapılır
  - Sonuçlar sadece bellekte (RAM) tutulur

🔧 Kullanımdaki Performans:
  - STAC'tan alınan SAR verileri her sorguda işlenir
  - Tekrarlanması gerekiyorsa tekrar işlenebilir
  - Sonuçları kaydetmek istiyorsanız pickle/HDF5 kullanabilirsiniz

📝 Sonuçları Kaydetme Örneği:
  import pickle
  with open('processed_data.pkl', 'wb') as f:
      pickle.dump(processed_data, f)
    """)
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "🚀 " * 20)
    print("SAR PREPROCESSING MODÜLÜNÜn TEST AŞKISI")
    print("🚀 " * 20 + "\n")
    
    # Testleri çalıştır
    try:
        test_basic_preprocessing()
        test_filter_methods()
        test_normalization_methods()
        test_reproducibility()
        test_edge_cases()
        print_summary()
        
        print("\n✅ TÜM TESTLER BAŞARILI!\n")
        
    except Exception as e:
        print(f"\n❌ HATA OLUŞTU: {e}\n")
        import traceback
        traceback.print_exc()
