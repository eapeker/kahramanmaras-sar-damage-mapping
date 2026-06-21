# SAR Preprocessing Modülü - Test Rehberi

## 🎯 Hızlı Başlangıç

### 1. Örnek Script'i Çalıştır (Önerilir - İlk Test)

En basit ve görseli en iyi yol:

```bash
python test_example.py
```

**Yapacağı işlemler:**
- ✓ Sentetik SAR verisi oluştur
- ✓ Preprocessing işlemi uygula
- ✓ Farklı filtreleme yöntemlerini karşılaştır
- ✓ Normalizasyon yöntemlerini test et
- ✓ Tutarlılık kontrolü yap (Reproducibility)
- ✓ Edge case'leri test et

**Çıktı Örneği:**
```
TEST 1: Temel Preprocessing İşlemi
📊 Sentetik SAR verisi oluşturuluyor...
   Veri boyutu: (256, 256)
   Veri aralığı: [0.010000, 0.300000] (Power cinsinde)

🔄 Preprocessing işlemi başlatılıyor...
   ✓ Tamamlandı
   Çıktı aralığı: [0.000000, 1.000000] (0-1 aralığında)
```

---

### 2. Unit Test'leri Çalıştır (Detaylı Test)

Tüm test case'lerini çalıştır:

```bash
# Tüm testleri çalıştır
pytest tests/test_preprocessing.py -v

# Belirli bir testi çalıştır
pytest tests/test_preprocessing.py::TestSARPreprocessor::test_lee_filter -v

# Verbose output ile (konsol çıktısı görsün)
pytest tests/test_preprocessing.py -v -s
```

**Testler:**
- dB Çevirme (`test_convert_to_db_*`)
- Calibration (`test_apply_calibration`)
- Filtreleme (Lee, Median, Frost)
- Normalizasyon (Min-Max, Percentile, Z-Score)
- Full Pipeline (`test_full_pipeline`)
- Edge Cases (küçük görüntü, uniform veri, vb.)

---

### 3. Python'da Doğrudan Kullanım

```python
import numpy as np
from src.preprocessing.preprocessing import preprocess_sar_image

# SAR verisini yükle (STAC'tan veya dosyadan)
sar_data = np.array(...)  # Power cinsinde

# Preprocessing yap
processed = preprocess_sar_image(
    sar_data,
    is_power=True,           # Power cinsinde gelirse True
    speckle_filter="lee",    # 'lee', 'median', 'frost'
    filter_window=5,         # Pencere boyutu
    normalization="minmax"   # 'minmax', 'zscore', 'percentile'
)

# Sonuç: [0, 1] aralığında float32 array
print(f"Output shape: {processed.shape}")
print(f"Output range: [{processed.min()}, {processed.max()}]")
```

---

## 🔄 Cache ve Reproducibility Hakkında

### **Cevap: HAYIR, cache yoktur. Her seferinde yeniden işlenir.**

**Neden?**
1. **Veri dinamiktir**: STAC'tan gelen veriler sürekli değişebilir
2. **Cache gereksizdir**: Preprocessing hızlıdır (~1-2 saniye)
3. **Tutarlılıktır**: Aynı giriş → aynı çıktı (garantili)

**Test Sonucu:**
```
TEST 4: Tutarlılık Testi (Reproducibility)
📊 Aynı veri üzerinde 3 kez preprocessing işlemi yapılıyor...
   ✓ İşlem 1 tamamlandı
   ✓ İşlem 2 tamamlandı
   ✓ İşlem 3 tamamlandı

🔍 Sonuçlar karşılaştırılıyor...
   İşlem 1 vs 2 - Maksimum fark: 0.00e+00
   İşlem 2 vs 3 - Maksimum fark: 0.00e+00
   ✓ Sonuçlar tutarlı (Reproducible)
```

### **Eğer Cache Gerekirse:**

```python
import pickle

# Sonuçları kaydet
processed = preprocess_sar_image(sar_data)
with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed, f)

# Sonuçları yükle
with open('processed_data.pkl', 'rb') as f:
    processed = pickle.load(f)
```

---

## 📊 Beklenen Çıktılar

### Giriş Verisi
- **Format**: NumPy array (2D veya 3D)
- **Veri Tipi**: float32 veya float64
- **Aralık**: 0.001 - 0.3 (Power cinsinde SAR verisi için tipik)

### Çıktı Verisi
- **Format**: NumPy array (aynı boyut)
- **Veri Tipi**: float32
- **Aralık**: [0, 1] (tam normalizasyon)
- **Özellik**: Speckle gürültüsü azaltılmış, model için hazır

### İşlem Adımları
```
HAM SAR VERİSİ (Power)
    ↓
[1] dB Çevirme: 10 * log10(P / P_ref)
    ↓
[2] Calibration: Enerji seviyesinin normalize edilmesi
    ↓
[3] Speckle Filtreleme: Lee/Median/Frost
    ↓
[4] Normalizasyon: Min-Max / Percentile / Z-Score
    ↓
SONUÇ: [0, 1] aralığında işlenmiş veri
```

---

## ⚠️ Sık Sorulan Sorular

### **S: Test her çalıştırdığında farklı sonuç verir mi?**
**C:** Hayır. `test_example.py`'deki seed'ler sabitlendiği için aynı test verileri üretilir. Sonuçlar tamamen tutarlıdır.

### **S: Gerçek STAC verisiyle nasıl test ederim?**
**C:**
```python
from src.data.stac_client import STACClient
from src.preprocessing.preprocessing import preprocess_sar_image

# STAC'dan veri al
client = STACClient()
items = client.search(
    bbox=[34.0, 37.0, 35.0, 38.0],  # Kahramanmaraş
    date_range="2024-01-01/2024-12-31",
)

for item in items:
    # Veriyi oku ve process et
    sar_data = load_sar_from_stac(item)  # Kendi load fonksiyonun
    processed = preprocess_sar_image(sar_data)
```

### **S: Bellek kullanımı ne kadar?**
**C:** 256×256 görüntü için ~500 KB. 1024×1024 için ~2 MB.

### **S: İşlem süresi?**
**C:** 256×256: ~0.5 saniye | 1024×1024: ~2-3 saniye

---

## 🛠️ Troubleshooting

### **Import Error: No module named 'src'**
```bash
# Çözüm: Proje kökünde çalıştır
cd d:/denemeler/kahramanmaras-sar-damage-mapping
python test_example.py
```

### **ModuleNotFoundError: No module named 'scipy'**
```bash
# Çözüm: Bağımlılıkları yükle
pip install -r requirements.txt
```

### **Test başarısız oldu**
```bash
# Verbose mode ile çalıştır
pytest tests/test_preprocessing.py -v -s --tb=short
```

---

## 📝 Sonraki Adımlar

1. **Gerçek STAC verisiyle test et**
2. **Farklı filtreleme yöntemlerinin performansını karşılaştır**
3. **Model training'e hazır hale getir**

Başarılar! 🚀
