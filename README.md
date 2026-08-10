# Sefalometrik Radyografi Analiz ve Sınıflandırma Sistemi

Bu proje, lateral sefalometrik röntgen görüntülerinin derin öğrenme tabanlı analizi, görüntü iyileştirmesi ve otomatik iskeletsel sınıflandırması (Sınıf I, II, III) için geliştirilmiş akademik ve üretime hazır (production-ready) bir uçtan uca boru hattıdır (pipeline).

Sistem; gelişmiş uzamsal görüntü iyileştirme yöntemlerini, derin evrişimli ağ mimarilerini, çok ölçekli dikkat (attention) mekanizmalarını ve bulanık mantık (fuzzy logic) tabanlı ölçek birleştirme modüllerini bir araya getiren özgün bir derin ağ mimarisine sahiptir.

---

## 📐 Mimari ve Veri Akış Şeması

Sisteme giren bir röntgen görüntüsü, aşağıdaki aşamalardan geçerek nihai sınıf tahminine dönüştürülür:

```
      Girdi Röntgen Görüntüsü [BGR]
                   │
                   ▼
         [ImageEnhancement] 
    (16-Adımlı Uzamsal İyileştirme)
                   │
                   ▼
       [EfficientNetV2-S Backbone]
 (Ölçekler Arası Özellik Piramitleri: P2, P3, P4, P5)
                   │
                   ▼
       [Multi-Scale AFEM Modülü]
   (Uyarlanabilir Özellik İyileştirme)
                   │
                   ▼
         [CFFM Fusion Modülü]
 (Bulanık Mantık Tabanlı Çok Ölçekli Birleştirme)
                   │
                   ▼
         [ACAM Attention Modülü]
      (Uzamsal Bağlam Kümeleme)
                   │
                   ▼
          [GRLM Emdedding Modülü]
  (512-Boyutlu L2 Normalize Temsil Vektörü)
                   │
                   ▼
        [Classification Head]
       (Dropout + Lineer Katman)
                   │
                   ▼
              Raw Logits
                   │
                   ▼
                Softmax
                   │
                   ▼
        Tahmin Sınıfı & Güven Skoru
```

---

## 🛠️ Temel Modüller ve İşlevleri

1. **Görüntü İyileştirme (`ImageEnhancement`)**:
   * CIELAB renk uzayında CLAHE ile yerel kontrast artırma.
   * Bilateral filtreleme ile kenar koruyucu gürültü azaltma.
   * $0^\circ, 45^\circ, 90^\circ, 135^\circ$ yönlü uzamsal evrişim filtreleri ile çok yönlü kenar çıkarımı.
   * Yön enerjilerine göre adaptif yön ağırlıklandırma (ADW).
   * Konveks özellik füzyonu ($0.65 \times \text{Lightness} + 0.35 \times \text{Edge}$).

2. **EfficientNetV2-S Backbone**:
   * ImageNet üzerinde önceden eğitilmiş ağırlıklar kullanır.
   * Aşama 2, 3, 4 ve 5 çıkışlarından piramit seviyeleri ($P_2, P_3, P_4, P_5$) oluşturulur ve $1 \times 1$ evrişimlerle kanallar 256'ya eşitlenir.

3. **Multi-Scale AFEM**:
   * Özellik haritalarının önem katsayılarını çıkararak kritik anatomik bölgelere odaklanır.

4. **CFFM (Fuzzy Fusion)**:
   * Farklı ölçeklerdeki özellikleri bulanık üyelik fonksiyonları üzerinden hesaplanan bulanık ölçek ağırlıklarıyla akıllı bir şekilde birleştirir.

5. **ACAM & GRLM**:
   * ACAM, küresel ve yerel bağlamsal ilişkileri kurar.
   * GRLM, özellikleri 512 boyutlu bir birim hiperküre üzerine izdüşürerek L2 normalize edilmiş robust temsil vektörleri üretir.

---

## 📊 Veri Seti Yapısı ve Entegrasyonu

Veri seti entegrasyonu (`real_dataset_integration.py`), veri sızıntılarını (data leakage) engellemek amacıyla **hasta düzeyinde gruplanmış ve sınıf bazında tabakalandırılmış (Stratified Patient-Level Group Split)** bir strateji kullanır.

* **Toplam Geçerli Röntgen**: 1430
* **Benzersiz Hasta Sayısı**: 585
* **Bölümleme Oranları**: %70 Eğitim, %15 Doğrulama, %15 Test (Train: 982, Val: 221, Test: 227)
* **Sınıf Dağılımı (Valid)**: Sınıf 1 (%14.7), Sınıf 2 (%71.0), Sınıf 3 (%14.3)
* **Eğitim Sınıf Ağırlıkları**: `[2.32, 0.47, 2.20]` (Çoklu Sınıf Dengesi İçin)
* **Hasta Örtüşmesi (Overlap)**: Kesinlikle **0** (Tüm split çiftlerinde doğrulanmıştır).

---

## 💻 Kurulum ve Gereksinimler

Proje Python 3.8+ tabanlıdır. Gerekli kütüphaneleri yüklemek için:

```bash
pip install torch torchvision numpy opencv-python scikit-learn flask matplotlib
```

---

## 🚀 Kullanım Rehberi

Sistemi eğitmek, test etmek veya görsel arayüzü başlatmak için aşağıdaki yöntemleri kullanabilirsiniz.

### Yöntem A: Web Dashboard (Görsel Kontrol Paneli)
Sistemi izlemek, eğitmek ve test etmek için geliştirilmiş görsel ekranı başlatır:

```bash
python app.py
```
* Tarayıcınızda `http://127.0.0.1:5000` adresini açın.
* **Hyperparameters** panelinden Epochs, Batch Size, Learning Rate, Patience ve Phase Epochs değerlerini ince ayarlayabilirsiniz.
* **Start Training** butonu ile eğitimi arka planda başlatabilir, konsol çıktılarını ve kayıp/başarı grafiklerini canlı takip edebilirsiniz.
* **Clinical Inference** sekmesinden sefalometrik röntgenlerinizi yükleyerek hızlı analiz (tahmin, sınıf olasılıkları, preprocessed görüntü ve GRLM embedding'i) yapabilirsiniz.

### Yöntem B: Terminal Üzerinden Eğitim
Tüm sistemi özelleştirilmiş hiperparametrelerle doğrudan uçtan uca eğitmek için:

```bash
python train_full_model.py --epochs 30 --batch_size 8 --lr 0.001 --patience 7 --phase1 10 --phase2 20
```

Eğitim sırasında en iyi model validation setindeki **Macro F1** skoruna göre seçilir ve `checkpoints/full_model/best_checkpoint.pth` konumuna kaydedilir. Eğitim bittiğinde test kümesinde değerlendirme yapılıp kapsamlı bir rapor `reports/full_model/full_model_report.md` olarak üretilir.

### Yöntem C: CLI Üzerinden Tekil Tahmin (Inference)
Eğitilmiş bir checkpoint ile bir röntgen görüntüsünü test etmek için:

```bash
python main.py --mode predict --image path/to/image.jpg --checkpoint checkpoints/full_model/best_checkpoint.pth
```

---

## 🛡️ Tıbbi Veri Gizliliği ve Güvenlik Kontrolleri

Sistem tıbbi görüntüleme standartlarına uygun şekilde tasarlanmıştır:
* **Gizlilik**: Loglarda hiçbir hasta adı, kimliği veya veri tabanı dizin yolları yer almaz.
* **Sayısal Güvenlik**: Logit ve olasılık vektörlerinde NaN/Inf kontrolleri yapılarak tutarsız sonuçların üretilmesi engellenir. Olasılıkların toplamının 1.0 olması ve negatif olmaması doğrulanır.
* **Klinik İletişim Dil**: Ekranda gösterilen olasılık skoru klinik bir kesinlik veya otomatik tanı iddiası yerine **"Prediction Confidence" (Tahmin Güven Skoru)** olarak ifade edilir.

---

## 📂 Dosya Yapısı

```
sefalometrik/
├── app.py                     # Web Dashboard Backend sunucusu
├── main.py                    # CLI giriş noktası (Demo, Predict, Train)
├── train_full_model.py        # Tam model eğitim scripti
├── real_dataset_integration.py# Dataset doğrulama ve gruplanmış split scripti
├── backbone/                  # EfficientNetV2-S mimarisi
├── models/                    # CephalometricNet ve Classification Head tanımları
├── modules/                   # AFEM, CFFM, ACAM, GRLM ve ImageEnhancement sınıfları
├── loss/                      # Sınıf ağırlıklı Cross Entropy kaybı
├── metrics/                   # Akademik metrik hesaplama ve grafik çizici
├── configs/                   # Global hiperparametre dataclass konfigürasyonları
├── docs/                      # Detaylı inference dokümantasyonu (inference.md)
├── templates/                 # Web Dashboard HTML arayüzü (index.html)
└── reports/                   # Kaydedilen eğitim geçmişleri, metrikler ve görsel çıktılar
```
