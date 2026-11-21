# Küçük İşletmeler İçin Yapay Zekâ Destekli Stok Talep Tahmin Sistemi

Geçmiş satış verilerinden ürün bazında **28 günlük talep tahmini** üreten ve bunu
**güvenlik stoku + yeniden sipariş noktası** önerisine çeviren karar destek sistemi.
Bitirme projesi — 4 kişilik ekip, 10 haftalık sprint.

## Hızlı başlangıç

```bash
git clone https://github.com/TNNladie/stok-talep-tahmin.git
cd stok-talep-tahmin
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test          # birim testleri
make demo          # internetsiz sentetik veriyle tüm akış
make data         # Online Retail II indir (UCI erişilemezse: make data-github)
make run-full     # gerçek veride backtest + final model eğitimi (models/) + rapor şekli
make app          # Streamlit karar destek ekranı (kayıtlı modeli kullanır)
make tune         # (isteğe bağlı) hiperparametre ayarını yeniden yap
```

## Akış

```
Ham satış verisi → temizleme (iade/negatif at, günlük topla, eksik gün = 0)
 → öznitelik (lag 28–56, rolling 7/28/91, geçen yıl aynı dönem, mağaza geneli, takvim, açık/kapalı gün, fiyat)
 → modeller (Persistence, Mevsimsel Naive, MA-7/28 | RandomForest, XGBoost, LightGBM)
 → zaman bazlı backtest (3 kat × 28 gün) → MAE / RMSE / MAPE / WAPE
 → doğrulama katlarında hiperparametre ayarı → final LightGBM (models/)
 → SS = z·σ·√L, ROP = D_L + SS → sipariş önerisi → Streamlit ekranı
```

## Güncel sonuç (Online Retail II, 50 ürün, 3 test katı × 28 gün)

| Model | MAE | WAPE |
|---|---:|---:|
| **LightGBM** | **42.68** | **%79.7** |
| XGBoost | 44.03 | %82.3 |
| MA-28 (en iyi baseline) | 53.82 | %100.6 |

LightGBM en iyi baseline'dan **%20.7** daha düşük MAE üretiyor (hedef: ≥ %10). Ayrıntı: `reports/model_egitimi_raporu.md`.

## Veri setleri

| Set | Rol | Erişim |
|---|---|---|
| [Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Ana set | `make data` (ucimlrepo) |
| [Store Item Demand (Kaggle)](https://www.kaggle.com/c/demand-forecasting-kernels-only/data) | Kontrol seti | `python -m src.data.download --source store_item` |
| [M5 Accuracy (Kaggle)](https://www.kaggle.com/c/m5-forecasting-accuracy) | İsteğe bağlı | Kaggle |

Ham veri repoya yüklenmez; `data/` klasörü `.gitignore` içindedir.

## Klasör yapısı

```
configs/config.yaml        # tüm parametreler
src/data/                  # indirme ve temizleme
src/features/              # öznitelik üretimi
src/models/                # baseline + ML modelleri
src/evaluation/            # metrikler ve backtest
src/inventory/             # stok politikası ve simülasyon
src/pipeline.py            # uçtan uca akış
app/streamlit_app.py       # karar destek ekranı
notebooks/                 # EDA ve deney defterleri
tests/                     # pytest
docs/                      # literatür, problem tanımı, veri sözlüğü
reports/                   # rapor ve şekiller
```

## Ekip

| Rol | Sorumluluk |
|---|---|
| Öğrenci 1 | Veri seti, literatür, temizleme |
| Öğrenci 2 | Baseline ve zaman serisi deneyleri |
| Öğrenci 3 | ML modelleri ve stok kuralları |
| Öğrenci 4 | Arayüz, entegrasyon ve test |

## Çalışma kuralları
- Her iş bir Issue → `feat/<ad>` dalı → Pull Request → 1 onay → `main`.
- CI her PR'da `ruff` ve `pytest` çalıştırır.
