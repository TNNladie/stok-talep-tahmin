# Küçük İşletmeler İçin Yapay Zekâ Destekli Stok Talep Tahmin Sistemi

Geçmiş satış verilerinden ürün bazında **28 günlük talep tahmini** üreten ve bunu
**güvenlik stoku + yeniden sipariş noktası** önerisine çeviren karar destek sistemi.
Bitirme projesi — 4 kişilik ekip, 10 haftalık sprint.

## Hızlı başlangıç

```bash
git clone https://github.com/<kullanici>/stok-talep-tahmin.git
cd stok-talep-tahmin
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
make test          # birim testleri
make demo          # internetsiz sentetik veriyle tüm akış
make data run      # Online Retail II indir + gerçek veriyle akış
make app           # Streamlit karar destek ekranı
```

## Akış

```
Ham satış verisi → temizleme (iade/negatif at, günlük topla, eksik gün = 0)
 → öznitelik (lag 28–56, rolling, takvim, tatil, fiyat)
 → modeller (Persistence, Mevsimsel Naive, MA-7/28 | RandomForest, XGBoost, LightGBM)
 → zaman bazlı backtest (3 kat × 28 gün) → MAE / RMSE / MAPE / WAPE
 → en iyi model → SS = z·σ·√L, ROP = D_L + SS → sipariş önerisi → Streamlit ekranı
```

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
