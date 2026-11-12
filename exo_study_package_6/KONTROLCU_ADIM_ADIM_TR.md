# YENİ KONTROLCÜ: yük algılayıcı + yük farkındalıklı dağıtıcı + QP güvenlik filtresi

Bu dosya, `ADIM_ADIM_TR.md` içindeki Adım 1-4 bittikten sonra yapılacakları anlatır
(en az: `ft_none_p20` ve `ft_demand_p20` eğitilmiş, `work/demand.json` ve `work/cfg/strategies.json` hazır).
Tüm komutlar depo kökünde, `PYTHONPATH=.` ile çalışır. `P=scripts` varsayılır (gerekirse kendi yolunu yaz).

## Kontrolcü ne yapar? (3 katman, basit anlatım)
1. Algılama: Son ~1.3 saniyedeki yer tepki kuvveti, eklem hızları, eklem hareket aralıkları ve biyolojik eklem momentlerinden
   küçük bir sinir ağı taşınan yükü (kg) tahmin eder. Yük yürürken değişirse tahmin birkaç saniyede güncellenir.
2. Dağıtım: Tahmin edilen yüke göre toplam yardım miktarı (ρ) ve kalça / diz / ayak bileği arasındaki pay belirlenir.
   Varsayılan ayar = sabit "talebe orantılı" strateji. Az sayıda parametre (7 adet) ES ile optimize edilebilir.
3. Güvenlik (QP filtresi): İstenen tork, şu kısıtlara en yakın izinli torka dönüştürülür:
   aktüatör sınırı, adım başına tork değişim sınırı, eklem açı sınırı yakını, toplam güç bütçesi.
   Ayrıca eklem kısıt kuvveti göstergesi (V_c) eşiğe yaklaşırsa yardım küçültülür.
- Tüm kısıt değerleri VARSAYIMDIR (tasarım kararı), ölçülmüş fiziksel sınır değildir; makalede böyle yazılır.

## Adım K0 - Yamayı yeniden kur ve QP testini çalıştır (1-2 dk)
```
python scripts/install_patch.py --repo .
python scripts/test_learned_qp.py --session <herhangi bir oturum klasörü>
```
Beklenen: "Test 1 PASS" ve "Test 2 PASS".

## Adım K1 - Yük algılayıcı verisi (exo'suz politika, rastgele yük 0-30 kg)
Önce küçük dene: `--n_rollouts 16 --workers 4`. Süre bu bilgisayarda ölçülmedi; çıktıdaki "samples" sayısına bak.
```
N=$(python -c "import json;print(json.load(open('work/sessions.json'))['ft_none_p20'])")
python scripts/collect_estimator_data.py --session $N --n_rollouts 160 --workers 8 --out work/est_data.npz
```
- Uyarı çıkarsa ("few samples") politika bazı yüklerde erken düşüyordur; `--hi 25` ile dene.

## Adım K2 - Yük algılayıcıyı eğit (dakikalar)
```
python scripts/train_estimator.py --data work/est_data.npz --out work/est.npz
```
- Çıktıdaki RMSE (kg) makalede "algılama hatası" olarak raporlanır; sabit tahmin ve ridge baz çizgileriyle karşılaştırılır.

## Adım K3 - Kısıt değerlerini kalibre et (ASSUMED)
```
python scripts/calibrate_limits.py --strategies work/cfg/strategies.json --registry work/sessions.json --payload_kg 20 --out work/limits.json
```
- Çıktıda "WARNING" varsa kalibrasyon yürüyüşü düşmüştür; değerler anlamlı değildir.

## Adım K4 (opsiyonel) - Dağıtıcı parametrelerini optimize et
```
python scripts/optimize_allocator.py --session $N --demand work/demand.json --limits work/limits.json --out work/alloc.json --payloads 10 20 30 --n_ind 4 --gens 12 --pairs 6 --workers 8
```
- "NOTE: no clear gain" yazarsa dürüstçe "optimizasyon talebe orantılı varsayılandan belirgin kazanç sağlamadı" diye raporlanır; bu kabul edilebilir bir sonuçtur.
- Bu adımı atlarsan `--alloc` parametresini K5'te verme.

## Adım K5 - Kontrolcü döngüdeyken politikayı ince ayarla
```
MN=$(ls -t $N/trained_models/model_*.zip | head -1)
python scripts/make_learned_config.py --estimator work/est.npz --limits work/limits.json --demand work/demand.json [--alloc work/alloc.json] --warm_start "$MN" --payload_kg 20 --out work/ft_learned_p20.json
python scripts/train_named.py --name ft_learned_p20 --config work/ft_learned_p20.json --registry work/sessions.json
```
- Eğitim sırasında yük 20 kg sabit; yük basamağı yalnız değerlendirmede uygulanır (bu bir sınırlılıktır, makalede yazılır).

## Adım K6 - Yük basamağı değerlendirmesi (ana sonuç)
```
python scripts/eval_learned.py --registry work/sessions.json --strategies work/cfg/strategies.json --learned_assist work/ft_learned_p20_assist.json --steps_list 10:25 25:10 --n_ind 8 --workers 8 --out work/eval_learned.csv
```
- Karşılaştırılan 5 kontrolcü: none, demand (sabit), learned_nofilter, learned_qp, learned_oracle (gerçek yük verilir; algılama hatasının üst sınırı).
- learned_* üçü de `ft_learned_p20` politikasını kullanır; fark yalnız çıkarım sırasındaki ayardadır (filtre kapalı / gerçek yük). Bu bir ablasyondur.
- Ekrana yazılan tablo: düşme sayısı, yük tahmin hatası (kg), yerleşme süresi (s), yardımın etkisi (yük basamağından önce/sonra) ve V_c ihlal oranı.

## Bana gönder
Her adımın son 8 satırı; `work/limits.json`, `work/alloc.json` (+`_log.json`), `work/eval_learned.csv` ve K6 ekran çıktısı.

## Sınırlılıklar (kodda ve makalede açıkça yazılacak)
- Kolsuz, sabit gövdeli alt ekstremite modeli; ideal aktüatör; E_act metabolik enerji değildir.
- Biyolojik eklem momenti (tau_bio) ölçülebilir kabul edilir (simülasyonda mevcut); gerçek cihazda tahmini gerekir.
- Denge göstergesi V_c bir vekildir (eklem kısıt kuvveti); doğrulanmış bir denge ölçüsü değildir.
- Sonuçlar bu modele ve parametre aralığına koşulludur; donanımda doğrulanmamıştır.
