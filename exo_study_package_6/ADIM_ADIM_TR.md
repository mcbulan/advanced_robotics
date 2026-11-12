# ADIM ADIM (güncel plan) - önce bunu oku

## Ne değişti?
- Gate 1 testi, modeli "exo kapalı" çalıştırıyordu. Ama eğitimde politika exo'yu da kendisi sürüyordu (exo AÇIK). Bu yüzden testte düştü, videoda yürüdü. Çelişki buradan.
- Düzeltme: `check_gate1.py` artık varsayılan olarak eğitimdeki gibi (exo AÇIK) test eder. `--device off` eski davranıştır (düşmesi beklenir).
- Karar: Gate 1 sonucu yeni testle PASS çıkarsa geçilmiş sayılır. PASS çıkmazsa videodaki yürüyüşün kısa/yeniden başlatmalı olabileceğini düşünürüz.

## Kapsam (öneri, makale için yeterli)
- Model: myolegs22 (bacak kasları, sabit gövde, kolsuz). Tam vücut denemesi YOK; makalede "alt ekstremite kas-iskelet modeli" diye yazılır. Sebep: 416 kaslı modelin eğitimi çok uzun sürer (süre doğrulanmadı).
- Sanal kişiler: yalnız şişman-zayıf (kütle ve kas gücü değişimi). Uzun-kısa yok.
- Senaryolar: yüklü (20 kg) / yüksüz (0 kg), exo'suz / exo'lu (6 strateji), düz zemin ve 8° yokuş. Merdiven yok (gelecek çalışma).
- Not: Yüksüz senaryo, 20 kg ile eğitilmiş politikayla yük 0 kg yapılarak değerlendirilir (yük dışı eğitim yok). Bu kabul makalede yazılır.

## Adım 1 - Gate 1'i yeniden çalıştır (2 dk)
```
cd <MyoAssist klasörü>; source <venv>/bin/activate
P=<paket>/scripts; export OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
S=$(python -c "import json;print(json.load(open('work/sessions.json'))['base_p20'])")
python $P/check_gate1.py --session $S --payload_kg 20 --steps 600 --n_pop 3 2>&1 | grep -E "model:|nominal|ind|GATE"
```
PASS ise Adım 2'ye geç. Çıktıyı bana gönder.

## Adım 2 - Exo'suz politika (düz, 20 kg) ~30 dk
Exo gücü sıfır, politika yardımsız yürümeyi öğrenir:
```
M=$(ls -t $S/trained_models/model_*.zip | head -1)
python $P/make_configs.py --msk myolegs22 --out work/ft_none_p20.json --payload_kg 20 --num_envs 8 --timesteps 3000000 --lr 5e-5 --warm_start "$M" --assist_json '{"alpha":{"hip":0,"knee":0,"ankle":0}}'
python $P/train_named.py --name ft_none_p20 --config work/ft_none_p20.json --registry work/sessions.json
```
Bitince: `python $P/check_gate1.py --session <ft_none_p20 oturumu> --payload_kg 20 --device off | grep -E "nominal|GATE"` (exo kapalı yürüyor mu?).

## Adım 3 - Eklem yük talebi ölçümü (1 dk)
```
N=$(python -c "import json;print(json.load(open('work/sessions.json'))['ft_none_p20'])")
python $P/measure_demand.py --session $N --payload_kg 20 --out work/demand.json
```

## Adım 4 - 5 yardım stratejisini eğit (her biri ~30 dk)
```
MN=$(ls -t $N/trained_models/model_*.zip | head -1)
python $P/make_strategy_configs.py --demand work/demand.json --rho 0.3 --warm_start "$MN" --payload_kg 20 --outdir work/cfg
for s in hip knee ankle uniform demand; do python $P/train_named.py --name ft_${s}_p20 --config work/cfg/ft_${s}_p20.json --registry work/sessions.json; done
```
(`none` zaten Adım 2'de eğitildi; tekrar eğitme.)

## Adım 5 - Değerlendirme ve sıralama (yüklü/yüksüz, şişman-zayıf)
```
python $P/eval_strategies.py --strategies work/cfg/strategies.json --registry work/sessions.json --reg_suffix _p20 --payloads 0 20 --n_pop 16 --out work/results_flat.csv
python $P/analyze_ranking.py --csv work/results_flat.csv --out work/summary_flat.md
```

## Adım 6 - Yokuş (8°) - Adım 2-5'in aynısı, `--slope_deg 8` ile
```
python $P/make_configs.py --msk myolegs22 --out work/ft_none_p20_s8.json --payload_kg 20 --slope_deg 8 --num_envs 8 --timesteps 5000000 --lr 5e-5 --warm_start "$MN" --assist_json '{"alpha":{"hip":0,"knee":0,"ankle":0}}'
python $P/train_named.py --name ft_none_p20_s8 --config work/ft_none_p20_s8.json --registry work/sessions.json
# sonra measure_demand (yeni oturumla), make_strategy_configs ... --slope_deg 8 --outdir work/cfg_s8, 5 strateji eğitimi, eval_strategies --reg_suffix _p20_s8
```
Yokuşta referans yürüyüş düzdür; yokuşta politika düşerse bunu "yokuş için ek eğitim gerekir" diye raporlarız.

## Adım 7 - Yeni kontrolcü (yük algılayıcı + dağıtıcı + QP filtre)
Adım 2-4 bittikten sonra `KONTROLCU_ADIM_ADIM_TR.md` dosyasındaki K0-K6 adımlarını uygula (önce `python scripts/install_patch.py --repo .` tekrar çalıştırılmalı).

## Bana gönder
Adım 1 çıktısı; sonra her adımın son 5 satırı ve `results_flat.csv` + `summary_flat.md`.

## Makalede bu çalışma ne gösterir?
- Yapay zeka (PPO politikası) + exo yardım dağılımı: yük altında yürüyüşün sürdüğü, yardımın eklemlere nasıl dağıldığı, biyolojik eklem momentinin ve aktivasyon göstergesinin (E_act, metabolik enerji değildir) stratejiye göre değişimi.
- Sonuçlar bu model ve parametre aralığına koşulludur; gerçek insan verisiyle doğrulanmamıştır.
