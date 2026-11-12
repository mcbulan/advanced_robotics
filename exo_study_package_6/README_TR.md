# Tam vücut exoskeleton: yardım tork dağıtım stratejilerinin sıralama kararlılığı (simülasyon paketi)

Bu paket tamamen **simülasyon** çalışması içindir (MyoSuite / MyoAssist, MuJoCo). Gerçek insan veya donanım verisi yoktur. Aşağıdaki hiçbir sonuç "gerçek dünyada geçerlidir" diye okunmamalıdır; yalnızca modele ve taranan parametre aralığına koşulludur ve doğrulanmamıştır.

## 1. Çalışma sorusu (tek cümle)

Sırt yükü altında yürürken, kalça / diz / ayak bileği arasındaki **yardım tork dağıtım stratejilerinin göreli sıralaması**, model parametresi belirsizliği (sanal bireyler) altında ne kadar kararlıdır?

Kapsam: yalnız düz zeminde yürüyüş + sırt yükü (varsayılan 20 kg; 10-30 kg taranabilir). Merdiven, eğim, koşu, kaldırma: gelecek çalışma. Bu yüzden kendi mini testindeki `stairs` arazisi bu çalışmada **kullanılmaz** (arazi = düz zemin).

## 2. MyoSuite / MyoAssist nedir (OpenSim'den gelenler için kısa)

- **MuJoCo**: fizik motoru. **MyoSuite**: MuJoCo üzerinde kas-iskelet modelleri (OpenSim'deki .osim karşılığı; kaslar MuJoCo kas aktüatörleri). **MyoAssist**: MyoSuite üzerine exoskeleton cihazları + PPO ile eğitim hattı.
- Kullandığın JSON bir **EnvSpec**: `msk` (kas-iskelet modeli) + `device` (cihaz) + `terrain` (arazi). Yani "ortamın modeli" tanımıdır; eğitim ayarı (ödül, ağ, adım sayısı) değildir. Eğitim ayarı ayrı bir **train config** JSON'udur (bu pakette `make_configs.py` üretir).
- Bu çalışmada `msk = myolegs22` (22 kas, kalça/diz/ayak bileği; gövde yok) + `device = Hippo_L1` kullanılır, çünkü MyoAssist'te yürüyüş eğitimi için hazır, test edilmiş config bu kombinasyonun. `myolegs26` ve `Tutorial_L1` bu çalışma için gerekmez (Tutorial_L1 pasif cihazdır).
- Dürüst sınır: `myolegs22`'de gövde/lumbar eklemi yok. "Tam vücut" iddiası ancak `myofullbody` (416 kas) eğitimi başarılırsa ve gövde kanalları eklenirse yapılabilir; bu pakette `make_configs.py --msk myofullbody` yalnız boru hattı (ortam kurulumu) olarak desteklenir, yürüyüş öğrenmesi doğrulanmamıştır.

## 3. Yöntem (baştan sona)

1. **Taban yürüyüş politikası**: yardımsız (cihaz kuvveti sıfır) insan politikası, `myolegs22`, 0 kg ve 20 kg sırt yükünde PPO ile eğitilir. Yük, gövdeye eklenen nokta kütledir (konum VARSAYIM).
2. **Sanal popülasyon**: Latin hypercube ile N birey; parametre aralıkları (`scripts/population.py`): kütle ölçeği 0.85-1.15, segment kütle sapması ±0.10, kas kuvveti ölçeği 0.75-1.25, uzunluk kayması ±0.05, sürtünme 0.5-1.0, cihaz gecikmesi 0-40 ms, cihaz tork ölçeği 0.75-1.0. Aralıklar literatürden seçilmiş varsayımlardır; makalede ayrıca gerekçelendirilecek.
3. **Yardım katmanı (overlay)**: 6 eklemde (kalça/diz/ayak bileği, sağ/sol) ideal eklem torku. Yasa: `tau_exo = clip(alpha_j * LP(tau_bio), +-umax * tau_scale)`; yani insanın ölçülebilir eklem momentiyle orantılı yardım. Stok MyoAssist cihazları tek eklemli olduğu için eklemler arası "dağıtım" böyle kuruldu. Aktüatör ideal ve kütlesizdir; arayüz esnekliği yoktur. Hippo cihazının kendi eylemi sıfırlanır (sıfır eylem = sıfır kuvvet, `test_assist.py` ile doğrulanır).
4. **Talep ölçümü** (`measure_demand.py`): yardımsız taban politikasıyla D_hip, D_knee, D_ankle (ortalama |eklem momenti|).
5. **Stratejiler** (`make_strategy_configs.py`): aynı toplam yardım bütçesi `sum(alpha_g * D_g) = rho * sum(D_g)` ile: `none`, `hip`, `knee`, `ankle`, `uniform` (bütçe 1/3'er), `demand` (bütçe talep oranında). alpha üst sınırı 1.0.
6. **İnce ayar**: her strateji için insan politikası, overlay açıkken taban politikadan **warm-start** ile kısa süre eğitilir (`none` de aynı ince ayardan geçer; adil karşılaştırma için).
7. **Değerlendirme** (`eval_strategies.py`): strateji x yük x (nominal + N sanal birey), aynı bireyler tüm stratejilerde. Çıktı: düşme, hayatta kalma süresi, hız, `E_act` (ort. kare aktivasyon; metabolik enerji DEĞİL, vekil), `R_bio` (biyolojik moment yükü), `P_exo` vb.
8. **Sıralama kararlılığı** (`analyze_ranking.py`): bireye göre strateji sıralaması, ortalama sıra, rank-1 payı, nominal ve bireyler arası Kendall tau_b, `none`'a karşı eşleştirilmiş fark (bootstrap %95 GA, Wilcoxon + Holm). Düşenler sıralamada en sona atılır.

## 4. Kurulum (macOS, PyCharm)

Senin kurulumun zaten var; kontrol listesi:

```bash
cd <myoassist deposunun kökü>            # PyCharm terminali; rl_train/ klasörünün olduğu yer
source <venv>/bin/activate               # PyCharm'ın kullandığı venv
export PYTHONPATH=.                      # her yeni terminalde
export MYOASSIST_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
git log -1 --oneline                     # sürümü not et (bu paket commit 629fc3f ile test edildi)
python -c "import myosuite, mujoco; print(myosuite.__version__, mujoco.__version__)"
pip install pandas scipy                 # analiz için (yoksa)
```

- `OPENBLAS_NUM_THREADS=1` satırını atlama: bazı makinelerde scipy/numpy içe aktarımı (analiz betiği dahil) bu olmadan çok yavaşlayabiliyor.
- Hepsi CPU'da çalışır. Resmi config 32 ortam / 30M adım ister; MacBook'ta `--num_envs 8` (veya 10) ve aşağıdaki adım sayıları kullanılır.
- Tüm komutlar **depo kökünden** çalıştırılır. `PYTHONPATH=.` unutulursa `rl_train` import hatası alırsın.
- Paketi depo içine aç: `exo_study_package/` klasörünü depo köküne koy (örn. `P=exo_study_package/scripts`).

## 5. Adım adım komutlar

Önce yama (idempotent; ikinci çalıştırmada "already patched" der, yedekler `*.orig_study`):

```bash
P=exo_study_package/scripts
python $P/install_patch.py --repo .
```
Anchor bulunamadı hatası alırsan (sürüm farkı): çıktıdaki elle ekleme talimatını izle ve bana hata metnini gönder.

### Gate 0: kurulum testleri (dakikalar)

```bash
python $P/test_payload.py        # kütle/atalet korunumu kontrolü: err değerleri ~0 olmalı
mkdir -p work
python $P/make_configs.py --msk myolegs22 --out work/tiny.json --payload_kg 20 --num_envs 2 --timesteps 3000 --n_steps 256 --batch 256 --epochs 1
python $P/train_named.py --name tiny --config work/tiny.json --registry work/sessions.json
S=$(python -c "import json;print(json.load(open('work/sessions.json'))['tiny'])")
python $P/test_assist.py --session $S     # PASS beklenir (sıfır eylem = sıfır kuvvet; overlay torku uygulanıyor)
```
Geçiş: üç betik hatasız biter, `test_assist` PASS yazar. (Bu tiny eğitim yürümeyi öğretmez; yalnız boru hattı testidir.)

### Gate 1: taban yürüyüş (asıl uzun işlem)

```bash
python $P/make_configs.py --msk myolegs22 --out work/base_p0.json  --payload_kg 0  --num_envs 8 --timesteps 5000000
python $P/make_configs.py --msk myolegs22 --out work/base_p20.json --payload_kg 20 --num_envs 8 --timesteps 5000000
python $P/train_named.py --name base_p0  --config work/base_p0.json  --registry work/sessions.json
python $P/train_named.py --name base_p20 --config work/base_p20.json --registry work/sessions.json
```
- `--timesteps` başlangıç önerisidir (VARSAYIM); resmi config 30M kullanır. Öğrenme yetmezse artır. Süre makineye bağlı; ilk 100-200k adımda `train_log.json`'daki ödülün yükseldiğine bak, sonra uzun koşuyu bırak.
- İpucu: 0 kg politikasından 20 kg'a geçmek istersen `make_configs.py --warm_start <model.zip>` kullanılabilir.
- **Gate 1 kontrol betiği** (eğitim bittikten sonra, depo kökünden):
  ```bash
  S0=$(python -c "import json;print(json.load(open('work/sessions.json'))['base_p0'])")
  S20=$(python -c "import json;print(json.load(open('work/sessions.json'))['base_p20'])")
  python $P/check_gate1.py --session $S0  --payload_kg 0  --steps 600
  python $P/check_gate1.py --session $S20 --payload_kg 20 --steps 600 --n_pop 3
  ```
  PASS ölçütü (varsayım): düşmeden >=10 sn ve ortalama hız hedefin +-0.35 m/s içinde. Yürüyüş örüntüsünün kalitesini (adım simetrisi, eklem açıları) ölçmez; onun için `train_log.json` ve gerekirse video/ekran görüntüsü de bakılır.
- **Geçiş ölçütü (Gate 1)**: 20 kg'da değerlendirme rollout'unda ≥10 sn düşmeden, hedef hıza (1.25 m/s) yakın yürüyüş. Sağlanmazsa strateji aşamasına geçme; `train_log.json` + `session_config.json` gönder, birlikte ayarlarız (ödül ağırlıkları, lr, adım sayısı).

### Gate 2: talep ölçümü ve strateji ince ayarı

```bash
S20=$(python -c "import json;print(json.load(open('work/sessions.json'))['base_p20'])")
python $P/measure_demand.py --session $S20 --payload_kg 20 --out work/demand.json
```
`demand.json`'daki D değerleri fiziksel olarak makul olmalı (tipik yürüyüşte birkaç on Nm mertebesi; <8 sn yürüme uyarısı çıkarsa taban politika yetersizdir, Gate 1'e dön).

```bash
M=$(ls -t $S20/trained_models/*.zip | head -1)      # en son kaydedilen model
python $P/make_strategy_configs.py --demand work/demand.json --rho 0.3 --warm_start $M \
       --payload_kg 20 --timesteps 3000000 --num_envs 8 --lr 5e-5 --umax 60 --outdir work/cfg
for n in none hip knee ankle uniform demand; do
  python $P/train_named.py --name ft_${n}_p20 --config work/cfg/ft_${n}_p20.json --registry work/sessions.json
done
```
- Her strateji ayrı bir koşudur (6 koşu). Önce `--timesteps 300000` ile tüm hattı dene, sonra uzat.
- `strategies.json` gerçekleşen bütçeyi (`budget_rho_realised`) gösterir; alpha kırpılırsa bütçe hedefin altında kalır (kontrol et).
- **Geçiş ölçütü (Gate 2)**: ince ayardan sonra `none` dahil tüm stratejiler yürümeyi sürdürmeli (düşme oranı düşük). Yoksa lr'yi düşür veya süreyi artır.

### Gate 3: sanal popülasyonda değerlendirme ve sıralama

```bash
python $P/eval_strategies.py --strategies work/cfg/strategies.json --registry work/sessions.json \
       --payloads 20 --n_pop 16 --pop_seed 0 --out work/results.csv
python $P/analyze_ranking.py --csv work/results.csv --metric E_act --out work/summary.md
```
- İlk denemede `--n_pop 4 --steps 100` ile hızlı çalıştır; sonra `--n_pop 16-32`, `--steps 300` (30 Hz'de 10 sn).
- Sıralama kararlılığı için birey sayısı kritik; ≥16 önerilir. Makalede Δ ve GA ile birlikte raporlanır.

## 5b. Yeni kontrolcü (öğrenen yük algılayıcı + dağıtıcı + QP filtre)
Ayrıntılı adımlar: `KONTROLCU_ADIM_ADIM_TR.md`. Yeni dosyalar: `patch_files/learned_controller.py`, `scripts/{test_learned_qp,collect_estimator_data,train_estimator,calibrate_limits,optimize_allocator,make_learned_config,eval_learned}.py`. Yama güncellendiği için `python scripts/install_patch.py --repo .` yeniden çalıştırılmalıdır.

## 6. Ayarlanacak parametreler (tuning)

| parametre | nerede | not |
|---|---|---|
| `--rho` (0.1-0.5) | make_strategy_configs | yardım bütçesi oranı; düşük rho'da farklar küçülür |
| `--umax` (Nm) | make_strategy_configs | tork sınırı, VARSAYIM 60 |
| `--lr`, `--timesteps` | make_configs / make_strategy_configs | ince ayar lr 5e-5 |
| `--num_envs`, `--n_steps`, `--batch`, `--epochs` | make_configs | Mac için num_envs 8-10 |
| `--payload_kg`, `--payload_mode` | make_configs / eval | back (ana), front (ikincil) |
| `delay_ms`, `exo_tau_scale` aralıkları | population.py `RANGES` | gerekçeyle değiştir |
| `--metric` | analyze_ranking | E_act, R_bio, V_c, P_exo |

## 7. Bana göndereceğin dosyalar

- Her koşu için: `<oturum>/train_log.json`, `<oturum>/session_config.json`, (gerekirse) `trained_models/*.zip`
- `work/sessions.json`, `work/demand.json`, `work/cfg/strategies.json`
- `work/results.csv`, `work/summary.md`
- Hata olursa: komut + terminalin son ~40 satırı + `git log -1 --oneline` + `pip show myosuite`

## 8. Bilinen riskler ve doğrulanmamış noktalar

- Yürüyüş öğrenmesi bu paketle **hiçbir konfigürasyonda doğrulanmadı**; uçtan uca boru hattı yalnız minik koşularla (hepsi düştü, beklenen) denendi.
- Warm-start dalı düz PPO yükler (mirror kaybı warm-start'ta uygulanmaz).
- Yük konumu, umax, popülasyon aralıkları ve yasa ASSUMED; makalede açıkça "varsayım" denmeli.
- `E_act` metabolik enerji değildir; sonuçlar "aktivasyon vekili" olarak raporlanır.
- Overlay ideal aktüatördür; gecikme overlay komutuna uygulanır, insan eylemine değil.
- Senin sürümünde `myolegs26` var; commit farkı varsa `install_patch.py` anchor bulamayabilir.
- Mac'te MuJoCo render gerekmiyor; ekran/GL hatası alırsan eğitim komutlarında render kapalı olmalı.

## Gate 1 başarısız olursa: eğitimi uzatma (warm-start)
- `train_log.json` içindeki `average_num_timestep` değeri **kontrol adımı** cinsindendir (30 Hz; 1000 adım ≈ 33 s). Gate 1 için eğitimde bu değerin yüzlerce adıma (≥300 ≈ 10 s) çıkması beklenir.
- Devam eğitimi (ağırlıklar `base_p20` son modelinden yüklenir; mirror kaybı warm-start dalında yoktur):
```
M=$(ls -t <base_p20 oturumu>/trained_models/model_*.zip | head -1)
python $P/make_configs.py --msk myolegs22 --out work/base_p20_c1.json --payload_kg 20 --num_envs 8 --timesteps 20000000 --n_steps 256 --batch 2048 --warm_start "$M" --lr 1e-4
python $P/train_named.py --name base_p20_c1 --config work/base_p20_c1.json --registry work/sessions.json
```
- Not: `n_steps 256 / batch 2048` (güncelleme sayısını artırmak için) ASSUMED bir ayardır; doğrulanmamıştır.
