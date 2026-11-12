> GÜNCEL PLAN: `ADIM_ADIM_TR.md` dosyasını kullan. Aşağıdaki Gate 1 anlatımı eskidir (exo kapalı test ediyordu).

# BURADAN BAŞLA (sade anlatım)

## 0. Ne yapıyoruz? (tek paragraf)
Sanal bir insan modeli bilgisayarda yürüyor. Sırtında yük var. Önce exoskeleton OLMADAN yürümeyi, sonra exoskeleton GİYİP yürümeyi öğreniyor. Exo'nun yardım torkunu eklemlere farklı yollarla (6 strateji) dağıtıp hangisinin daha iyi olduğuna bakıyoruz. Sonra sanal kişiyi değiştiriyoruz (daha ağır/hafif, kaslar daha güçlü/zayıf). Soru: "en iyi strateji kişi değişince de aynı mı kalıyor?"

## 1. Kelimeler (6 satır)
- **MuJoCo**: fizik motoru (sanal dünyayı hesaplar).
- **MyoSuite / MyoAssist**: içinde kaslı insan modelleri ve eğitim kodu olan paketler.
- **Politika**: yürümeyi öğrenen yapay sinir ağı (eğitimle oluşan model dosyası: `model_*.zip`).
- **Eğitim (train)**: modelin deneme-yanılmayla yürümeyi öğrenmesi. Uzun sürer.
- **Oturum (session)**: bir eğitimin klasörü (`rl_train/results/train_session_...`).
- **Gate (kapı)**: "sıradaki adıma geçmeden önce şunu kontrol et" testi. **Gate 1** = yardımsız model en az 10 sn düşmeden yürüyor mu? Geçmezse strateji karşılaştırmasının anlamı olmaz, çünkü karşılaştırılacak sağlam bir yürüyüş yoktur.

## 2. Kollar ve tam vücut neden yok? Durum ne?
- Şimdiye kadar kullandığımız model (`myolegs22`) yalnız bacak kaslarına sahip; gövde sabit, kol eklemi yok. Bunu, tam vücut modeli çok daha ağır olduğu için, önce "yürüyüş öğreniliyor mu?" sorusunu ucuza test etmek amacıyla seçmiştik. Bunu sana net anlatmamam hataydı; asıl hedefin tam vücut.
- Tam vücut modeli (`myofullbody`) pakette çalışıyor. Sandbox ölçümü: 416 kas + 2 exo = 418 eylem, 438 gözlem, tek ortamda ≈39 adım/sn. Kol eklemleri var (omuz ×3, dirsek). Gövde (lumbar) eklemleri de var.
- Riskler (dürüst):
  - Bacak modeli bile 5M adımda henüz yürümedi. 416 kaslı modelin sıfırdan yürümeyi öğrenmesi çok daha uzun sürer; süreyi doğrulamadım, tahminen günler.
  - Eldeki referans yürüyüş yalnız bacak eklemlerini içerir. Kollar için referans yok; kol sallanması kendiliğinden öğrenilmez.
  - "Şişman-zayıf" sanal kişi = vücut kütlesi çarpanı (şu an ±%15, kas gücü ±%25 vb. hazır). "Uzun-kısa" (boy ölçekleme) şu an kodda YOK; eklenmesi ayrı iştir.
- Önerilen aşamalar (her aşama öncekinin geçmesine bağlı):
  - **A. Düz zemin, bacak modeli, 20 kg yük, yardımsız**: yürümeyi öğret. (ŞU AN)
  - **B. Aynı model, 8° yokuş**: yardımsız → exo'lu, 6 strateji, birden çok sanal kişi. Yokuş desteği hazır ve sandbox'ta doğrulandı (`--slope_deg 8`, yukarı doğru).
  - **C. Kollu tam vücut**: önce yalnız fizibilite testi (aşağıda, 2 dk). Sonucuna göre kolların eklenip eklenmeyeceğine birlikte karar veririz. Makalede şimdilik "alt ekstremite kas-iskelet modeli + sabit gövde" diye yazılır.
- Kapsam notu: yokuş daha önce "gelecek çalışma"ydı; senin isteğinle B aşamasına aldık.

## 3. ÖNCE TEMİZLİK (güvenli)
Eski eğitim klasörleri çok yer kaplar. Eski zip'leri ve açılmış eski paket klasörlerini sil; yalnız en son `exo_study_package.zip` kalsın. Sonra:
```
cd <MyoAssist klasörü>
P=exo_study_package/scripts
python $P/cleanup.py --keep base_p20 --registry work/sessions.json          # önce sadece listeler
python $P/cleanup.py --keep base_p20 --registry work/sessions.json --delete # sonra gerçekten siler
```
- `--keep` yanına tutmak istediğin oturum adlarını yaz (devam eğitimi koştuysan `base_p20_c1`'i de ekle).
- Devam eden bir eğitim varsa bitmeden silme.
- `tiny` ve `base_p0` oturumları artık gereksiz; silinir.

## 4. ŞİMDİ YAPILACAK (sırayla)
1. Devam eğitimi (henüz başlatmadıysan; yaklaşık 3,5 saat):
```
M=$(ls -t <base_p20 oturum klasörü>/trained_models/model_*.zip | head -1)
python $P/make_configs.py --msk myolegs22 --out work/base_p20_c1.json --payload_kg 20 --num_envs 8 --timesteps 20000000 --n_steps 256 --batch 2048 --warm_start "$M" --lr 1e-4
python $P/train_named.py --name base_p20_c1 --config work/base_p20_c1.json --registry work/sessions.json
```
2. Bitince Gate 1:
```
export OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
python $P/check_gate1.py --session $(python -c "import json;print(json.load(open('work/sessions.json'))['base_p20_c1'])") --payload_kg 20 --n_pop 3 2>&1 | grep -E "model:|nominal|ind|GATE"
```
   "PASS" görürsen Gate 1 geçildi.
3. İsteğe bağlı, bugün yapılabilir (2 dk, eğitim yok) – tam vücut fizibilite testi:
```
python $P/make_configs.py --msk myofullbody --out work/fb_probe.json --payload_kg 20 --num_envs 1 --timesteps 1000
python $P/fullbody_probe.py --config work/fb_probe.json 2>&1 | tail -4
```

## 5. Bana ne göndereceksin?
- Adım 2'nin çıktısı (grep'li kısım) ve `train_log.json` (yeni oturumdan).
- Adım 3'ün çıktısı (Mac'indeki adım/sn değerini görmek için).
- Gate 1 geçmezse başka hiçbir şey çalıştırma; ben ayarı değiştiririm.

## 6. Sonuçlar makaleye nasıl girer?
- Çalışan son ayarlar → Yöntem. Başarısız denemeler makaleye girmez.
- Gate 1 (düşmeden yürüme) → "model doğrulaması" tablosu/cümlesi.
- Strateji sıralaması ve sanal kişilerde kararlılığı → Bulgular (tablo + şekil).
- Kolsuz/sabit gövde, E_act'ın metabolik enerji OLMAMASI, varsayılan aralıklar → Sınırlılıklar. Tüm sonuçlar "bu model ve parametre aralığına koşulludur, doğrulanmamıştır" diye yazılır.

## 7. Yokuş komutu (Gate 1 geçtikten sonra, şimdi çalıştırma)
```
python $P/make_configs.py --msk myolegs22 --out work/slope_p20.json --payload_kg 20 --slope_deg 8 --num_envs 8 --timesteps 10000000 --warm_start "$M" --lr 1e-4
```
