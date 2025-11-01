// Developer: inkbytefo
// AI: Trae Coding Assistant
// Modified: 2025-11-01

# 🚀 TMA-1 Quick Start - MorphoPiece Tokenizer Eğitimi

## Hızlı Başlangıç

```bash
# Küçük veriyle hızlı kurulum
python scripts/make_test_corpus.py

# Adım adım
python src/train_morphopiece.py --preprocess   --corpus-file data/test_corpus.txt --preprocessed-file data/corpus_morpho_processed.txt
python src/train_morphopiece.py --train        --preprocessed-file data/corpus_morpho_processed.txt --output tokenizer/morphopiece --vocab-size 1000
```

## Detaylı Kullanım

### 1. Veri İndirme (1.5 GB)

```bash
# Opsiyonel: Büyük corpus indirme (internet ve disk gerektirir)
python src/train_morphopiece.py --download --corpus-file data/corpus_combined.txt
```

**Çıktı:**
- `data/mc4_turkish.txt` (0.75 GB)
- `data/wikipedia_turkish.txt` (0.75 GB)
- `data/corpus_combined.txt` (1.5 GB)

### 2. Morfem Ayrımı ile Ön İşleme

```bash
# Morfem ayrımı (Zemberek varsa Java ile, yoksa regex fallback)
python src/train_morphopiece.py --preprocess --corpus-file data/corpus_combined.txt --preprocessed-file data/corpus_morpho_processed.txt
```

**İşlem:**
- Her kelime → kök + ekler
- Örnek: "Evlerimdekiler" → "ev ler im de ki ler"

**Çıktı:**
- `data/corpus_morpho_processed.txt` (morfem ayrımı yapılmış)

### 3. MorphoPiece Tokenizer Eğitimi

```bash
# SentencePiece ile tokenizer eğit
python src/train_morphopiece.py --train --preprocessed-file data/corpus_morpho_processed.txt --output tokenizer/morphopiece --vocab-size 32000 --model-type unigram --character-coverage 1.0
```

**Parametreler:**
- `vocab_size=32000`: Vocabulary boyutu
- `model_type='unigram'`: Unigram modeli
- `character_coverage=1.0`: Tam karakter kapsamı

**Çıktı:**
- `tokenizer/morphopiece.model` - SentencePiece model
- `tokenizer/morphopiece.vocab` - Vocabulary dosyası
- `tokenizer/morphopiece_vocab.json` - JSON format vocabulary

## Kullanım

```python
from src.morphopiece import MorphoPiece

# Tokenizer yükle
morphopiece = MorphoPiece("tokenizer/morphopiece.model")

# Morfem-aware encoding
tokens = morphopiece.encode(
    "Dün markete gittim",
    morpho_aware=True,
    out_type=int
)
# Output: [1234, 5678, 9012, ...]  # Kök ve ekler ayrı token'lar

# Decoding
text = morphopiece.decode(tokens)
# Output: "Dün markete gittim"
```

## Özellikler

✅ **Morfem Ayrımı**: Zemberek ile kök + ek ayrımı  
✅ **Kök = Ayrı Token**: Kökler ayrı token olarak saklanır  
✅ **Ek = Ayrı Token**: Ekler ayrı token olarak saklanır  
✅ **1.5 GB Corpus**: MC4 + Wikipedia Türkçe  
✅ **32k Vocab**: 32,000 token vocabulary  
✅ **Unigram Model**: SentencePiece unigram algoritması  
✅ **Character Coverage 1.0**: Tüm karakterleri kapsar  

## Süre Tahmini

| Adım | Süre |
|------|------|
| Veri İndirme | 10-30 dk (internet hızına bağlı) |
| Morfem Ön İşleme | 30-60 dk (corpus boyutuna bağlı) |
| Tokenizer Eğitimi | 10-30 dk (CPU'ya bağlı) |
| **TMA-1 Preprocessing** (YENİ) | 20-40 dk (corpus boyutuna bağlı) |
| **Toplam (Tokenizer)** | **50-120 dk** |
| **TMA-1 Model Eğitimi** | Değişken (epoch sayısı, corpus boyutu, GPU) |

**Not**: TMA-1 preprocessing yapıldığında eğitim süresi **10-100x azalır** (runtime morfolojik analiz yok).

## Notlar

- İlk çalıştırmada internet bağlantısı gerekli (veri indirme)
- Morfem analizi zaman alabilir (1.5 GB corpus)
- Tokenizer eğitimi CPU-intensive (multi-threading kullanır)
- Çıktı dosyaları ~100-200 MB olabilir

## Sorun Giderme

### "datasets not found"
```bash
pip install datasets
```

### "sentencepiece not found"
```bash
pip install sentencepiece
```

### "Memory error"
- `--max-lines` parametresiyle satır sayısını sınırlayın
- Örnek: `--max-lines 1000000`

### "Download timeout"
- Daha küçük corpus boyutu deneyin
- Örnek: `--mc4-size 0.5 --wikipedia-size 0.5`

## TMA-1 Model Eğitimi

### Hızlı Eğitim (Önerilen - Optimize Edilmiş)

```bash
# 1. Corpus ön işleme (morfolojik analiz - BİR KEZ)
python scripts/preprocess_for_tma1.py \
    --input data/corpus_morpho_processed.txt \
    --output data/train_data.jsonl \
    --tokenizer tokenizer/morphopiece.model

# 2. TMA-1 eğitimi (ön işlenmiş JSONL ile - HIZLI)
python train_tma1.py \
    --corpus data/train_data.jsonl \
    --tokenizer tokenizer/morphopiece.model \
    --output-dir models/tma1 \
    --batch-size 8 \
    --learning-rate 3e-4
```

**Önemli**: JSONL formatı kullanıldığında morfolojik analiz eğitim sırasında yapılmaz, bu da **10-100x hızlanma** sağlar.

### Eski Yöntem (Yavaş - Sadece Test İçin)

```bash
# Text formatı kullanılırsa runtime morfolojik analiz yapılır (YAVAŞ)
python train_tma1.py \
    --corpus data/corpus_morpho_processed.txt \
    --tokenizer tokenizer/morphopiece.model \
    --output-dir models/tma1
```

## Sonraki Adımlar

1. ✅ MorphoPiece tokenizer hazır
2. ✅ **Corpus preprocessing** (`preprocess_for_tma1.py`)
3. ✅ Testler (`pytest -q`)
4. 🔄 TMA-1 model eğitimi (`train_tma1.py` ile - **JSONL format kullanın**)
5. 🔄 Inference testi (`llm_engine.py` ile)

---

**"Morfem farkındalığı = Türkçe'nin DNA'sı"** 🧬

