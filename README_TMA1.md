// Developer: inkbytefo
// AI: Claude Sonnet 4.5
// Modified: 2025-11-01

# 🧠 TMA-1: Türkçe Mantık Ağı

> **Morfem farkındalıklı, eklemeli yapıya özel transformer modeli**

TMA-1, Türkçe'nin eklemeli yapısını modelin DNA'sına yerleştiren, morfolojik farkındalığa sahip bir transformer modelidir. Bu doküman, proje yapısıyla uyumlu, profesyonel kullanım talimatları ve açıklamalar içerir.

## 🎯 TMA-1 Özellikleri

### 1. Morfem Ayrımı (MorphoSplitter)
- Zemberek entegrasyonu ile kelime analizi
- Kök + ek ayrımı
- Ünlü uyumu kontrolü

### 2. MorphoPiece Tokenizer
- SentencePiece + morfem analizi kombinasyonu
- Kökler ve ekler ayrı token'lar
- Morfoloji-aware eğitim ve encoding opsiyonu

### 3. Agglutinative Attention
- SOV yapısına göre özel attention
- Yüklem token'larına ekstra bias
- Kök ve ek token'larına farklı ağırlıklar

### 4. Grammar Engine
- Ünlü uyumu kuralları
- Ek sırası kontrolü
- Yasak kombinasyon tespiti
- Logit bias sistemi

### 5. TMA-1 Model
- Morfem farkındalıklı transformer
- Grammar-aware generation
- Türkçe'ye özel mimari
- `AgglutinativeAttention` ve `GrammarEngine` ile logit/attention bias

## 🚀 Kullanım

### Morfem Ayrımı

```python
from src.morpho_splitter import MorphoSplitter

splitter = MorphoSplitter()
result = splitter.split_word("Evlerimdekiler")

print(result)
# {
#   "kelime": "evlerimdekiler",
#   "kök": "ev",
#   "ekler": ["ler", "im", "de", "ki", "ler"],
#   "morfemler": [...]
# }
```

### MorphoPiece Tokenizer

```python
from src.morphopiece import MorphoPiece

# Eğit
morphopiece = MorphoPiece()
morphopiece.train(
    corpus_file="data/corpus.txt",
    output_prefix="tokenizer/morphopiece",
    vocab_size=32000,
    morpho_aware=True
)

# Kullan
tokens = morphopiece.encode("Dün markete gittim", morpho_aware=True)
text = morphopiece.decode(tokens)
```

### TMA-1 Model

```python
from src.tma1_model import TMA1Model
from src.model import ModelConfig

config = ModelConfig(
    vocab_size=32000,
    hidden_size=768,
    num_layers=12,
    num_heads=12
)

model = TMA1Model(config)

# Forward pass
input_ids = torch.randint(0, 32000, (2, 10))
logits, _ = model(input_ids, vocab=vocab_list)
```

## 📊 Mimari Karşılaştırma

| Özellik | Standard Transformer | TMA-1 |
|---------|---------------------|-------|
| Tokenization | Word-based / BPE | MorphoPiece (kök+ek) |
| Attention | Standard | Agglutinative (SOV) |
| Grammar | None | Grammar Engine |
| Morfem Awareness | ❌ | ✅ |
| Vowel Harmony | ❌ | ✅ |
| Suffix Order | ❌ | ✅ |

## 🔧 Eğitim

Komut satırı örnekleri:

```bash
# MorphoPiece eğitimi (morfem ön işlemeyle)
python src/train_morphopiece.py --preprocess --corpus-file data/test_corpus.txt --preprocessed-file data/corpus_morpho_processed.txt --train --output tokenizer/morphopiece --vocab-size 1000

# Baseline Transformer eğitimi
python train.py --corpus data/test_corpus.txt --tokenizer tokenizer/morphopiece.model --output-dir models/baseline

# TMA-1 eğitimi
python train_tma1.py --corpus data/test_corpus.txt --tokenizer tokenizer/morphopiece.model --output-dir models/tma1
```

## 📝 Örnek Çıktı

**Input:** "Dün ne yaptın?"

**Standard Model:** "Dünü unuttum ama sanırım evdeydim."

**TMA-1:** "Dünü unuttum ama sanırım markete gittim." ✅

(TMA-1, ünlü uyumu ve ek sırası kurallarına daha uygun çıktı üretir)

## 🎯 Sonraki Adımlar

1. ✅ Morfem ayrımı (Zemberek/regex fallback)
2. ✅ MorphoPiece tokenizer
3. ✅ Agglutinative attention
4. ✅ Grammar engine
5. ✅ TMA-1 model
6. ✅ Testler (`pytest -q`)
7. 🔄 Geniş corpus ile uzun eğitim
8. 🔄 Değerlendirme metrikleri ve fine-tuning

## 📚 Dosya Yapısı

```
src/
├── morpho_splitter.py      # Morfem ayrımı
├── morphopiece.py          # MorphoPiece tokenizer
├── agglutinative_attention.py  # SOV attention
├── grammar_engine.py       # Dilbilgisi kuralları
└── tma1_model.py           # TMA-1 model
```

---

**Sahiplik ve Lisans:** Bu proje ve modeller Tevfik İşkın'a aittir. Ayrıntılar için `LICENSE.md`.

**"Türkçe'nin eklemeli yapısı = Model'in DNA'sı"** 🚀

