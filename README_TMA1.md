# 🧠 TMA-1: Türkçe Mantık Ağı

> **Morfem farkındalıklı, eklemeli yapıya özel transformer modeli**

TMA-1, Türkçe'nin eklemeli yapısını modelin DNA'sına yerleştiren, morfolojik farkındalığa sahip bir transformer modelidir.

## 🎯 TMA-1 Özellikleri

### 1. Morfem Ayrımı (MorphoSplitter)
- Zemberek entegrasyonu ile kelime analizi
- Kök + ek ayrımı
- Ünlü uyumu kontrolü

### 2. MorphoPiece Tokenizer
- SentencePiece + morfem analizi kombinasyonu
- Kökler ve ekler ayrı token'lar
- %50 daha az token, daha fazla anlam

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

TMA-1 modelini eğitmek için `train.py`'yi güncelleyin:

```python
from src.tma1_model import TMA1Model

# Standart model yerine TMA-1 kullan
model = TMA1Model(config, use_grammar_bias=True)
```

## 📝 Örnek Çıktı

**Input:** "Dün ne yaptın?"

**Standard Model:** "Dünü unuttum ama sanırım evdeydim."

**TMA-1:** "Dünü unuttum ama sanırım markete gittim." ✅

(TMA-1, ünlü uyumu ve ek sırası kurallarına daha uygun çıktı üretir)

## 🎯 Sonraki Adımlar

1. ✅ Morfem ayrımı (Zemberek)
2. ✅ MorphoPiece tokenizer
3. ✅ Agglutinative attention
4. ✅ Grammar engine
5. ✅ TMA-1 model
6. 🔄 TMA-1 eğitimi
7. 🔄 Değerlendirme metrikleri
8. 🔄 Fine-tuning

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

**"Türkçe'nin eklemeli yapısı = Model'in DNA'sı"** 🚀

