# Semantic Categorization - Gelecek İyileştirmeler

## Mevcut Durum

Şu anda `get_semantic_category()` fonksiyonu **manuel keyword-based matching** kullanıyor. Bu yaklaşım:
- ✅ Hızlı ve basit
- ✅ Bağımlılık gerektirmiyor
- ❌ Manuel bakım gerektiriyor
- ❌ Yeni kelimeler için keyword listesi genişletilmeli
- ❌ Kapsam sınırlı

## Gelecek İyileştirme Yol Haritası

### 1. Zemberek Semantic Tags Entegrasyonu

**Öncelik: Yüksek** | **Tahmini Süre: 1-2 hafta**

Zemberek zaten morfolojik analiz için kullanılıyor. Semantic kategorizasyon için de kullanılabilir:

#### Avantajlar:
- Mevcut altyapı ile entegre
- POS tag'lerden semantic kategori çıkarımı
- Lemma bilgisi ile daha doğru kategorizasyon
- Otomatik kategorizasyon (manuel keyword listesi gerekmez)

#### Implementasyon Planı:
```python
def get_semantic_category_zemberek(token: str, morpho_splitter: MorphoSplitter) -> Optional[int]:
    """
    Zemberek analizinden semantic kategori çıkar
    
    Zemberek POS tag'lerinden semantic kategori mapping:
    - Noun -> mekan, insan, hayvan, eşya, yiyecek kategorilerine göre
    - Verb -> fiil_eylem
    - Adjective -> sıfat
    - Adverb -> belirsiz (veya zaman için özel kontrol)
    """
    if not morpho_splitter.use_java:
        return None
    
    zemberek_result = morpho_splitter._zemberek_analyze(token)
    if not zemberek_result:
        return None
    
    # POS tag'den semantic kategori çıkar
    pos_tag = zemberek_result[0].get('pos_tag', '')
    lemma = zemberek_result[0].get('lemma', '')
    
    # POS tag mapping
    if 'Noun' in pos_tag:
        # Lemma bazlı daha detaylı kategorizasyon
        return _classify_noun_semantics(lemma)
    elif 'Verb' in pos_tag:
        return SEMANTIC_CATEGORY_MAP['fiil_eylem']
    elif 'Adjective' in pos_tag:
        return SEMANTIC_CATEGORY_MAP['sıfat']
    # ...
    
    return None
```

#### Zemberek POS Tag Mapping:
- `Noun` → Lemma'ya göre mekan/insan/hayvan/eşya/yiyecek kategorileri
- `Verb` → `fiil_eylem`
- `Adjective` → `sıfat`
- `Adverb` → `zaman` (temporal adverbs) veya `belirsiz`
- `Pronoun` → `belirsiz` veya özel kategori
- `Number` → `zaman` (tarih/saat) veya `belirsiz`

#### Zemberek Lemma Bazlı Semantic Mapping:
Zemberek lemma bilgisi ile daha doğru kategorizasyon:
- Lemma'nın semantic özelliklerini kullanarak kategori atama
- Örnek: "okul" lemması → mekan kategorisi

### 2. WordNet Ontoloji Entegrasyonu

**Öncelik: Orta** | **Tahmini Süre: 2-4 hafta**

WordNet (veya Türkçe WordNet) ontolojisi ile otomatik semantic kategorizasyon:

#### Avantajlar:
- Ontoloji tabanlı doğru kategorizasyon
- Hypernym/Hyponym ilişkilerinden otomatik kategori çıkarımı
- Geniş kapsam (binlerce kelime)
- Synset bazlı semantic benzerlik

#### Türkçe WordNet Seçenekleri:
1. **KaNet (KartalNet)** - Türkçe WordNet projesi
   - URL: https://github.com/kartalnet/kanet
   - Status: Aktif geliştirme
   
2. **Open Multilingual WordNet**
   - URL: http://compling.hss.ntu.edu.sg/omw/
   - Türkçe desteği var
   
3. **BabelNet**
   - URL: https://babelnet.org/
   - Çok dilli semantic network

#### Implementasyon Planı:
```python
def get_semantic_category_wordnet(token: str, root: str) -> Optional[int]:
    """
    WordNet ontolojisinden semantic kategori çıkar
    
    Hypernym hiyerarşisinden kategori çıkarımı:
    - location.n.01 → mekan
    - person.n.01 → insan
    - animal.n.01 → hayvan
    - food.n.01 → yiyecek
    - artifact.n.01 → eşya
    - verb → fiil_eylem
    - adjective → sıfat
    """
    try:
        from nltk.corpus import wordnet as wn
        import nltk
        
        # Türkçe WordNet kullanılıyorsa
        # synsets = wn.synsets(root, lang='tur')
        
        # Veya İngilizce WordNet ile cross-lingual mapping
        # (Türkçe kelime → İngilizce çeviri → WordNet lookup)
        synsets = wn.synsets(root, lang='eng')  # fallback
        
        if not synsets:
            return None
        
        # İlk synset'i al (en yaygın anlam)
        synset = synsets[0]
        
        # Hypernym hiyerarşisinden semantic kategori çıkar
        hypernyms = synset.hypernym_paths()
        if hypernyms:
            top_hypernym = hypernyms[0][-1]  # En üst hypernym
            category = _map_wordnet_to_semantic_category(top_hypernym.name())
            if category:
                return SEMANTIC_CATEGORY_MAP[category]
        
        # Synset'in kendisinden kategori çıkar
        category = _map_wordnet_to_semantic_category(synset.name())
        return SEMANTIC_CATEGORY_MAP.get(category, SEMANTIC_CATEGORY_MAP['belirsiz'])
        
    except ImportError:
        return None  # NLTK/WordNet yoksa None dön

def _map_wordnet_to_semantic_category(wordnet_synset_name: str) -> Optional[str]:
    """
    WordNet synset isminden semantic kategori eşleştir
    
    Mapping kuralları:
    - location.n.* → mekan
    - person.n.* → insan
    - animal.n.* → hayvan
    - food.n.* → yiyecek
    - artifact.n.* → eşya
    - time.n.* → zaman
    - emotion.n.* → duygu
    - verb.* → fiil_eylem
    - adjective.* → sıfat
    """
    name_lower = wordnet_synset_name.lower()
    
    if 'location' in name_lower or 'place' in name_lower:
        return 'mekan'
    elif 'person' in name_lower or 'human' in name_lower:
        return 'insan'
    elif 'animal' in name_lower:
        return 'hayvan'
    elif 'food' in name_lower:
        return 'yiyecek'
    elif 'artifact' in name_lower or 'object' in name_lower:
        return 'eşya'
    elif 'time' in name_lower or 'temporal' in name_lower:
        return 'zaman'
    elif 'emotion' in name_lower or 'feeling' in name_lower:
        return 'duygu'
    elif 'verb' in name_lower or name_lower.startswith('v.'):
        return 'fiil_eylem'
    elif 'adjective' in name_lower or name_lower.startswith('a.'):
        return 'sıfat'
    
    return None
```

#### Dependency Gereksinimleri:
```bash
pip install nltk
python -c "import nltk; nltk.download('wordnet')"
python -c "import nltk; nltk.download('omw')"  # Open Multilingual WordNet
```

### 3. Hybrid Yaklaşım (Önerilen)

**Öncelik: Yüksek** | **Tahmini Süre: 3-4 hafta**

En iyi yaklaşım: **Fallback hiyerarşisi**

```
1. Zemberek semantic tags (en hızlı, mevcut altyapı)
   ↓ (eğer başarısız)
2. WordNet ontoloji (otomatik, geniş kapsam)
   ↓ (eğer başarısız)
3. Keyword-based matching (manuel, fallback)
   ↓ (eğer başarısız)
4. Morfolojik tip fallback (fiil/sıfat için)
   ↓ (eğer başarısız)
5. Belirsiz kategorisi
```

#### Implementasyon:
```python
def get_semantic_category(token: str, morpho_splitter: MorphoSplitter) -> int:
    """
    Hybrid yaklaşım: Zemberek → WordNet → Keyword → Morpho → Belirsiz
    """
    # ... özel token kontrolü ...
    
    analysis = morpho_splitter.split_word(token.strip())
    root = analysis.get('kök', token.strip().lower())
    root_lower = root.lower()
    
    # 1. Zemberek semantic tags (en hızlı)
    semantic_cat = get_semantic_category_zemberek(token, morpho_splitter)
    if semantic_cat is not None:
        return semantic_cat
    
    # 2. WordNet ontoloji (otomatik, geniş kapsam)
    semantic_cat = get_semantic_category_wordnet(token, root_lower)
    if semantic_cat is not None:
        return semantic_cat
    
    # 3. Keyword-based matching (manuel fallback)
    for category, keywords in SEMANTIC_KEYWORDS.items():
        if root_lower in keywords:
            return SEMANTIC_CATEGORY_MAP[category]
    
    # 4. Morfolojik tip fallback
    morpho_type = get_detailed_morpho_type(token, morpho_splitter)
    if morpho_type == MORPHEME_TYPE_MAP['fiil_kök']:
        return SEMANTIC_CATEGORY_MAP['fiil_eylem']
    if morpho_type == MORPHEME_TYPE_MAP['sıfat_kök']:
        return SEMANTIC_CATEGORY_MAP['sıfat']
    
    # 5. Belirsiz
    return SEMANTIC_CATEGORY_MAP['belirsiz']
```

### 4. Performans Optimizasyonu

**Öncelik: Orta** | **Tahmini Süre: 1 hafta**

- **Caching**: Semantic kategoriler cache'lenebilir (kelime → kategori mapping)
- **Batch Processing**: Toplu işleme için optimize edilmiş fonksiyonlar
- **Lazy Loading**: WordNet sadece gerektiğinde yüklenmeli

### 5. Test ve Değerlendirme

**Öncelik: Yüksek** | **Tahmini Süre: 1-2 hafta**

- Test corpus'u oluşturma (her kategori için örnek kelimeler)
- Accuracy metrikleri (manuel annotasyon ile karşılaştırma)
- Performans benchmark'ları

## Uygulama Öncelikleri

### Faz 1 (Kısa Vadeli - 2-3 hafta):
1. ✅ Manuel keyword dictionary (tamamlandı)
2. 🔄 Zemberek semantic tags entegrasyonu
3. 🔄 Hybrid yaklaşım implementasyonu

### Faz 2 (Orta Vadeli - 1-2 ay):
4. 🔄 WordNet ontoloji entegrasyonu
5. 🔄 Performans optimizasyonu (caching)
6. 🔄 Test ve değerlendirme

### Faz 3 (Uzun Vadeli - 3+ ay):
7. 🔄 Özel Türkçe semantic ontology geliştirme
8. 🔄 ML-based semantic categorization (supervised learning)
9. 🔄 Context-aware semantic categorization (cümle bağlamı)

## Notlar

- **Mevcut keyword listesi** yeterli bir başlangıçtır ve gelecekte genişletilebilir
- **Zemberek entegrasyonu** en düşük effort ile en yüksek kazanç sağlar
- **WordNet entegrasyonu** geniş kapsam sağlar ancak dependency gerektirir
- **Hybrid yaklaşım** hem doğruluk hem de kapsam açısından en iyi sonucu verir

## Referanslar

- [Zemberek NLP](https://github.com/ahmetaa/zemberek-nlp)
- [NLTK WordNet](https://www.nltk.org/howto/wordnet.html)
- [Open Multilingual WordNet](http://compling.hss.ntu.edu.sg/omw/)
- [KaNet (Türkçe WordNet)](https://github.com/kartalnet/kanet)

