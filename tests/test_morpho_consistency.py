#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
Morpho Consistency Test - Veri Tutarlılığı Kontrolü
train_morphopiece.py ve train_tma1.py arasındaki morfolojik ayrım tutarlılığını test eder
============================================================================
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.morpho_splitter import MorphoSplitter
from src.train_morphopiece import preprocess_with_morpho
import tempfile

def test_morpho_consistency():
    """Morfem ayrımı tutarlılık testi"""
    print("\n" + "=" * 60)
    print("    Morpho Consistency Test")
    print("=" * 60)
    
    splitter = MorphoSplitter(use_java=False)  # Sadece regex fallback
    
    # Test cümleleri
    test_sentences = [
        "Dün markete gittim",
        "Evlerimdekiler",
        "Yapay zeka geleceğin teknolojisidir",
        "Okuldan gelirken arkadaşımı gördüm"
    ]
    
    print("\n[INFO] Test Cümleleri ve Morfem Ayrımları:")
    for sentence in test_sentences:
        analysis = splitter.split_sentence(sentence)
        morphemes = []
        
        for word_analysis in analysis["kelimeler"]:
            root = word_analysis["kök"]
            if root:
                morphemes.append(root)
            
            for suffix in word_analysis["ekler"]:
                suffix_clean = suffix.lstrip('-')
                if suffix_clean:
                    morphemes.append(suffix_clean)
        
        processed = ' '.join(morphemes)
        print(f"\n   Cümle: {sentence}")
        print(f"   Morfemler: {processed}")
        print(f"   Kelime sayısı: {len(analysis['kelimeler'])}")
    
    # Dosya işleme testi
    print("\n\n[INFO] Dosya İşleme Tutarlılık Testi:")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix='.txt') as f_in:
        # Test cümlelerini yaz
        for sentence in test_sentences:
            f_in.write(sentence + '\n')
        input_file = f_in.name
    
    output_file = input_file.replace('.txt', '_processed.txt')
    
    try:
        # Preprocess
        success = preprocess_with_morpho(input_file, output_file, splitter)
        
        if not success:
            print("[FAIL] Preprocessing failed!")
            return False
        
        # İşlenmiş dosyayı oku ve kontrol et
        with open(output_file, 'r', encoding='utf-8') as f:
            processed_lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"[PASS] Preprocessing successful")
        print(f"   Input lines: {len(test_sentences)}")
        print(f"   Output lines: {len(processed_lines)}")
        
        print("\n[EXAMPLES] Örnek işlemeler:")
        for i, (orig, proc) in enumerate(zip(test_sentences, processed_lines[:3])):
            print(f"\n   {i+1}. Orijinal: {orig}")
            print(f"      İşlenmiş: {proc}")
        
        # Tutarlılık kontrolü: Her kelime için aynı morfemi üretmeli
        print("\n[CHECK] Tutarlılık Kontrolü:")
        all_consistent = True
        
        for sentence in test_sentences:
            # 1. split_sentence ile analiz
            analysis1 = splitter.split_sentence(sentence)
            morphemes1 = []
            for word_analysis in analysis1["kelimeler"]:
                root = word_analysis["kök"]
                if root:
                    morphemes1.append(root)
                for suffix in word_analysis["ekler"]:
                    suffix_clean = suffix.lstrip('-')
                    if suffix_clean:
                        morphemes1.append(suffix_clean)
            
            # 2. Her kelimeyi ayrı ayrı analiz et
            morphemes2 = []
            import re
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                word_analysis = splitter.split_word(word)
                root = word_analysis["kök"]
                if root:
                    morphemes2.append(root)
                for suffix in word_analysis["ekler"]:
                    suffix_clean = suffix.lstrip('-')
                    if suffix_clean:
                        morphemes2.append(suffix_clean)
            
            # Karşılaştır
            consistent = morphemes1 == morphemes2
            if not consistent:
                all_consistent = False
                print(f"   [FAIL] {sentence}")
                print(f"      split_sentence: {' '.join(morphemes1)}")
                print(f"      split_word: {' '.join(morphemes2)}")
        
        if all_consistent:
            print("   [PASS] Tüm cümleler tutarlı")
        
        # Cleanup
        os.unlink(input_file)
        os.unlink(output_file)
        
        return all_consistent
        
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)
        
        return False

def test_regex_fallback_quality():
    """Regex fallback kalite testi"""
    print("\n" + "=" * 60)
    print("    Regex Fallback Quality Test")
    print("=" * 60)
    
    splitter = MorphoSplitter(use_java=False)
    
    # Karmaşık kelimeler
    complex_words = [
        "Evlerimdekiler",  # ev-ler-im-de-ki-ler
        "Gittim",  # git-ti-m
        "Anladım",  # anla-dı-m
        "Yapayzeka",  # yapay-zeka (compund)
        "Teknolojisidir"  # teknoloji-si-dir
    ]
    
    print("\n[INFO] Karmaşık Kelime Analizi:")
    for word in complex_words:
        result = splitter.split_word(word)
        print(f"\n   Kelime: {word}")
        print(f"   Kök: {result['kök']}")
        print(f"   Ekler: {', '.join(result['ekler']) if result['ekler'] else '(yok)'}")
        print(f"   Toplam morfem: {len(result['morfemler'])}")
        
        # Ünlü uyumu kontrolü
        if result['ekler']:
            last_vowel = [c for c in result['kök'] if c in 'aeıiouöü']
            if last_vowel:
                is_valid = splitter.is_valid_vowel_harmony(result['kök'], result['ekler'][0])
                print(f"   Ünlü uyumu: {'[PASS]' if is_valid else '[FAIL]'}")

if __name__ == "__main__":
    # Set UTF-8 encoding for Windows
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    
    print("\n[RUNNING] Morpho Consistency Tests...")
    print("")
    
    # Test 1: Tutarlılık
    consistency_ok = test_morpho_consistency()
    
    # Test 2: Regex kalite
    test_regex_fallback_quality()
    
    print("\n" + "=" * 60)
    if consistency_ok:
        print("[PASS] All tests passed!")
    else:
        print("[FAIL] Some tests failed!")
    print("=" * 60)
    print("")

