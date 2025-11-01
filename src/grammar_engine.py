#!/usr/bin/env python3
# Developer: inkbytefo
# AI: QuantumSoul-Engineer-v1
# Modified: 2024-12-19

"""
============================================================================
Grammar Engine - Türkçe Dilbilgisi Kuralları
Ünlü uyumu, ek sırası, yasak kombinasyonlar için bias sistemi
============================================================================
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import re

class GrammarEngine:
    """Türkçe dilbilgisi kuralları motoru"""
    
    def __init__(self, penalty: float = -100.0, reward: float = 5.0):
        """
        GrammarEngine başlat
        
        Args:
            penalty: Yasak kombinasyonlar için ceza
            reward: Doğru kombinasyonlar için ödül
        """
        self.penalty = penalty
        self.reward = reward
        
        # Ünlü uyumu kuralları
        self.back_vowels = {'a', 'ı', 'o', 'u'}
        self.front_vowels = {'e', 'i', 'ö', 'ü'}
        
        # Yasak ek kombinasyonları
        self.forbidden_combinations = [
            ('a', 'e'),  # Kalın-ince uyumsuzluğu
            ('ı', 'i'),
            ('o', 'ö'),
            ('u', 'ü'),
            ('de', 'da'),  # Ünlü uyumu ile uyumsuz
            ('te', 'ta'),
            ('den', 'dan'),
            ('ten', 'tan'),
        ]
        
        # Doğru ek sıralaması (öncelik sırası)
        self.suffix_order = {
            'çoğul': 1,
            'iyelik': 2,
            'bulunma': 3,
            'ayrılma': 4,
            'yönelme': 5,
            'belirtme': 6,
            'geçmiş_zaman': 7,
            'şimdiki_zaman': 8,
            'gelecek_zaman': 9,
        }
    
    def check_vowel_harmony(self, root: str, suffix: str) -> bool:
        """
        Ünlü uyumu kontrolü
        
        Args:
            root: Kök kelime
            suffix: Ek
        
        Returns:
            Ünlü uyumu doğru mu?
        """
        if not root or not suffix:
            return True
        
        # Kökün son ünlüsünü bul
        root_vowels = [c for c in root if c in 'aeıiouöü']
        if not root_vowels:
            return True
        
        last_vowel = root_vowels[-1].lower()
        
        # Ekin ünlülerini bul
        suffix_vowels = [c for c in suffix.lower() if c in 'aeıiouöü']
        if not suffix_vowels:
            return True
        
        first_vowel = suffix_vowels[0].lower()
        
        # Kalın-ince uyumu
        if last_vowel in self.back_vowels:
            # Kalın ünlüden sonra kalın veya belirli ünlüler gelmeli
            if first_vowel in self.front_vowels and first_vowel not in {'i'}:
                return False
        elif last_vowel in self.front_vowels:
            # İnce ünlüden sonra ince ünlüler gelmeli
            if first_vowel in self.back_vowels and first_vowel not in {'a'}:
                return False
        
        # Düz-yuvarlak uyumu (basit)
        if last_vowel in {'o', 'u'} and first_vowel in {'ö', 'ü', 'e', 'i'}:
            return False
        elif last_vowel in {'ö', 'ü'} and first_vowel in {'a', 'ı', 'o', 'u'}:
            return False
        
        return True
    
    def check_suffix_order(self, suffixes: List[str]) -> bool:
        """
        Ek sırası kontrolü
        
        Args:
            suffixes: Ek listesi
        
        Returns:
            Ek sırası doğru mu?
        """
        # Basit kontrol: yüklem ekleri genelde sonda olmalı
        time_suffixes = ['geçmiş_zaman', 'şimdiki_zaman', 'gelecek_zaman']
        
        time_indices = []
        other_indices = []
        
        for i, suffix in enumerate(suffixes):
            if any(ts in suffix for ts in time_suffixes):
                time_indices.append(i)
            else:
                other_indices.append(i)
        
        # Zaman ekleri diğer eklerden sonra olmalı
        if time_indices and other_indices:
            if min(time_indices) < max(other_indices):
                return False
        
        return True
    
    def apply_grammar_bias(
        self,
        logits: torch.Tensor,
        vocab: List[str],
        previous_tokens: List[str],
        morpho_analysis: Optional[List[Dict]] = None
    ) -> torch.Tensor:
        """
        Dilbilgisi kurallarına göre logit bias'ı uygula
        
        Args:
            logits: Logit tensor [batch, seq_len, vocab_size]
            vocab: Vocabulary listesi
            previous_tokens: Önceki token'lar
            morpho_analysis: Morfem analizi (opsiyonel)
        
        Returns:
            Bias uygulanmış logits
        """
        batch_size, seq_len, vocab_size = logits.shape
        biased_logits = logits.clone()
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Son token'ı al
                if s > 0 and len(previous_tokens) > s - 1:
                    last_token = previous_tokens[s - 1]
                else:
                    last_token = None
                
                # Her vocabulary token için kontrol
                for v_idx, token in enumerate(vocab):
                    if v_idx >= vocab_size:
                        break
                    
                    bias = 0.0
                    
                    # Ünlü uyumu kontrolü
                    if last_token:
                        if not self.check_vowel_harmony(last_token, token):
                            bias += self.penalty
                        else:
                            bias += self.reward * 0.1
                    
                    # Yasak kombinasyon kontrolü
                    if last_token:
                        for forbidden in self.forbidden_combinations:
                            if (forbidden[0] in last_token.lower() and forbidden[1] in token.lower()) or \
                               (forbidden[1] in last_token.lower() and forbidden[0] in token.lower()):
                                bias += self.penalty * 0.5
                    
                    # Morfem analizi varsa daha detaylı kontrol
                    if morpho_analysis and s < len(morpho_analysis):
                        morpho = morpho_analysis[s]
                        
                        # Kök + ek uyumu
                        if morpho.get('tür') == 'kök' and last_token and '-' in last_token:
                            # Önceki token bir ek ise, yeni token kök olabilir (iyi)
                            bias += self.reward * 0.2
                        elif morpho.get('tür') == 'ek' and last_token and '-' not in last_token:
                            # Önceki token kök, yeni token ek (doğru sıra)
                            if self.check_vowel_harmony(last_token, token):
                                bias += self.reward * 0.3
                            else:
                                bias += self.penalty * 0.3
                    
                    # Bias'ı uygula
                    biased_logits[b, s, v_idx] += bias
        
        return biased_logits
    
    def get_vowel_harmony_mask(
        self,
        vocab: List[str],
        last_vowel: str,
        device: torch.device
    ) -> torch.Tensor:
        """
        Ünlü uyumu maskesi oluştur
        
        Args:
            vocab: Vocabulary listesi
            last_vowel: Son ünlü
            device: Tensor device
        
        Returns:
            Mask tensor [vocab_size] (1=uyumlu, 0=uyumsuz)
        """
        mask = torch.ones(len(vocab), device=device)
        
        if not last_vowel or last_vowel not in 'aeıiouöü':
            return mask
        
        last_vowel_lower = last_vowel.lower()
        
        for i, token in enumerate(vocab):
            token_vowels = [c for c in token.lower() if c in 'aeıiouöü']
            if not token_vowels:
                continue
            
            first_vowel = token_vowels[0].lower()
            
            # Ünlü uyumu kontrolü
            if not self.check_vowel_harmony(last_vowel, first_vowel):
                mask[i] = 0.0
        
        return mask
    
    def validate_sequence(self, tokens: List[str]) -> Tuple[bool, List[str]]:
        """
        Token dizisinin dilbilgisi kurallarına uygunluğunu kontrol et
        
        Args:
            tokens: Token listesi
        
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        for i in range(1, len(tokens)):
            prev_token = tokens[i - 1]
            curr_token = tokens[i]
            
            # Ünlü uyumu kontrolü
            if not self.check_vowel_harmony(prev_token, curr_token):
                errors.append(f"Ünlü uyumu hatası: '{prev_token}' + '{curr_token}'")
            
            # Yasak kombinasyon kontrolü
            for forbidden in self.forbidden_combinations:
                if (forbidden[0] in prev_token.lower() and forbidden[1] in curr_token.lower()) or \
                   (forbidden[1] in prev_token.lower() and forbidden[0] in curr_token.lower()):
                    errors.append(f"Yasak kombinasyon: '{prev_token}' + '{curr_token}'")
        
        return len(errors) == 0, errors

def main():
    """Test fonksiyonu"""
    print("\n" + "=" * 60)
    print("    Grammar Engine Test")
    print("=" * 60)
    
    engine = GrammarEngine()
    
    # Ünlü uyumu testleri
    test_cases = [
        ("ev", "de", True),   # Doğru
        ("ev", "da", True),   # Doğru (alternatif)
        ("ev", "ta", False),  # Yanlış (sert ünsüz uyumsuzluğu)
        ("kitap", "da", True), # Doğru
        ("kitap", "de", False), # Yanlış
        ("güzel", "de", True),  # Doğru
        ("güzel", "da", False), # Yanlış
    ]
    
    print("\n📝 Ünlü Uyumu Testleri:")
    for root, suffix, expected in test_cases:
        result = engine.check_vowel_harmony(root, suffix)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {root} + {suffix} → {result} (beklenen: {expected})")
    
    # Token dizisi validasyonu
    print("\n📝 Dizi Validasyonu:")
    sequences = [
        ["ev", "de", "ki", "ler"],
        ["ev", "da", "ki", "ler"],  # Ünlü uyumu hatası
        ["kitap", "dan", "aldım"],
        ["güzel", "de", "bir", "kitap"],
    ]
    
    for seq in sequences:
        is_valid, errors = engine.validate_sequence(seq)
        status = "✅" if is_valid else "❌"
        print(f"   {status} {' '.join(seq)}")
        if errors:
            for error in errors:
                print(f"      - {error}")

if __name__ == "__main__":
    main()

