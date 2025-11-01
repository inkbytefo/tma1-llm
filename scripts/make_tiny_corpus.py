#!/usr/bin/env python3
# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-02

"""
Create a small, high-quality Turkish corpus for experiments.
Writes data/tiny_corpus.txt with a classic fairy tale paragraph.
"""

import os

TINY_CORPUS = (
    "Bir varmış bir yokmuş, evvel zaman içinde kalbur saman içinde, develer tellal iken "
    "pireler berber iken, ben annemin beşiğini tıngır mıngır sallar iken, uzak bir diyarda "
    "küçük bir köy varmış. Bu köyün en akıllı çocuğu, adıyla sanıyla Keloğlan'mış. "
    "Keloğlan'ın ne saçı varmış ne de parası, ama kıvrak bir zekası ve altından daha değerli "
    "bir kalbi varmış. Bir gün padişahın kızı hastalanmış. Ülkenin en ünlü hekimleri gelmiş "
    "ama hiçbiri prensesi iyileştirememiş. Padişah, kızını iyileştirene kırk gün kırk gece "
    "düğün yapıp onu kızıyla evlendireceğini ilan etmiş. Keloğlan, anasından izin alıp yola "
    "koyulmuş. Az gitmiş uz gitmiş, dere tepe düz gitmiş. Yolda karşısına yaşlı bir nine "
    "çıkmış. Nine, Keloğlan'ın iyi niyetini anlamış ve ona sihirli bir elma vermiş. Bu elmayı "
    "yiyen her türlü hastalıktan kurtulurmuş. Keloğlan saraya varmış, padişahın huzuruna çıkmış "
    "ve elmayı prensese uzatmış. Prenses elmayı yer yemez iyileşmiş. Padişah sözünü tutmuş, "
    "Keloğlan ile prensesi evlendirmiş. Onlar ermiş muradına, biz çıkalım kerevetine."
)

def main():
    os.makedirs('data', exist_ok=True)
    out_path = os.path.join('data', 'tiny_corpus.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(TINY_CORPUS.strip() + "\n")
    print(f"✅ Wrote tiny corpus: {out_path}")
    print(f"   Characters: {len(TINY_CORPUS)}")
    print(f"   Contains Turkish letters: {any(ch in TINY_CORPUS for ch in 'ğüşiöçĞÜŞİÖÇ')}")

if __name__ == '__main__':
    main()