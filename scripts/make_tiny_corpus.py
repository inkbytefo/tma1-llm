import os
import re

# Çeşitli kaynaklardan alınmış, temiz ve basit Türkçe cümleler
CORPUS_CONTENT = """
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
"""
def main():
    # Projenin ana dizininde olduğumuzu varsayarak data klasörü oluştur
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'tiny_corpus_sentences.txt')
    
    # Metni cümlelere ayır ve temizle
    # Noktalama işaretlerinden sonra boşluk bırakarak ayır
    sentences = re.split(r'(?<=[.!?])\s+', CORPUS_CONTENT.strip())
    cleaned_sentences = [s.strip() for s in sentences if s and len(s.strip()) > 5]
    
    # Cümleleri dosyaya yaz
    with open(output_path, 'w', encoding='utf-8') as f:
        for sentence in cleaned_sentences:
            f.write(sentence + '\n')
    
    print(f"✅ '{output_path}' adında {len(cleaned_sentences)} cümleden oluşan mini veri seti oluşturuldu.")
    print("💡 Bu dosyayı şimdi Git'e ekleyebilirsiniz.")

if __name__ == '__main__':
    main()