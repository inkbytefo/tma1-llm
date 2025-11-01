import os
import re

# Çeşitli kaynaklardan alınmış, temiz ve basit Türkçe cümleler
CORPUS_CONTENT = """
Kırmızı araba hızla yoldan geçti. Ali topu Ayşe'ye attı. Güneş doğudan doğar ve batıdan batar. Kitap okumak zihni geliştirir. Keloğlan, anasından izin alıp yola koyulmuş. Az gitmiş uz gitmiş, dere tepe düz gitmiş. Padişahın kızı çok güzelmiş. Bilgisayar programları kod yazılarak oluşturulur. Türkiye'nin başkenti Ankara şehridir. İstanbul, tarihi ve doğal güzellikleriyle ünlüdür. Kediler genellikle fareleri avlar. Su, sıfır derecede donar. Dünya, kendi ekseni etrafında döner. Başarı, düzenli çalışmanın bir sonucudur. Yarın hava güneşli olacakmış. Arkadaşım bana bir hediye aldı. Okula gitmek için sabah erken kalkarım. Annem çok lezzetli yemekler yapar. Babam her gün gazete okur. Gelecekte yapay zeka hayatımızın bir parçası olacak. Onlar ermiş muradına, biz çıkalım kerevetine.
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