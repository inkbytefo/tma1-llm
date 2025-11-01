#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from morpho_splitter import MorphoSplitter
from morphopiece import MorphoPiece

print("Testing MorphoSplitter:")
ms = MorphoSplitter()
result = ms.split_word('evler')
print(f"Result: {result}")
print(f"Kök: {result['kök']}")
print(f"Ekler: {result['ekler']}")

print("\nTesting MorphoPiece:")
mp = MorphoPiece()
tokens = mp.get_morpho_tokens('evler')
print(f"Tokens: {tokens}")
for t in tokens:
    print(f"  Token: {t['token']}, Type: {t['type']}, Word: {t['word']}")