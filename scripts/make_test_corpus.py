# Developer: inkbytefo
# AI: Claude Sonnet 4.5
# Modified: 2025-11-01
import os
import itertools

def main():
    os.makedirs('data', exist_ok=True)
    src_primary = os.path.join('data', 'corpus_morpho_processed.txt')
    src_fallback = os.path.join('tests', 'test_corpus.txt')
    dst = os.path.join('data', 'test_corpus.txt')

    if os.path.exists(src_primary):
        with open(src_primary, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f.readlines()]
        # If we have at least 100 lines, take the first 100; else cycle to reach 100.
        if len(lines) >= 100:
            out_lines = lines[:100]
        else:
            out_lines = list(itertools.islice(itertools.cycle(lines), 100))
    else:
        # Fallback to tests corpus and cycle to 100 lines
        with open(src_fallback, 'r', encoding='utf-8') as f:
            lines = [ln.rstrip('\n') for ln in f.readlines()]
        out_lines = list(itertools.islice(itertools.cycle(lines), 100))

    with open(dst, 'w', encoding='utf-8') as f:
        for ln in out_lines:
            f.write(f"{ln}\n")

    # Validate
    with open(dst, 'r', encoding='utf-8') as f:
        final_count = sum(1 for _ in f)
    print(f"Created {dst} with {final_count} lines.")

if __name__ == '__main__':
    main()