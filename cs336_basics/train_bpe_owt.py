import time
import tracemalloc
from cs336_basics.bpe import train_bpe

if __name__ == "__main__":
    tracemalloc.start()
    start = time.time()
    
    # Run training (Note: Watch out for the typo in your filename: GPT$ -> GPT4)
    vocab, merges = train_bpe("./data/owt_valid.txt", 10000, ["<|endoftext|>"])
    end = time.time()
    
    # 1. Capture clean memory metrics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Peak memory: {peak / 10**6:.2f} MB")

    # 2. Serialize Merges
    with open("merges.txt", "w", encoding="utf-8") as f:
        for pair in merges:
            # Decode the bytes into strings, falling back to 'replace' if it hits a weird byte
            p1 = pair[0].decode("utf-8", errors="replace")
            p2 = pair[1].decode("utf-8", errors="replace")
            f.write(f"{p1} {p2}\n")

    # 3. Serialize Vocab
    with open("vocab.txt", "w", encoding="utf-8") as f:
        for token_id, token_bytes in vocab.items():
            token_str = token_bytes.decode("utf-8", errors="replace")
            # Replace literal newlines so it doesn't break your text file formatting
            token_str = token_str.replace("\n", "\\n") 
            f.write(f"{token_id}: {token_str}\n")
            
    # 4. Find the longest token
    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_str = longest_token_bytes.decode("utf-8", errors="replace")
    
    print(f"Longest token length: {len(longest_token_bytes)} bytes")
    print(f"Longest token string: '{longest_token_str}'")