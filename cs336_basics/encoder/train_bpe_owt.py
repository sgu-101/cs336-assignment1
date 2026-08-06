import time
import tracemalloc
import pickle
from cs336_basics.encoder.bpe import train_bpe

if __name__ == "__main__":
    tracemalloc.start()
    start = time.time()
    
    # Run training (Note: Watch out for the typo in your filename: GPT$ -> GPT4)
    vocab, merges = train_bpe("./data/owt_train.txt", 32000, ["<|endoftext|>"])
    end = time.time()
    
    # 1. Capture clean memory metrics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Time taken: {end - start:.2f} seconds")
    print(f"Peak memory: {peak / 10**6:.2f} MB")

    with open("merges.pkl", "wb") as f:
      pickle.dump(merges, f)
    
    with open("vocab.pkl", "wb") as f:
      pickle.dump(vocab, f)
            
    # 4. Find the longest token
    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_str = longest_token_bytes.decode("utf-8", errors="replace")
    
    print(f"Longest token length: {len(longest_token_bytes)} bytes")
    print(f"Longest token string: '{longest_token_str}'")