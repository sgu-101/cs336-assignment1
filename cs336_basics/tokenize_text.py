import multiprocessing as mp
import numpy as np
import time
from cs336_basics.tokenizer import Tokenizer # Adjust import as needed

# 1. The Worker Function (MUST be at the top level of the script)
def _encode_chunk(args):
    """Worker process: Encodes a list of lines and returns a compressed NumPy array."""
    tokenizer, lines = args
    chunk_ids = []
    
    for line in lines:
        chunk_ids.extend(tokenizer.encode(line))
        
    # Convert to uint16 INSIDE the worker to minimize memory sent between processes
    return np.array(chunk_ids, dtype=np.uint16)


# 2. The Lazy Line Generator
def _line_generator(filepath, chunk_size=50_000):
    """Yields chunks of lines lazily to keep RAM usage near zero."""
    chunk = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
    
    # Yield whatever is left over at the end of the file
    if chunk:
        yield chunk


# 3. The Main Coordinator Function
def encode_file_mp(filepath: str, filename: str, tokenizer, num_cores: int = None):
    if num_cores is None:
        # Use all available cores except one to keep the OS responsive
        num_cores = max(1, mp.cpu_count() - 1)

    print(f"Encoding {filepath} using {num_cores} cores...")
    start_time = time.time()

    # Create the generator that pairs the tokenizer with each text chunk
    args_generator = ((tokenizer, chunk) for chunk in _line_generator(filepath, chunk_size=50_000))

    encoded_chunks = []
    
    # Start the multiprocessing pool
    with mp.Pool(processes=num_cores) as pool:
        # pool.imap guarantees that the chunks are returned in the EXACT order they were sent!
        for np_array in pool.imap(_encode_chunk, args_generator):
            encoded_chunks.append(np_array)

    # Stitch the parallelized chunks back together
    catencoded = np.concatenate(encoded_chunks)
    
    # Save the final binary
    np.save(filename, catencoded)
    
    elapsed = time.time() - start_time
    print(f"Successfully saved {filename}")
    print(f"-> Tokens: {len(catencoded):,} | Time: {elapsed / 60:.2f} minutes\n")


if __name__ == "__main__":
    print("Starting...")
    
    # 1. Load Tokenizers
    tstory_tokenizer = Tokenizer.from_files(
        "./results/tinystories-bpe/vocab_tinystories.pkl", 
        "./results/tinystories-bpe/merges_tinystories.pkl", 
        special_tokens=["<|endoftext|>"]
    )
    
    owt_tokenizer = Tokenizer.from_files(
        "./results/owt-bpe/vocab_owt.pkl", 
        "./results/owt-bpe/merges_owt.pkl", 
        special_tokens=["<|endoftext|>"]
    )

    # 2. Run Parallel Encoding (Notice we now pass the specific tokenizer!)
    encode_file_mp(
        "./data/TinyStoriesV2-GPT4-valid.txt", 
        "tstories_valid_encoded.npy", 
        tstory_tokenizer
    )
    
    encode_file_mp(
        "./data/TinyStoriesV2-GPT4-train.txt", 
        "tstories_train_encoded.npy", 
        tstory_tokenizer
    )
    
    encode_file_mp(
        "./data/owt_valid.txt", 
        "owt_valid_encoded.npy", 
        owt_tokenizer
    )
    
    encode_file_mp(
        "./data/owt_train.txt", 
        "owt_train_encoded.npy", 
        owt_tokenizer
    )