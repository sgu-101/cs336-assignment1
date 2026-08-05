from tokenizer import Tokenizer
import random
import time

def reservoir_sample(filepath, samples):
  reservoir = []
  currdoc = []
  seen = 0

  with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
      if "<|endoftext|>" in line:
        fulldoc = "".join(currdoc).strip()

        if not fulldoc:
          currdoc = []
          continue

        if seen < samples:
          reservoir.append(fulldoc)
        else:
          ri = random.randint(0, seen)
          if ri < samples:
            reservoir[ri] = fulldoc

        currdoc = []
        seen += 1
      else:
        currdoc.append(line)

  return reservoir


if __name__ == "__main__":
  print("tinystories sample\n")
  tstories_samples = reservoir_sample("./data/TinyStoriesV2-GPT4-train.txt", 10)
  print("owt sample\n")
  owt_samples = reservoir_sample("./data/owt_train.txt", 10)

  tstory_tokenizer = Tokenizer.from_files("./results/tinystories-bpe/vocab_tinystories.pkl", "./results/tinystories-bpe/merges_tinystories.pkl", ["<|endoftext|>"])
  owt_tokenizer = Tokenizer.from_files("./results/owt-bpe/vocab_owt.pkl", "./results/owt-bpe/merges_owt.pkl", ["<|endoftext|>"])

  # # compression ratios
  # tstory_idlen = 0
  # tstory_bytelen = 0
  # for tstory in tstories_samples:
  #     encoded = tstory_tokenizer.encode(tstory)
  #     tstory_bytelen += len(tstory.encode("utf-8"))
  #     tstory_idlen += len(encoded)

  # print(f"tstory compression ratio (bytes/token): {tstory_bytelen/tstory_idlen}\n")

  # owt_idlen = 0
  # owt_bytelen = 0
  # for owt in owt_samples:
  #     encoded = owt_tokenizer.encode(owt)
  #     owt_bytelen += len(owt.encode("utf-8"))
  #     owt_idlen += len(encoded)

  # print(f"owt compression ratio (bytes/token): {owt_bytelen/owt_idlen}\n")

  # # tinystories encoder on owt
  # owt_idlen = 0
  # owt_bytelen = 0
  # for owt in owt_samples:
  #     encoded = tstory_tokenizer.encode(owt)
  #     owt_bytelen += len(owt.encode("utf-8"))
  #     owt_idlen += len(encoded)

  # print(f"owt compression ratio using tinystories tokenizer (bytes/token): {owt_bytelen/owt_idlen}\n")

  # estimated throughput
  smallts = "".join(tstories_samples)
  smalltsbytetotal = len(smallts.encode("utf-8"))

  start = time.time()
  tstory_tokenizer.encode(smallts)
  end = time.time()

  print(f"Estimated Tiny Stories throughput (bytes/s): {smalltsbytetotal/(end - start)}")
  print(f"Total time: {end - start}")
  print(f"Bytes: {smalltsbytetotal}")
  smallowt = "".join(owt_samples)
  owtbytetotal = len(smallowt.encode("utf-8"))

  start = time.time()
  owt_tokenizer.encode(smallowt)
  end = time.time()

  print(f"Estimated OWT throughput (bytes/s): {owtbytetotal/(end - start)}")
  print(f"Total time: {end - start}")
  print(f"Bytes: {owtbytetotal}")
