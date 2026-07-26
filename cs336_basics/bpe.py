import regex as re
from collections import defaultdict 

def train_bpe(
    input_path: str, 
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  """Given the path to an input corpus, run train a BPE tokenizer and
  output its vocabulary and merges.

  Args:
      input_path (str | os.PathLike): Path to BPE tokenizer training data.
      vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
      special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
          These strings will never be split into multiple tokens, and will always be
          kept as a single token. If these special tokens occur in the `input_path`,
          they are treated as any other string.

  Returns:
      tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
          vocab:
              The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
              to bytes (token bytes)
          merges:
              BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
              representing that <token1> was merged with <token2>.
              Merges are ordered by order of creation.
  """

  # open file
  try:
    with open(input_path, "r") as f:
      text = f.read()
  except FileNotFoundError:
    print(f"No file at {input_path}.")
    raise FileNotFoundError
  
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  specials = f"({'|'.join([re.escape(token) for token in special_tokens])})"
  splitted = re.split(specials, text)

  # fill pretokenized_vocab
  pretokenized_vocab = {} # dict[tuple[bytes, bytes, ...], int]

  for spl in splitted:
    if spl in special_tokens:
      continue
    matches = re.finditer(PAT, spl)
    for m in matches:
      currstr = m.group()
      bytematch = currstr.encode("utf-8")
      match = tuple(bytematch[i:i+1] for i in range(len(bytematch)))
      pretokenized_vocab[match] = pretokenized_vocab.get(match, 0) + 1

  # populate initial vocab
  currsize = 256
  vocab = {}
  for i in range(currsize):
    vocab[i] = bytes([i])

  # initial pair counts + pair mapping to vocab
  paircounts = {}
  vocabmap = defaultdict(set)
  for byteword, count in pretokenized_vocab.items():
    for i in range(len(byteword) - 1):
      pair = (byteword[i], byteword[i + 1])
      paircounts[pair] = paircounts.get(pair, 0) + count
      vocabmap[pair].add(byteword)

  # merge until desired vocab size
  merges = []
  while currsize < (vocab_size - len(special_tokens)):
    if len(paircounts) == 0:
      break
    
    bestpair = tuple()
    bestcount = 0

    # prioritize lexicographically greater
    for pair, count in paircounts.items():
      if count > bestcount:
        bestpair = pair
        bestcount = count
      elif count == bestcount and pair >  bestpair:
        bestpair = pair
    
    merges.append(bestpair)
    vocab[currsize] = bestpair[0] + bestpair[1]
    currsize += 1
    paircounts[bestpair] = 0

    # update the counts and dict
    for byteword in vocabmap[bestpair]:
      currcount = pretokenized_vocab[byteword]

      # use a sliding window to update/create new pairs adjacent to the bestpair
      # start with the start and end as edge case
      if (len(byteword) <= 2):
        continue

      i = 1
      
      if ((byteword[0], byteword[1]) == bestpair):
        rightpair = (byteword[1], byteword[2])
        newrightpair = (byteword[0] + byteword[1], byteword[2])

        paircounts[rightpair] -= currcount
        paircounts[newrightpair] = paircounts.get(newrightpair, 0) + currcount
        vocabmap[newrightpair].add(byteword)

        i += 1
      
      while i < (len(byteword) - 3):
        if ((byteword[i], byteword[i + 1]) == bestpair):
          leftpair = (byteword[i - 1], byteword[i])
          rightpair = (byteword[i + 1], byteword[i + 2])

          newleftpair = (byteword[i - 1], byteword[i] + byteword[i + 1])
          newrightpair = (byteword[i] + byteword[i + 1], byteword[i + 2])

          paircounts[leftpair] -= currcount
          paircounts[newleftpair] = paircounts.get(newleftpair, 0) + currcount
          vocabmap[newleftpair].add(byteword)
          
          paircounts[rightpair] -= currcount
          paircounts[newrightpair] = paircounts.get(newrightpair, 0) + currcount
          vocabmap[newrightpair].add(byteword)

          i += 2
        else:
          i += 1

      if (i == (len(byteword) - 2) and (byteword[i], byteword[i + 1]) == bestpair):
        leftpair = (byteword[i - 1], byteword[i])
        newleftpair = (byteword[i - 1], byteword[i] + byteword[i + 1])

        paircounts[leftpair] -= currcount
        paircounts[newleftpair] = paircounts.get(newleftpair, 0) + currcount
        vocabmap[newleftpair].add(byteword)

    vocabmap[bestpair] = set()

  # add the special tokens
  for i in range(len(special_tokens)):
    vocab[currsize] = special_tokens[i].encode("utf-8")
    currsize += 1

  return (vocab, merges)