import regex as re
import multiprocessing as mp
from collections import defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries


def _get_pretokenized_vocab(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    Uses given regex pattern to split the corpus into a pretokenized vocab to help with pair counting later. Also removes the special tokens via splitting.

    """

    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    specials = f"({'|'.join([re.escape(token) for token in special_tokens])})"
    splitted = re.split(specials, text)

    pretokenized_vocab = {}
    for spl in splitted:
        if spl in special_tokens:
            continue
        for m in re.finditer(PAT, spl):
            currstr = m.group()
            bytematch = currstr.encode("utf-8")
            match = tuple(bytematch[i:i+1] for i in range(len(bytematch)))
            pretokenized_vocab[match] = pretokenized_vocab.get(match, 0) + 1

    return pretokenized_vocab


def _counts_and_vocabmap(
    pretokenized_vocab: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """Builds the initial paircounts and reverse lookup vocabmap. The paircounts is the count/frequency of each pair in the entire corpus and the vocabmap is a map from each different pair to all words that contain it"""
    paircounts = {}
    vocabmap = defaultdict(set)

    for byteword, count in pretokenized_vocab.items():
        for i in range(len(byteword) - 1):
            pair = (byteword[i], byteword[i + 1])
            paircounts[pair] = paircounts.get(pair, 0) + count
            vocabmap[pair].add(byteword)

    return paircounts, vocabmap


def _apply_merge(
    bestpair: tuple[bytes, bytes],
    pretokenized_vocab: dict[tuple[bytes, ...], int],
    paircounts: dict[tuple[bytes, bytes], int],
    vocabmap: defaultdict[tuple[bytes, bytes], set[tuple[bytes, ...]]]
) -> None:
    """
    Updates counts and structures in-place based on the chosen merge (given by bestpair).
    """

    shallowcopy = set(vocabmap[bestpair])

    for byteword in shallowcopy:
        currcount = pretokenized_vocab[byteword]

        # 1. Delete all traces of old word
        for i in range(len(byteword) - 1):
            pair = (byteword[i], byteword[i + 1])
            paircounts[pair] -= currcount
            vocabmap[pair].discard(byteword)

        # 2. Construct new word
        construct = []
        i = 0
        while i < (len(byteword) - 1):
            pair = (byteword[i], byteword[i + 1])
            if pair == bestpair:
                construct.append(byteword[i] + byteword[i + 1])
                i += 2
            else:
                construct.append(byteword[i])
                i += 1
        if i == len(byteword) - 1:
            construct.append(byteword[i])

        new_word = tuple(construct)

        # 3. Add counts and vocabmap for new word
        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i + 1])
            paircounts[pair] = paircounts.get(pair, 0) + currcount
            vocabmap[pair].add(new_word)

        # 4. Update main ledger
        pretokenized_vocab.pop(byteword)
        pretokenized_vocab[new_word] = pretokenized_vocab.get(
            new_word, 0) + currcount

    vocabmap[bestpair] = set()


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a Byte Pair Encoding tokenizer
    """

    num_cores = max(1, mp.cpu_count() - 4)

    # Open file/corpus
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_cores, b"<|endoftext|>")

    pool_args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        pool_args.append((input_path, start, end, special_tokens))

    # Initialize vocab and counts
    with mp.Pool(processes=num_cores) as pool:
        vocabperchunk = pool.starmap(_get_pretokenized_vocab, pool_args)

    pretokenized_vocab = {}
    for minivocab in vocabperchunk:
        for word, count in minivocab.items():
            pretokenized_vocab[word] = pretokenized_vocab.get(word, 0) + count

    paircounts, vocabmap = _counts_and_vocabmap(pretokenized_vocab)

    currsize = 256
    vocab = {i: bytes([i]) for i in range(currsize)}
    merges = []

    # Merge until desired vocab size
    while currsize < (vocab_size - len(special_tokens)):
        # no pairs to merge
        if len(paircounts) == 0:
            break

        bestpair = tuple()
        bestcount = 0

        # Prioritize lexicographically greater pair
        for pair, count in paircounts.items():
            if count > bestcount:
                bestpair = pair
                bestcount = count
            elif count == bestcount and pair > bestpair:
                bestpair = pair

        if bestcount == 0:
            break

        merges.append(bestpair)
        vocab[currsize] = bestpair[0] + bestpair[1]
        currsize += 1
        paircounts[bestpair] = 0

        # update the counts and dict
        _apply_merge(bestpair, pretokenized_vocab, paircounts, vocabmap)
        # if currsize % 200 == 0:
        #   print(f"[{time.time()-t0:.0f}s] merges={currsize} unique_pairs={len(paircounts)} unique_words={len(pretokenized_vocab)}", flush=True)

    # add the special tokens
    for i in range(len(special_tokens)):
        vocab[currsize] = special_tokens[i].encode("utf-8")
        currsize += 1

    return (vocab, merges)
