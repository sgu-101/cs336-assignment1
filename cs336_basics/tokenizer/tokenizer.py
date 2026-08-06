import pickle
import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # for encoding
        self.merges_dict = {pair: rank for rank,
                            pair in enumerate(self.merges)}
        self.vocab_dict = {pair: num for num, pair in self.vocab.items()}

    # assumes vocab and merges are saved as pickle
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        vocab = {}
        merges = []
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def _pretokenize(
        self,
        text: str,
        special_tokens: list[str] = None
    ) -> list[str]:
        """
        Splits text using the BPE regex. If special_tokens are provided,
        they are isolated and kept intact.
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        res = []

        if not special_tokens:
            for m in re.finditer(PAT, text):
                res.append(m.group())
            return res

        # Sort special tokens by length, longest first
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        specials_pattern = f"({'|'.join([re.escape(token) for token in sorted_specials])})"
        splitted = re.split(specials_pattern, text)

        for spl in splitted:
            if not spl:
                continue
            if spl in special_tokens:
                res.append(spl)
            else:
                for m in re.finditer(PAT, spl):
                    res.append(m.group())

        return res

    def encode(self, text: str) -> list[int]:
        regsplit = self._pretokenize(text, self.special_tokens)
        mdict = self.merges_dict
        vdict = self.vocab_dict

        res = []
        encodingdict = {}
        for t in regsplit:
            # special token check
            if t in self.special_tokens:
                res.append(vdict[t.encode("utf-8")])
                continue
            if t in encodingdict:
                res.extend(encodingdict[t])
                continue

            currword = tuple(bytes([b]) for b in t.encode("utf-8"))
            while True:
                bestpair = ()
                bestrank = float('inf')
                for i in range(0, len(currword) - 1):
                    currpair = (currword[i], currword[i + 1])
                    if currpair in mdict and mdict[currpair] < bestrank:
                        bestpair = currpair
                        bestrank = mdict[currpair]

                if bestrank == float('inf'):
                    break

                construct = []
                i = 0
                while i < (len(currword) - 1):
                    pair = (currword[i], currword[i + 1])
                    if pair == bestpair:
                        construct.append(currword[i] + currword[i + 1])
                        i += 2
                    else:
                        construct.append(currword[i])
                        i += 1
                if i == len(currword) - 1:
                    construct.append(currword[i])
                currword = tuple(construct)

            encoded = []
            for b in currword:
                encoded.append(vdict[b])

            encodingdict[t] = encoded
            res.extend(encoded)
        return res

    # lazy
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            chunk_ids = self.encode(chunk)
            yield from chunk_ids

    def decode(self, ids: list[int]) -> str:
        """
        decodes a sequence of ids by joining all the byte representation first and then decoding it all at once
        """
        total = b"".join([self.vocab[num] for num in ids])
        return total.decode("utf-8", errors='replace')
