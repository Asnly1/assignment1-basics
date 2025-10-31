import multiprocessing
from pathlib import Path
import pickle
from collections.abc import Iterable
import regex as re
import bisect

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab: dict[int, bytes] = vocab
        self.vocab_to_int: dict[bytes, int] = {v : k for k,v in vocab.items()}
        self.merges: list[tuple[bytes, bytes]] = merges
        if special_tokens:
            for special_token in special_tokens:
                special_token_bytes = special_token.encode("utf-8")
                if special_token_bytes not in self.vocab_to_int:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = special_token_bytes
                    self.vocab_to_int[special_token_bytes] = new_id
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        self.special_tokens: list[str] | None = special_tokens
        if self.special_tokens:
            sorted_special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_token_bytes = [special_token.encode("utf-8") for special_token in sorted_special_tokens]
            self.longest_special_token_len = len(self.special_token_bytes[0])
            # Plus b"(" and b")"" to capture special tokens
            self.special_tokens_pattern = b"(" + b"|".join(re.escape(special_token_byte) for special_token_byte in self.special_token_bytes) + b")"
            self.compile_special_tokens_pattern = re.compile(self.special_tokens_pattern)
        else:
            self.longest_special_token_len = 0
            self.special_token_bytes = None
            self.special_tokens_pattern = b""
            self.compile_special_tokens_pattern = re.compile(self.special_tokens_pattern)
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab_filepath = Path(vocab_filepath)
        merges_filepath = Path(merges_filepath)
        
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
            
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
            
        return cls(vocab, merges, special_tokens)
    
    def find_chunk_boundaries(
        self,
        file: bytes,
        desired_num_chunks: int,
    ) -> list[int]:
        file_size = len(file)
        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size
        
        # pre-calculate the positions of all special tokens
        spans = [(m.start(), m.end()) for m in self.compile_special_tokens_pattern.finditer(file)]
        edges = []
        for s, e in spans:
            edges.append(s)
            edges.append(e)
        edges.sort()
        
        def snap(pos: int) -> int:
            # if pos falls into a span, snap it to the nearest side
            for s, e in spans:
                if s < pos < e:
                    return s if pos - s <= e - pos else e

            # if pos does not fall into a span, choose the nearest side
            i = bisect.bisect_left(edges, pos)
            candidates = [] # one or two candidates
            if i < len(edges):
                candidates.append(edges[i]) # the smallest position that is bigger than pos
            if i > 0:
                candidates.append(edges[i - 1]) # the greatest position that is smaller than pos
            if not candidates:
                return pos
            # first to compare the distance between pos and candidate
            # if the distance is the same, then choose the smaller position
            return min(candidates, key=lambda b: (abs(b - pos), b)) 

        for bi in range(1, len(chunk_boundaries) - 1):
            chunk_boundaries[bi] = snap(chunk_boundaries[bi])

        return chunk_boundaries
            
    def process_chunk(self, text_bytes: bytes, start:int, end:int) -> list[str]:
        chunk: bytes = text_bytes[start:end]
        if self.special_tokens_pattern:
            chunk_list: list[bytes] = re.split(self.compile_special_tokens_pattern, chunk)
        else:
            chunk_list: list[bytes] = [chunk]
        result = []
        
        for chunk in chunk_list:
            if not chunk:
            # skip "" caused by re.split
                continue 
            
            chunk: str = chunk.decode("utf-8", errors='replace')
            if self.special_tokens is not None and chunk in self.special_tokens:
                result.append(self.vocab_to_int[chunk.encode("utf-8")])
                continue
            for word in re.finditer(self.PAT, chunk):
                word: str = word.group(0)
                result.append(word)
                
        return result
    
    def encode(self, text: str) -> list[int]:
        text_bytes: bytes = text.encode("utf-8")
        num_processes = 4
        boundaries = self.find_chunk_boundaries(text_bytes, num_processes)
        tasks = []
        
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((text_bytes, start, end))
            
        with multiprocessing.Pool(processes=num_processes) as pool:
            results:list[list[str | int]] = pool.starmap(self.process_chunk, tasks)
        
        final_result = []
        
        for result in results:
            for word in result:
                if isinstance(word, int):
                    final_result.append(word)
                if isinstance(word, str):
                    word_bytes = word.encode("utf-8")
                    word_number: list[int] = list(self.vocab_to_int[word_bytes[i:i+1]] for i in range(len(word_bytes)))
                    # merge
                    for merge_pair in self.merges:
                        before_number_1, before_number_2 = self.vocab_to_int.get(merge_pair[0]), self.vocab_to_int.get(merge_pair[1])
                        new_number = self.vocab_to_int.get(merge_pair[0] + merge_pair[1])
                        
                        if before_number_1 is None or before_number_2 is None or new_number is None:
                            continue
                        
                        new_word_number: list[int] = []
                        index = 0
                        while index < len(word_number):
                            if index < len(word_number) -1 and word_number[index] == before_number_1 and word_number[index+1] == before_number_2:
                                new_word_number.append(new_number)
                                index += 2
                            else:
                                new_word_number.append(word_number[index])
                                index += 1
                        word_number = new_word_number
                    
                    final_result.extend(word_number)
                
        return final_result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        for item_line in iterable:
            item_line = item_line.encode("utf-8")
            if self.special_tokens_pattern:
                chunk_list: list[bytes] = re.split(self.compile_special_tokens_pattern, item_line)
            else:
                chunk_list: list[bytes] = [item_line]
            result = []
            
            for chunk in chunk_list:
                if not chunk:
                    continue
                
                chunk: str = chunk.decode("utf-8", errors='replace')
                
                if self.special_tokens is not None and chunk in self.special_tokens:
                    result.append(self.vocab_to_int[chunk.encode("utf-8")])
                    continue
                for word in re.finditer(self.PAT, chunk):
                    word: str = word.group(0)
                    result.append(word)
            
            final_result = []
        
            for word in result:
                if isinstance(word, int):
                    final_result.append(word)
                if isinstance(word, str):
                    word_bytes = word.encode("utf-8")
                    word_number: list[int] = list(self.vocab_to_int[word_bytes[i:i+1]] for i in range(len(word_bytes)))
                    # merge
                    for merge_pair in self.merges:
                        before_number_1, before_number_2 = self.vocab_to_int.get(merge_pair[0]), self.vocab_to_int.get(merge_pair[1])
                        new_number = self.vocab_to_int.get(merge_pair[0] + merge_pair[1])
                                                
                        if before_number_1 is None or before_number_2 is None or new_number is None:
                            continue
                        
                        new_word_number: list[int] = []
                        index = 0
                        while index < len(word_number):
                            if index < len(word_number) -1 and word_number[index] == before_number_1 and word_number[index+1] == before_number_2:
                                new_word_number.append(new_number)
                                index += 2
                            else:
                                new_word_number.append(word_number[index])
                                index += 1
                        word_number = new_word_number
                    
                    final_result.extend(word_number)
                    
            yield from final_result
            
    def decode(self, ids:list[int]) -> str:
        bytes_string = b""
        for id in ids:
            byte = self.vocab.get(id, b'\xef\xbf\xbd')
            bytes_string += byte
            
        return bytes_string.decode("utf-8", errors='replace')