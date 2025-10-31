import multiprocessing
from pathlib import Path
import pickle
from collections.abc import Iterable
import regex as re

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
        self.special_token_bytes = [special_token.encode("utf-8") for special_token in self.special_tokens]
        # Plus b"(" and b")"" to capture special tokens
        if self.special_token_bytes:
            self.special_tokens_pattern = b"(" + b"|".join(re.escape(special_token_byte) for special_token_byte in self.special_token_bytes) + b")"
        else:
            self.special_tokens_pattern = b""
    
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
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        file_size = len(file)
        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            end_position = initial_position + mini_chunk_size
            while True:
                mini_chunk = file[initial_position:end_position]

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def process_chunk(self, text_bytes: bytes, start:int, end:int) -> list[str]:
        chunk: bytes = text_bytes[start:end]
        chunk_list: list[bytes] = re.split(self.special_tokens_pattern, chunk)
        result = []
        
        for chunk in chunk_list:
            if not chunk:
                continue # skip "" caused by re.split
            
            try:
                chunk: str = chunk.decode("utf-8")
            except UnicodeDecodeError:
                continue
            
            if chunk in self.special_tokens:
                result.append(self.vocab_to_int[chunk.encode("utf-8")])
                continue
            for word in re.finditer(self.PAT, chunk):
                word: str = word.group(0)
                result.append(word)
                
        return result
    
    def encode(self, text: str) -> list[int]:
        text_bytes: bytes = text.encode("utf-8")
        num_processes = 4
        boundaries = self.find_chunk_boundaries(text_bytes, num_processes, b"<|endoftext|>")
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
                    word_number: list[int] = list(word.encode("utf-8"))
                    # merge
                    for merge_pair in self.merges:
                        before_number_1, before_number_2 = self.vocab_to_int.get(merge_pair[0]), self.vocab_to_int.get(merge_pair[1])
                        new_number = self.vocab_to_int.get(merge_pair[0] + merge_pair[1])
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
            chunk_list: list[bytes] = re.split(self.special_tokens_pattern, item_line)
            result = []
            
            for chunk in chunk_list:
                if not chunk:
                    continue
                try:
                    chunk: str = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                
                if chunk in self.special_tokens:
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
                    word_number: list[int] = list(word.encode("utf-8"))
                    # merge
                    for merge_pair in self.merges:
                        before_number_1, before_number_2 = self.vocab_to_int.get(merge_pair[0]), self.vocab_to_int.get(merge_pair[1])
                        new_number = self.vocab_to_int.get(merge_pair[0] + merge_pair[1])
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