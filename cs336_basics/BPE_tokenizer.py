from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re

def train_bpe(
    input_path: str,
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with open(input_path, "rb") as f: # read binary
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        # int : bytes
        vocab: dict[int, bytes] = {i : bytes([i]) for i in range(256)}
        vocab_bytes_to_int: dict[bytes, int] = {bytes([i]) : i for i in range(256)}
        merges: list[tuple[bytes, bytes]] = []
        
        for i, special_token in enumerate(special_tokens):
            vocab[i+256] = special_token.encode("utf-8")
            vocab_bytes_to_int[special_token.encode("utf-8")] = i+256
        
        word_count: dict[tuple[int, ...], int] = dict()
        
        special_token_bytes = [special_token.encode("utf-8") for special_token in special_tokens]
        special_tokens_pattern = b"|".join(re.escape(special_token_byte) for special_token_byte in special_token_bytes )
        
        # Pre-tokenization
        for start, end in zip(boundaries[:-1], boundaries[1:]): # todo: parallelize
            f.seek(start)
            chunk = f.read(end - start)
            chunk_list: list[bytes] = re.split(special_tokens_pattern, chunk)
            for chunk in chunk_list:
                chunk = chunk.decode("utf-8")
                for word in re.finditer(PAT, chunk):
                    word_bytes: tuple[int, ...] = tuple(word.group(0).encode("utf-8"))
                    word_count[word_bytes] = word_count.get(word_bytes, 0) + 1
        
        # Map bytes_pair to its word  
        bytes_pair_to_word: dict[tuple[int, int], set[tuple[int, ...]]] = dict()
        # Get initial bytes_pair count
        bytes_pair_count: dict[tuple[int, int], int] = dict()
        
        for word, count in word_count.items():
            pair_list = get_pairs_from_word(word)
            for pair in pair_list:
                bytes_pair_count[pair] = bytes_pair_count.get(pair, 0) + count
                
                if pair not in bytes_pair_to_word.keys():
                    bytes_pair_to_word[pair] = set()
                bytes_pair_to_word[pair].add(word)
            
        # Train Tokenizer
        special_tokens_length = len(special_tokens)
        num_merge = vocab_size - 256 - special_tokens_length
        
        for i in range(num_merge):
            most_frequent_pair: tuple[int, int] = get_most_frequent_pair(bytes_pair_count, vocab)
            if most_frequent_pair is None:
                break
            
            token1_bytes = vocab.get(most_frequent_pair[0])
            token2_bytes = vocab.get(most_frequent_pair[1])
            new_token_bytes = token1_bytes + token2_bytes
            new_token_id = i+256+special_tokens_length
            
            vocab[new_token_id] = new_token_bytes
            vocab_bytes_to_int[new_token_bytes] = new_token_id
            merges.append((token1_bytes, token2_bytes))
            
            word_need_to_recount: list[tuple[int, ...]] = bytes_pair_to_word.get(most_frequent_pair)
            
            # Delete old pairs
            for word in word_need_to_recount:
                count = word_count.get(word)
                del word_count[word]
                
                pair_list = get_pairs_from_word(word)
                for pair in pair_list:
                    bytes_pair_to_word[pair].remove(word)
                    bytes_pair_count[pair] = bytes_pair_count.get(pair, 0) - count
                    if bytes_pair_count.get(pair) == 0:
                        del bytes_pair_count[pair]
            
            # Add new pairs
            for word in word_need_to_recount:
                word = replace_word_with_new_token(word, most_frequent_pair[0], most_frequent_pair[1], new_token_id)
                word_count[word] = count
                
                pair_list = get_pairs_from_word(word)
                for pair in pair_list:
                    bytes_pair_count[pair] = bytes_pair_count.get(pair, 0) + count
                    
                    if pair not in bytes_pair_to_word.keys():
                        bytes_pair_to_word[pair] = set()
                    bytes_pair_to_word[pair].add(word)

    return vocab, merges

def get_pairs_from_word(word: tuple[int, ...]) -> list[tuple[int, int]]:
    return [(byte1, byte2) for byte1, byte2 in zip(word, word[1:])]

def get_most_frequent_pair(bytes_pair_count: dict[tuple[int, int], int], vocab: dict[int, bytes]) -> tuple[int, int]:
    if not bytes_pair_count:
        return None
    
    max_count = max(bytes_pair_count.values())
    frquent_pairs = [bytes_pair for bytes_pair, count in bytes_pair_count.items() if count == max_count]
    
    def helper(bytes_pair: tuple[int, int]) -> tuple[bytes, bytes]:
        return vocab.get(bytes_pair[0]), vocab.get(bytes_pair[1])
    
    return max(frquent_pairs, key=helper)

def replace_word_with_new_token(word: tuple[int, ...], 
                                old_token_1:int, 
                                old_token_2:int, 
                                new_token: int) -> tuple[int, ...]:
    new_word: list[int] = []
    index = 0
    while index < len(word):
        if index < len(word) -1 and word[index] == old_token_1 and word[index+1] == old_token_2:
            new_word.append(new_token)
            index += 2
        else:
            new_word.append(word[index])
            index += 1
    
    return tuple(new_word)