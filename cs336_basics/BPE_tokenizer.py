from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re

def train_bpe(
    input_path: str,
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with (input_path, "rb") as f: # read binary
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        
        # int : bytes
        vocab: dict[int, bytes] = {i : str(i).encode("utf-8") for i in range(256)}
        vocab_bytes_to_int: dict[bytes, int] = {str(i).encode("utf-8") : i for i in range(256)}
        merges: list[tuple[bytes, bytes]] = []
        
        # (chunk_id, word, bytes_pair) : counts
        cache: dict[tuple[int, list[int], tuple[bytes, bytes]], int]= dict()
        
        # Pre-tokenization
        chunk_id = 0
        for start, end in zip(boundaries[:-1], boundaries[1:]): # todo: parallelize
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            
            for word in re.finditer(PAT, chunk):
                word: list[int] = list(word)
                for byte_id in len(word):
                    if (chunk_id, word, (word(byte_id), word(byte_id+1))) in cache.keys:
                        cache[(chunk_id, word, (word(byte_id), word(byte_id+1)))] += 1
                    else:
                        cache[(chunk_id, word, (word(byte_id), word(byte_id+1)))] = 1
                
            chunk_id += 1
            
        # Train Tokenizer
        for i in range(vocab_size - 256):
            bytes_pairs_with_count = count_bytes_pairs(cache)
            most_frequent_bytes_pair = get_most_frequent_bytes_pairs(bytes_pairs_with_count)
            vocab[i+256] = most_frequent_bytes_pair
            vocab_bytes_to_int[most_frequent_bytes_pair] = i + 256
            merges.append(most_frequent_bytes_pair)
            
            for _, word, bytes_pair in cache:
                word_need_to_recount: set[list[int]] = set()
                if bytes_pair == most_frequent_bytes_pair:
                    word_need_to_recount.add(word)
                    
            for _, word, _ in cache:
                if word in word_need_to_recount:
                    cache = recount_word(cache, word, vocab_bytes_to_int, merges)

    return vocab, merges

def count_bytes_pairs(cache: dict[tuple[int, list[int], tuple[bytes, bytes]], int]) -> dict[tuple[bytes, bytes], int]:
    count_dict: dict[tuple[bytes, bytes], int] = dict()
    for _, _, bytes_pair in cache.keys:
        if bytes_pair not in count_dict.keys:
            count_dict[bytes_pair] = 1
        else:
            count_dict[bytes_pair] += 1

    return count_dict

def get_most_frequent_bytes_pairs(count_dict: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    max_count = max(count_dict, key=count_dict.get)
    max_items = [(bytes_pair, count) for (bytes_pair, count) in count_dict.items() if count == max_count]

    # Choose the lexicographically greatest bytes_pair
    max_item = max(max_items, key=max_items[0])
    
    return max_item

def recount_word(cache: dict[tuple[int, list[int], tuple[bytes, bytes]], int], 
                 word_need_to_recound: list[int],
                 vocab_bytes_to_int: dict[bytes, int],
                 merges: list[tuple[bytes, bytes]]) -> dict[tuple[int, list[int], tuple[bytes, bytes]], int]:
    for chunk_id, word, bytes_pair in cache:
        if word == word_need_to_recound:
            cache[chunk_id, word, bytes_pair] = 0
            
    for merge in merges:
        merge_int_1, merge_int_2 = vocab_bytes_to_int[merge[0]], vocab_bytes_to_int[merge[1]]
        index = 0
        merge_in_vocab = vocab_bytes_to_int[merge]
        for int1, int2 in zip(word, word[1:]):
            if int1 == merge_int_1 and int2 == merge_int_2:
                del word[index]
                del word[index+1]
                
                word.insert[index, merge_in_vocab]
    
    for byte_id in len(word):
        if (chunk_id, word, (word(byte_id), word(byte_id+1))) in cache.keys:
            cache[(chunk_id, word, (word(byte_id), word(byte_id+1)))] += 1
        else:
            cache[(chunk_id, word, (word(byte_id), word(byte_id+1)))] = 1
            
    return cache
    
        
    