from cs336_basics.train_bpe_tokenizer import train_bpe
from pathlib import Path
import pickle

def save_tokenization(vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]], save_path:str) -> None:
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    vocab_file = save_path / "vocab.pkl"
    merges_file = save_path / "merges.pkl"
    
    with open(vocab_file, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_file, "wb") as f:
        pickle.dump(merges, f)
    print("save vocab.pkl and merges.pkl")
    
if __name__ == "__main__":
    input_path = Path(__file__).parent.parent / "data" /  "TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    save_tokenization(vocab, merges, save_path="/Users/hovsco/Documents/CSDIY/CS336/assignment1-basics/results")