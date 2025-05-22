# utils.py
import torch

class Vocab:
    def __init__(self, vocab_dict, tokens):
        self.vocab_dict = vocab_dict
        self.tokens = tokens

    def __getitem__(self, key):
        return self.vocab_dict.get(key, self.vocab_dict['<unk>'])

    def get_itos(self):
        return self.tokens

    def __contains__(self, key):
        return key in self.vocab_dict

    def __reduce__(self):
        return (self.__class__, (self.vocab_dict, self.tokens))

def build_vocab(data):
    print("Starting build_vocab...")
    tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

    # Проверка данных и сбор токенов
    for i, pair in enumerate(data):
        if len(pair) != 2:
            print(f"Warning: Invalid pair at index {i}: {pair}")
            continue
        question, answer = pair
        if not question.strip() or not answer.strip():
            print(f"Warning: Empty question or answer at index {i}: {pair}")
            continue
        tokens.extend(question.strip().split())
        tokens.extend(answer.strip().split())

    # Удаление дубликатов
    tokens = list(set(tokens))
    print(f"Unique tokens collected: {len(tokens)}")

    # Создание словаря
    vocab_dict = {token: idx for idx, token in enumerate(tokens)}
    # Приоритет для специальных токенов
    vocab_dict['<pad>'] = 0
    vocab_dict['<sos>'] = 1
    vocab_dict['<eos>'] = 2
    vocab_dict['<unk>'] = 3

    print(f"Vocabulary size: {len(vocab_dict)}")

    vocab = Vocab(vocab_dict, tokens)
    print("Vocabulary built successfully")
    return vocab

def create_masks(src, tgt, pad_idx, nhead=4):
    batch_size, src_seq_len = src.shape
    _, tgt_seq_len = tgt.shape
    src_key_padding_mask = (src == pad_idx)  # [batch_size, src_seq_len]
    tgt_key_padding_mask = (tgt == pad_idx)  # [batch_size, tgt_seq_len]
    src_mask = None
    tgt_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)).bool()
    print(f"create_masks: src_key_padding_mask shape: {src_key_padding_mask.shape}")
    print(f"create_masks: tgt_key_padding_mask shape: {tgt_key_padding_mask.shape}")
    print(f"create_masks: tgt_mask shape: {tgt_mask.shape}")
    return src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask

def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                src = lines[i].strip()
                tgt = lines[i + 1].strip()
                data.append((src, tgt))
    return data

def text_to_tensor(text, vocab, max_len=20):
    tokens = ['<sos>'] + text.strip().split()[:max_len - 2] + ['<eos>']
    indices = []
    for token in tokens:
        if token not in vocab.word2idx:  # Проверка через word2idx
            print(f"Warning: token '{token}' not found in vocab, replaced with '<unk>'")
            indices.append(vocab.word2idx.get('<unk>', 0))  # Безопасный доступ
        else:
            indices.append(vocab.word2idx[token])  # Используем word2idx напрямую
    tensor = torch.tensor(indices, dtype=torch.long)
    if len(tensor) < max_len:
        tensor = torch.cat([tensor, torch.ones(max_len - len(tensor), dtype=torch.long) * vocab.word2idx.get('<pad>', 0)])
    print(f"text_to_tensor: Input text: {text}, Tensor: {tensor}")
    return tensor