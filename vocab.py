# vocab.py
import torch
from collections import Counter

class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.add_word('<pad>', 0)  # Токен заполнения
        self.add_word('<sos>', 1)  # Начало последовательности
        self.add_word('<eos>', 2)  # Конец последовательности
        self.add_word('<unk>', 3)  # Неизвестный токен

    def add_word(self, word, idx=None):
        if idx is not None:
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        else:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def build_vocab(self, sentences):
        for sentence in sentences:
            words = sentence.lower().split()
            self.word_freq.update(words)
        for word in self.word_freq:
            self.add_word(word)

    def text_to_tensor(self, text):
        words = text.lower().split()
        indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in words]
        indices = [self.word2idx['<sos>']] + indices + [self.word2idx['<eos>']]
        # Дополнение до максимальной длины 20
        if len(indices) < 20:
            indices.extend([self.word2idx['<pad>']] * (20 - len(indices)))
        else:
            indices = indices[:20]
        return torch.tensor(indices, dtype=torch.long)

    def get_itos(self):
        return self.idx2word

    def __len__(self):
        return len(self.word2idx)

    # Добавляем метод __getitem__ для поддержки индексации
    def __getitem__(self, key):
        return self.word2idx.get(key, self.word2idx.get('<unk>', 0))

    def __contains__(self, key):
        return key in self.word2idx