import torch
import torch.nn as nn
from model import TransformerModel
from vocab import Vocabulary
from utils import load_data
import random

def create_masks(src, tgt):
    src_key_padding_mask = (src == 0)  # Форма: (batch_size, seq_len)
    tgt_key_padding_mask = (tgt == 0)  # Форма: (batch_size, seq_len)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(src.device)
    return None, tgt_mask, src_key_padding_mask, tgt_key_padding_mask

def train_model(model, data, nhead=4, epochs=60, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_size = 32  
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        random.shuffle(data)  
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            src_batch = torch.stack([pair[0] for pair in batch]).to(device)  # Форма: (batch_size, seq_len)
            tgt_batch = torch.stack([pair[1] for pair in batch]).to(device)  # Форма: (batch_size, seq_len)

            optimizer.zero_grad()
            src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_masks(src_batch, tgt_batch[:, :-1])
            output = model(src_batch, tgt_batch[:, :-1], src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask)
            loss = criterion(output[:, :-1].contiguous().view(-1, output.size(-1)), tgt_batch[:, 1:].contiguous().view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Эпоха {epoch + 1}, Потери: {total_loss / (len(data) // batch_size)}")

    torch.save(model.state_dict(), "chatbot_model.pth")
    return total_loss / (len(data) // batch_size)

if __name__ == "__main__":
    train_data = load_data("data.txt")
    vocab = Vocabulary()
    vocab.build_vocab([pair[0] for pair in train_data] + [pair[1] for pair in train_data])
    torch.save(vocab, "vocab.pt")

    train_data = [(vocab.text_to_tensor(src), vocab.text_to_tensor(tgt)) for src, tgt in train_data]


    nhead = 4
    model = TransformerModel(vocab_size=len(vocab.get_itos()), d_model=512, nhead=nhead, num_layers=6)
    losses = train_model(model, train_data, nhead=nhead)
    print(f"Обучение завершено со средними потерями: {losses}")
