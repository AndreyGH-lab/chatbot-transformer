import torch
from model import TransformerModel
from utils import build_vocab, text_to_tensor, create_masks

print("Loading vocabulary...")
vocab = torch.load('vocab.pt')

# Получение индексов специальных токенов через vocab.word2idx
pad_idx = vocab.word2idx['<pad>']
try:
    sos_idx = vocab.word2idx['<sos>'] if '<sos>' in vocab.word2idx else vocab.word2idx['<bos>']
except KeyError:
    raise ValueError("Словарь не содержит <sos> или <bos> токены!")
try:
    eos_idx = vocab.word2idx['<eos>']
except KeyError:
    raise ValueError("Словарь не содержит <eos> токен!")

print(f"Vocab size: {len(vocab.idx2word)}, <sos>/<bos> index: {sos_idx}, <eos> index: {eos_idx}")

# Инициализация модели
print("Initializing model...")
nhead = 8
model = TransformerModel(vocab_size=len(vocab.idx2word), d_model=512, nhead=nhead, num_layers=6)
model.load_state_dict(torch.load('chatbot_model.pth'))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model loaded on device: {device}")

def generate_response(input_text, max_len=20, nhead=8, top_k=50, top_p=0.9, temperature=0.7):
    print(f"Processing input: {input_text}")
    src = text_to_tensor(input_text, vocab, max_len).unsqueeze(0).to(device)
    print(f"Input tensor: {src}, max index: {src.max().item()}")

    if src.max().item() >= len(vocab.idx2word):
        raise ValueError(
            f"Input contains invalid token index: {src.max().item()} >= vocab_size {len(vocab.idx2word)}")

    tgt = torch.tensor([[sos_idx]], dtype=torch.long).to(device)
    response = []

    for i in range(max_len):
        print(f"Generation step {i + 1}/{max_len}")
        src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask = create_masks(src, tgt, pad_idx, nhead=nhead)
        print(f"src_mask shape: {src_mask.shape if src_mask is not None else 'None'}")
        print(f"tgt_mask shape: {tgt_mask.shape}, tgt shape: {tgt.shape}")

        with torch.no_grad():
            output = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                           src_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)

        if torch.isnan(output).any() or torch.isinf(output).any():
            print("Warning: Model output contains nan or inf!")
            print(f"Output sample: {output[0, -1, :10]}")
            return "Error: Model output is invalid."

        probs = torch.softmax(output[:, -1, :] / temperature, dim=-1)
        print(f"Output probs shape: {probs.shape}, max prob: {probs.max().item()}")

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[0, indices_to_remove] = 0
        probs = probs / probs.sum()

        next_token_idx = torch.multinomial(probs, num_samples=1).item()
        next_token = sorted_indices[0, next_token_idx].item()

        print(f"Next token: {vocab.idx2word[next_token]} (index: {next_token})")

        if next_token == eos_idx or i == max_len - 1:
            print("End of sequence or max length reached")
            break
        response.append(next_token)
        tgt = torch.cat([tgt, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=-1)

    response_text = ' '.join(vocab.idx2word[idx] for idx in response if idx not in [sos_idx, eos_idx, pad_idx])
    print(f"Generated response: {response_text}")
    return response_text

if __name__ == "__main__":
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        response = generate_response(user_input, nhead=nhead, top_k=50, top_p=0.9, temperature=0.7)
        print(f"Bot: {response}")
