import torch

from src.model.masks import create_padding_mask, create_tgt_mask


def greedy_decode(
    model,
    src,
    sos_idx,
    eos_idx,
    max_len,
    device
):
    model.eval()

    src = src.to(device)
    if src.dim() == 1:
        src = src.unsqueeze(0)

    src_mask = create_padding_mask(src)

    with torch.no_grad():
        memory = model.encoder(src, src_mask)

    ys = torch.tensor([[sos_idx]], device=device)

    for _ in range(max_len - 1):
        tgt_mask = create_tgt_mask(ys)

        with torch.no_grad():
            out = model.decoder(
                ys,
                memory,
                tgt_mask=tgt_mask
            )

            logits = model.fc(out)

        next_token = logits[:, -1, :].argmax(dim=-1).item()
        next_token_tensor = torch.tensor([[next_token]], device=device)

        ys = torch.cat([ys, next_token_tensor], dim=1)

        if next_token == eos_idx:
            break

    return ys
