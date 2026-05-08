import torch
import torch.nn.functional as F

from src.model.masks import create_padding_mask, create_tgt_mask


def beam_search_decode(
    model,
    src,
    sos_idx,
    eos_idx,
    max_len,
    beam_width,
    device
):
    model.eval()

    src = src.to(device)
    src_mask = create_padding_mask(src)

    with torch.no_grad():
        memory = model.encoder(src, src_mask)

    sequences = [(torch.tensor([[sos_idx]], device=device), 0)]

    for _ in range(max_len):
        all_candidates = []

        for seq, score in sequences:
            if seq[0, -1].item() == eos_idx:
                all_candidates.append((seq, score))
                continue

            tgt_mask = create_tgt_mask(seq)

            with torch.no_grad():
                out = model.decoder(seq, memory, tgt_mask=tgt_mask)
                logits = model.fc_out(out)

            probs = F.log_softmax(logits[:, -1, :], dim=-1)

            topk_probs, topk_idx = probs.topk(beam_width)

            for i in range(beam_width):
                candidate = torch.cat(
                    [seq, topk_idx[:, i].unsqueeze(0)],
                    dim=1
                )
                candidate_score = score + topk_probs[:, i].item()
                all_candidates.append((candidate, candidate_score))

        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]

    return sequences[0][0]
