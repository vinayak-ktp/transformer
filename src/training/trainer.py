from src.model.masks import create_padding_mask, create_tgt_mask


def train_one_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()

    total_loss = 0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        # Teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = create_padding_mask(src)
        tgt_mask = create_tgt_mask(tgt_input)

        logits = model(
            src,
            tgt_input,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )

        logits = logits.reshape(-1, logits.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(logits, tgt_output)

        optimizer.zero_grad()
        loss.backward()
        if scheduler is not None:
            scheduler.step()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
