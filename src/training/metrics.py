def accuracy(predictions, targets):
    correct = (predictions == targets).sum().item()

    total = targets.numel()

    return correct / total
