class TransformerLRScheduler:
    def __init__(
        self,
        optimizer,
        embed_dim,
        warmup_steps=4000,
        factor=1.0,
        step_num=0
    ):
        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than 0")

        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be greater than 0")

        self.optimizer = optimizer
        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.step_num = step_num

    def get_lr(self, step_num=None):
        if step_num is None:
            step_num = self.step_num

        if step_num <= 0:
            return 0.0

        scale = self.embed_dim ** -0.5
        warmup = step_num * (self.warmup_steps ** -1.5)
        decay = step_num ** -0.5

        return self.factor * scale * min(warmup, decay)

    def step(self):
        self.step_num += 1
        lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def state_dict(self):
        return {
            "embed_dim": self.embed_dim,
            "warmup_steps": self.warmup_steps,
            "factor": self.factor,
            "step_num": self.step_num,
        }

    def load_state_dict(self, state_dict):
        self.embed_dim = state_dict["embed_dim"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.factor = state_dict["factor"]
        self.step_num = state_dict["step_num"]

        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
