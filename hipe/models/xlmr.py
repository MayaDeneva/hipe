# hipe/models/xlmr.py
import numpy as np
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.markers import marked_text, MARKER_TOKENS


def _class_weights(labels, label_list):
    """Inverse-frequency (balanced) weights aligned to label_list order."""
    import torch
    counts = np.array([max(1, labels.count(l)) for l in label_list], dtype=float)
    w = counts.sum() / (len(label_list) * counts)
    return torch.tensor(w, dtype=torch.float)


def _build_module(model_name, n_at, n_isat, vocab_size, dropout):
    """Shared encoder + two linear heads over the [CLS] representation."""
    import torch.nn as nn
    from transformers import AutoModel

    class _MultiTaskXLMR(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.encoder.resize_token_embeddings(vocab_size)
            h = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.at_head = nn.Linear(h, n_at)
            self.isat_head = nn.Linear(h, n_isat)

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.dropout(out.last_hidden_state[:, 0])   # [CLS]
            return self.at_head(pooled), self.isat_head(pooled)

    return _MultiTaskXLMR()


@registry.register("xlmr")
class XLMRModel(RelationModel):
    """Entity-marker XLM-R with a SHARED encoder and two classification heads
    (at, isAt), trained jointly (multi-task) with per-head class-weighted
    cross-entropy summed. One ~270M encoder serves both targets — half the size
    of two separate models — and the related tasks share representation."""
    name = "xlmr"

    def __init__(self, model_name="xlm-roberta-base", epochs=5, batch_size=16,
                 lr=2e-5, max_length=192, dropout=0.1, max_train=None, seed=0):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.dropout = dropout
        self.max_train = max_train
        self.seed = seed
        self.tok = None
        self.module = None
        self._device = None

    def fit(self, train, dev=None):
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, set_seed
        if self.max_train is not None:
            train = train[:self.max_train]
        set_seed(self.seed)

        at2id = {l: i for i, l in enumerate(cfg.AT_LABELS)}
        isat2id = {l: i for i, l in enumerate(cfg.ISAT_LABELS)}
        texts = [marked_text(p) for p in train]
        at_y = [at2id[p.gold_at] for p in train]
        isat_y = [isat2id[p.gold_isat] for p in train]

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
        enc = self.tok(texts, truncation=True, max_length=self.max_length)

        self.module = _build_module(self.model_name, len(cfg.AT_LABELS),
                                    len(cfg.ISAT_LABELS), len(self.tok), self.dropout)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.module.to(self._device)

        at_w = _class_weights([p.gold_at for p in train], cfg.AT_LABELS).to(self._device)
        isat_w = _class_weights([p.gold_isat for p in train], cfg.ISAT_LABELS).to(self._device)
        pad = self.tok.pad_token_id or 0

        class _DS(Dataset):
            def __len__(s):
                return len(at_y)

            def __getitem__(s, i):
                return (enc["input_ids"][i], enc["attention_mask"][i],
                        at_y[i], isat_y[i])

        def collate(batch):
            m = max(len(b[0]) for b in batch)
            ids, mask, at, isat = [], [], [], []
            for b in batch:
                n = m - len(b[0])
                ids.append(b[0] + [pad] * n)
                mask.append(b[1] + [0] * n)
                at.append(b[2])
                isat.append(b[3])
            return (torch.tensor(ids), torch.tensor(mask),
                    torch.tensor(at), torch.tensor(isat))

        loader = DataLoader(_DS(), batch_size=self.batch_size, shuffle=True,
                            collate_fn=collate)
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr)
        ce = torch.nn.functional.cross_entropy
        self.module.train()
        for _ in range(self.epochs):
            for ids, mask, at, isat in loader:
                ids, mask = ids.to(self._device), mask.to(self._device)
                at, isat = at.to(self._device), isat.to(self._device)
                opt.zero_grad()
                at_log, isat_log = self.module(ids, mask)
                loss = ce(at_log, at, weight=at_w) + ce(isat_log, isat, weight=isat_w)
                loss.backward()
                opt.step()

    def predict(self, pairs):
        import torch
        texts = [marked_text(p) for p in pairs]
        self.module.eval()
        out = []
        for i in range(0, len(texts), 32):
            batch = self.tok(texts[i:i + 32], truncation=True,
                             max_length=self.max_length, padding=True,
                             return_tensors="pt")
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.no_grad():
                at_log, isat_log = self.module(batch["input_ids"],
                                               batch["attention_mask"])
            at_p = torch.softmax(at_log, dim=-1).cpu().numpy()
            isat_p = torch.softmax(isat_log, dim=-1).cpu().numpy()
            for a, s in zip(at_p, isat_p):
                out.append({
                    "at": cfg.AT_LABELS[int(a.argmax())],
                    "isAt": cfg.ISAT_LABELS[int(s.argmax())],
                    "at_proba": {l: float(a[k]) for k, l in enumerate(cfg.AT_LABELS)},
                    "isAt_proba": {l: float(s[k]) for k, l in enumerate(cfg.ISAT_LABELS)},
                })
        return out
