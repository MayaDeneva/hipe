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
    """Entity-marker XLM-R: a SHARED encoder with two heads (at, isAt) trained
    jointly with per-head class-weighted cross-entropy. `epochs` is a CEILING:
    after each epoch we score macro-recall on an internal validation split (held
    out of train, document-grouped), keep the best checkpoint, and early-stop
    with `patience`. The held-out harness dev is never used for selection."""
    name = "xlmr"

    def __init__(self, model_name="xlm-roberta-base", epochs=8, batch_size=16,
                 lr=2e-5, max_length=192, dropout=0.1, val_frac=0.15, patience=2,
                 max_train=None, seed=0):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.dropout = dropout
        self.val_frac = val_frac
        self.patience = patience
        self.max_train = max_train
        self.seed = seed
        self.tok = None
        self.module = None
        self._device = None

    def fit(self, train, dev=None):
        import copy
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoTokenizer, set_seed
        from hipe.data.split import split_by_document
        if self.max_train is not None:
            train = train[:self.max_train]
        set_seed(self.seed)

        # internal validation split for checkpoint selection / early stopping
        tr, va = split_by_document(train, dev_frac=self.val_frac, seed=self.seed)
        if not tr or not va:                 # too few documents to split
            tr, va = train, []

        at2id = {l: i for i, l in enumerate(cfg.AT_LABELS)}
        isat2id = {l: i for i, l in enumerate(cfg.ISAT_LABELS)}
        texts = [marked_text(p) for p in tr]
        at_y = [at2id[p.gold_at] for p in tr]
        isat_y = [isat2id[p.gold_isat] for p in tr]

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
        enc = self.tok(texts, truncation=True, max_length=self.max_length)

        self.module = _build_module(self.model_name, len(cfg.AT_LABELS),
                                    len(cfg.ISAT_LABELS), len(self.tok), self.dropout)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self.module.to(self._device)

        at_w = _class_weights([p.gold_at for p in tr], cfg.AT_LABELS).to(self._device)
        isat_w = _class_weights([p.gold_isat for p in tr], cfg.ISAT_LABELS).to(self._device)
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

        best_global, best_state, bad = -1.0, None, 0
        tr_sample = tr[:600]
        for epoch in range(self.epochs):
            self.module.train()
            for ids, mask, at, isat in loader:
                ids, mask = ids.to(self._device), mask.to(self._device)
                at, isat = at.to(self._device), isat.to(self._device)
                opt.zero_grad()
                at_log, isat_log = self.module(ids, mask)
                loss = ce(at_log, at, weight=at_w) + ce(isat_log, isat, weight=isat_w)
                loss.backward()
                opt.step()
            if va:
                tr_g = self._eval_global(tr_sample)
                va_g = self._eval_global(va)
                print(f"[xlmr] epoch {epoch + 1}/{self.epochs} "
                      f"train_macroR={tr_g:.4f} val_macroR={va_g:.4f}", flush=True)
                if va_g > best_global + 1e-4:
                    best_global = va_g
                    best_state = copy.deepcopy(
                        {k: v.cpu() for k, v in self.module.state_dict().items()})
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        print(f"[xlmr] early stop at epoch {epoch + 1}; "
                              f"best val_macroR={best_global:.4f}", flush=True)
                        break
        if best_state is not None:
            self.module.load_state_dict(best_state)
            self.module.to(self._device)

    def _infer(self, texts):
        import torch
        self.module.eval()
        ats, iss = [], []
        for i in range(0, len(texts), 32):
            batch = self.tok(texts[i:i + 32], truncation=True,
                             max_length=self.max_length, padding=True,
                             return_tensors="pt")
            batch = {k: v.to(self._device) for k, v in batch.items()}
            with torch.no_grad():
                at_log, isat_log = self.module(batch["input_ids"],
                                               batch["attention_mask"])
            ats.append(torch.softmax(at_log, dim=-1).cpu().numpy())
            iss.append(torch.softmax(isat_log, dim=-1).cpu().numpy())
        at_p = np.vstack(ats) if ats else np.empty((0, len(cfg.AT_LABELS)))
        is_p = np.vstack(iss) if iss else np.empty((0, len(cfg.ISAT_LABELS)))
        return at_p, is_p

    def _eval_global(self, pairs):
        from hipe.eval.metrics import macro_recall
        from hipe.models.base import apply_consistency
        at_p, is_p = self._infer([marked_text(p) for p in pairs])
        at_pred, is_pred = [], []
        for a, s in zip(at_p, is_p):
            d = {"at": cfg.AT_LABELS[int(a.argmax())],
                 "isAt": cfg.ISAT_LABELS[int(s.argmax())]}
            apply_consistency(d, "soft")
            at_pred.append(d["at"])
            is_pred.append(d["isAt"])
        at_t = [p.gold_at for p in pairs]
        is_t = [p.gold_isat for p in pairs]
        return (macro_recall(at_t, at_pred) + macro_recall(is_t, is_pred)) / 2

    def predict(self, pairs):
        at_p, is_p = self._infer([marked_text(p) for p in pairs])
        out = []
        for a, s in zip(at_p, is_p):
            out.append({
                "at": cfg.AT_LABELS[int(a.argmax())],
                "isAt": cfg.ISAT_LABELS[int(s.argmax())],
                "at_proba": {l: float(a[k]) for k, l in enumerate(cfg.AT_LABELS)},
                "isAt_proba": {l: float(s[k]) for k, l in enumerate(cfg.ISAT_LABELS)},
            })
        return out
