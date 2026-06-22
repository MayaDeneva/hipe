# hipe/models/xlmr.py
import numpy as np
from hipe import config as cfg
from hipe.models.base import RelationModel
from hipe.models import registry
from hipe.features.markers import (marked_text, scheme_marker_tokens,
                                    scheme_pool_markers)


def _quiet_hf():
    """Silence HuggingFace download bars + verbose logs (Kaggle renders tqdm badly)."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass


def _class_weights(labels, label_list):
    """Inverse-frequency (balanced) weights aligned to label_list order."""
    import torch
    counts = np.array([max(1, labels.count(l)) for l in label_list], dtype=float)
    w = counts.sum() / (len(label_list) * counts)
    return torch.tensor(w, dtype=torch.float)


def _build_module(model_name, n_at, n_isat, vocab_size, dropout, markers):
    """Full R-BERT pooling: average each entity's SPAN tokens (between its
    markers), push each entity (shared FC) and [CLS] (own FC) through dense->tanh,
    concatenate, then classify. markers = (e1_start, e1_end, e2_start, e2_end)."""
    import torch
    import torch.nn as nn
    from transformers import AutoModel
    e1s, e1e, e2s, e2e = markers

    class _FC(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.dense = nn.Linear(h, h)

        def forward(self, x):
            return torch.tanh(self.dense(self.dropout(x)))

    class _RBERT(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            self.encoder.resize_token_embeddings(vocab_size)
            h = self.encoder.config.hidden_size
            self.cls_fc = _FC(h)
            self.ent_fc = _FC(h)                 # shared between the two entities
            self.dropout = nn.Dropout(dropout)
            self.at_head = nn.Linear(3 * h, n_at)
            self.isat_head = nn.Linear(3 * h, n_isat)

        @staticmethod
        def _pos(input_ids, marker_id):
            mask = input_ids == marker_id
            idx = mask.float().argmax(dim=1).long()
            return torch.where(mask.any(dim=1), idx, torch.zeros_like(idx))

        def _span_mean(self, H, input_ids, start_id, end_id):
            B, L, _ = H.shape
            pos = torch.arange(L, device=H.device).unsqueeze(0)         # (1, L)
            s = self._pos(input_ids, start_id).unsqueeze(1)             # (B, 1)
            e = self._pos(input_ids, end_id).unsqueeze(1)
            mask = ((pos > s) & (pos < e)).unsqueeze(-1).float()        # tokens between markers
            cnt = mask.sum(dim=1)                                       # (B, 1)
            mean = (H * mask).sum(dim=1) / cnt.clamp(min=1.0)
            # fall back to the start-marker token if the span is empty/truncated
            start_h = H[torch.arange(B, device=H.device), self._pos(input_ids, start_id)]
            empty = (cnt == 0).float()
            return mean * (1.0 - empty) + start_h * empty

        def forward(self, input_ids, attention_mask):
            H = self.encoder(input_ids=input_ids,
                             attention_mask=attention_mask).last_hidden_state
            cls = self.cls_fc(H[:, 0])
            e1 = self.ent_fc(self._span_mean(H, input_ids, e1s, e1e))
            e2 = self.ent_fc(self._span_mean(H, input_ids, e2s, e2e))
            rep = self.dropout(torch.cat([cls, e1, e2], dim=-1))
            return self.at_head(rep), self.isat_head(rep)

    return _RBERT()


@registry.register("xlmr")
class XLMRModel(RelationModel):
    """Entity-marker XLM-R: a SHARED encoder with two heads (at, isAt) trained
    jointly with per-head class-weighted cross-entropy. `epochs` is a CEILING:
    after each epoch we score macro-recall on an internal validation split (held
    out of train, document-grouped), keep the best checkpoint, and early-stop
    with `patience`. The held-out harness dev is never used for selection."""
    name = "xlmr"

    def __init__(self, model_name="xlm-roberta-base", epochs=8, batch_size=16,
                 lr=2e-5, max_length=192, dropout=0.1, weight_decay=0.01,
                 marker_scheme="plain", add_date=False, add_kb=False,
                 dual_scope=False,
                 val_frac=0.15, patience=2, max_train=None, seed=0):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.marker_scheme = marker_scheme
        self.add_date = add_date
        self.add_kb = add_kb
        self.dual_scope = dual_scope        # at<-wide context, isAt<-narrow context
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
        _quiet_hf()
        if self.max_train is not None:
            train = train[:self.max_train]
        set_seed(self.seed)

        # internal validation split for checkpoint selection / early stopping
        tr, va = split_by_document(train, dev_frac=self.val_frac, seed=self.seed)
        if not tr or not va:                 # too few documents to split
            tr, va = train, []

        at2id = {l: i for i, l in enumerate(cfg.AT_LABELS)}
        isat2id = {l: i for i, l in enumerate(cfg.ISAT_LABELS)}
        def _mk(p, scope):
            return marked_text(p, self.marker_scheme, self.add_date, self.add_kb, scope)
        texts = [_mk(p, "wide") for p in tr]
        at_y = [at2id[p.gold_at] for p in tr]
        isat_y = [isat2id[p.gold_isat] for p in tr]

        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.add_special_tokens({"additional_special_tokens":
                                     scheme_marker_tokens(self.marker_scheme, self.add_date)})
        enc = self.tok(texts, truncation=True, max_length=self.max_length)
        # narrow view (for the isAt head) — same encoding as wide unless dual_scope
        enc_n = (self.tok([_mk(p, "narrow") for p in tr], truncation=True,
                          max_length=self.max_length) if self.dual_scope else enc)
        markers = tuple(self.tok.convert_tokens_to_ids(t)
                        for t in scheme_pool_markers(self.marker_scheme))

        self.module = _build_module(self.model_name, len(cfg.AT_LABELS),
                                    len(cfg.ISAT_LABELS), len(self.tok),
                                    self.dropout, markers)
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
                        enc_n["input_ids"][i], enc_n["attention_mask"][i],
                        at_y[i], isat_y[i])

        def _pad(seqs, fill):
            m = max(len(s) for s in seqs)
            return torch.tensor([s + [fill] * (m - len(s)) for s in seqs])

        def collate(batch):
            return (_pad([b[0] for b in batch], pad), _pad([b[1] for b in batch], 0),
                    _pad([b[2] for b in batch], pad), _pad([b[3] for b in batch], 0),
                    torch.tensor([b[4] for b in batch]),
                    torch.tensor([b[5] for b in batch]))

        loader = DataLoader(_DS(), batch_size=self.batch_size, shuffle=True,
                            collate_fn=collate)
        # weight decay on everything except biases / LayerNorm (standard recipe)
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        grouped = [
            {"params": [p for n, p in self.module.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self.weight_decay},
            {"params": [p for n, p in self.module.named_parameters()
                        if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(grouped, lr=self.lr)
        from transformers import get_linear_schedule_with_warmup
        total_steps = max(1, len(loader) * self.epochs)
        sched = get_linear_schedule_with_warmup(
            opt, int(0.1 * total_steps), total_steps)   # 10% LR warmup, then decay
        ce = torch.nn.functional.cross_entropy

        best_global, best_state, bad = -1.0, None, 0
        tr_sample = tr[:600]
        for epoch in range(self.epochs):
            self.module.train()
            for wid, wmask, nid, nmask, at, isat in loader:
                wid, wmask = wid.to(self._device), wmask.to(self._device)
                at, isat = at.to(self._device), isat.to(self._device)
                opt.zero_grad()
                at_log, isat_log = self.module(wid, wmask)
                if self.dual_scope:                      # isAt head reads the narrow view
                    nid, nmask = nid.to(self._device), nmask.to(self._device)
                    _, isat_log = self.module(nid, nmask)
                loss = ce(at_log, at, weight=at_w) + ce(isat_log, isat, weight=isat_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
                opt.step()
                sched.step()
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

    def _dual_infer(self, pairs):
        """at probs from the WIDE view; isAt probs from the NARROW view (dual_scope)."""
        def mk(scope):
            return [marked_text(p, self.marker_scheme, self.add_date, self.add_kb, scope)
                    for p in pairs]
        at_p, is_p = self._infer(mk("wide"))
        if self.dual_scope:
            _, is_p = self._infer(mk("narrow"))
        return at_p, is_p

    def _eval_global(self, pairs):
        from hipe.eval.metrics import macro_recall
        from hipe.models.base import apply_consistency
        at_p, is_p = self._dual_infer(pairs)
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
        at_p, is_p = self._dual_infer(pairs)
        out = []
        for a, s in zip(at_p, is_p):
            out.append({
                "at": cfg.AT_LABELS[int(a.argmax())],
                "isAt": cfg.ISAT_LABELS[int(s.argmax())],
                "at_proba": {l: float(a[k]) for k, l in enumerate(cfg.AT_LABELS)},
                "isAt_proba": {l: float(s[k]) for k, l in enumerate(cfg.ISAT_LABELS)},
            })
        return out
