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


class _Target:
    """One fine-tuned sequence classifier for a single relation target.

    Falls back to a constant prediction when <2 classes are present in training
    (a transformer cannot be trained on a single class)."""

    def __init__(self, label_list, model_name, max_length):
        self.label_list = label_list
        self.lab2id = {l: i for i, l in enumerate(label_list)}
        self.model_name = model_name
        self.max_length = max_length
        self.tok = None
        self.model = None
        self.const = None

    def train(self, texts, labels, *, epochs, batch_size, lr, seed):
        import torch
        from torch.utils.data import Dataset
        from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                  Trainer, TrainingArguments, DataCollatorWithPadding,
                                  set_seed)
        if len(set(labels)) < 2:
            self.const = labels[0] if labels else "FALSE"
            return
        set_seed(seed)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.add_special_tokens({"additional_special_tokens": MARKER_TOKENS})
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=len(self.label_list))
        self.model.resize_token_embeddings(len(self.tok))

        enc = self.tok(texts, truncation=True, max_length=self.max_length)
        y = [self.lab2id[l] for l in labels]

        class _DS(Dataset):
            def __len__(self_inner):
                return len(y)

            def __getitem__(self_inner, i):
                item = {k: enc[k][i] for k in enc}
                item["labels"] = y[i]
                return item

        weights = _class_weights(labels, self.label_list)

        class _WeightedTrainer(Trainer):
            def compute_loss(self_t, model, inputs, return_outputs=False, **kw):
                labels_ = inputs.pop("labels")
                outputs = model(**inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits, labels_, weight=weights.to(outputs.logits.device))
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=str(cfg.CACHE_DIR / "xlmr_tmp"),
            num_train_epochs=epochs, per_device_train_batch_size=batch_size,
            learning_rate=lr, logging_strategy="no", save_strategy="no",
            report_to=[], seed=seed)
        trainer = _WeightedTrainer(
            model=self.model, args=args, train_dataset=_DS(),
            data_collator=DataCollatorWithPadding(self.tok))
        trainer.train()

    def predict(self, texts):
        if self.const is not None:
            proba = {l: (1.0 if l == self.const else 0.0) for l in self.label_list}
            return [self.const] * len(texts), [dict(proba) for _ in texts]
        import torch
        self.model.eval()
        labels, probas = [], []
        for i in range(0, len(texts), 32):
            batch = self.tok(texts[i:i + 32], truncation=True,
                             max_length=self.max_length, padding=True,
                             return_tensors="pt")
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                logits = self.model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                j = int(row.argmax())
                labels.append(self.label_list[j])
                probas.append({l: float(row[k]) for k, l in enumerate(self.label_list)})
        return labels, probas


@registry.register("xlmr")
class XLMRModel(RelationModel):
    name = "xlmr"

    def __init__(self, model_name="xlm-roberta-base", epochs=3, batch_size=16,
                 lr=2e-5, max_length=192, max_train=None, seed=0):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.max_train = max_train
        self.seed = seed
        self._at = _Target(cfg.AT_LABELS, model_name, max_length)
        self._isat = _Target(cfg.ISAT_LABELS, model_name, max_length)

    def fit(self, train, dev=None):
        if self.max_train is not None:
            train = train[:self.max_train]
        texts = [marked_text(p) for p in train]
        kw = dict(epochs=self.epochs, batch_size=self.batch_size,
                  lr=self.lr, seed=self.seed)
        self._at.train(texts, [p.gold_at for p in train], **kw)
        self._isat.train(texts, [p.gold_isat for p in train], **kw)

    def predict(self, pairs):
        texts = [marked_text(p) for p in pairs]
        at, at_p = self._at.predict(texts)
        isat, isat_p = self._isat.predict(texts)
        return [{"at": a, "isAt": i, "at_proba": ap, "isAt_proba": ip}
                for a, i, ap, ip in zip(at, isat, at_p, isat_p)]
