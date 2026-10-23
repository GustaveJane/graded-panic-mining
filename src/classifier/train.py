import os, numpy as np, pandas as pd, torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from src.config import cfg, resolve

class PanicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length",
                             max_length=self.max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(self.labels[idx], dtype=torch.long)}

def train_model(train_df, val_df=None, model_dir=None, max_len=None, batch_size=None, epochs=None, lr=None, patience=None):
    c = cfg["classifier"]
    max_len = max_len or c["max_length"]; batch_size = batch_size or c["batch_size"]
    epochs = epochs or c["max_epochs"]; lr = lr or c["learning_rate"]; patience = patience or c["patience"]
    model_dir = model_dir or str(resolve("model_dir") / "panic_bert")
    os.makedirs(model_dir, exist_ok=True)
    tokenizer = BertTokenizer.from_pretrained(c["model_name"])
    model = BertForSequenceClassification.from_pretrained(c["model_name"], num_labels=2)
    train_ds = PanicDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len)
    train_load = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_load = None
    if val_df is not None:
        val_ds = PanicDataset(val_df["text"].tolist(), val_df["label"].tolist(), tokenizer, max_len)
        val_load = DataLoader(val_ds, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=0, num_training_steps=len(train_load)*epochs)
    best_loss, stop = float("inf"), 0
    for ep in range(epochs):
        model.train(); total = 0
        for b in train_load:
            opt.zero_grad()
            o = model(**{k: v.to(device) for k,v in b.items()})
            o.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); total += o.loss.item()
        if val_load:
            model.eval(); vl = 0
            with torch.no_grad():
                for b in val_load:
                    vl += model(**{k: v.to(device) for k,v in b.items()}).loss.item()
            av = vl / len(val_load); print(f"E{ep+1}: train={total/len(train_load):.4f} val={av:.4f}")
            if av < best_loss:
                best_loss = av; torch.save(model.state_dict(), os.path.join(model_dir, "panic_model.pt")); stop = 0
            else:
                stop += 1
                if stop >= patience: print(f"Early stop {ep+1}"); break
        else:
            print(f"E{ep+1}: train={total/len(train_load):.4f}")
    return model

def predict(model, texts, tokenizer=None, max_len=None, batch_size=None):
    c = cfg["classifier"]; max_len = max_len or c["max_length"]; batch_size = batch_size or c["batch_size"]
    if tokenizer is None: tokenizer = BertTokenizer.from_pretrained(c["model_name"])
    ds = PanicDataset(list(texts), [0]*len(texts), tokenizer, max_len)
    load = DataLoader(ds, batch_size=batch_size)
    device = next(model.parameters()).device; model.eval(); probs = []
    with torch.no_grad():
        for b in load:
            logits = model(**{k: v.to(device) for k,v in b.items()}).logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return np.array(probs)
