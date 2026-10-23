import pandas as pd, numpy as np

def augment_text(text, methods=None):
    return text

def augment_df(df, text_col="text", label_col="label", target_ratio=0.3, methods=None):
    panic = df[df[label_col] == 1]; non_panic = df[df[label_col] == 0]
    n_needed = int(len(non_panic) * target_ratio / (1 - target_ratio)) - len(panic)
    if n_needed <= 0: return df
    rows = [{text_col: augment_text(panic.iloc[i % len(panic)][text_col]), label_col: 1} for i in range(n_needed)]
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
