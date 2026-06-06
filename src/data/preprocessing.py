import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ..config import CONFIG


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, task_type: str):
        super().__init__()
        self.task_type = task_type
        self.X = torch.from_numpy(X.astype(np.float32))
        if task_type == 'classification':
            self.y = torch.from_numpy(y.astype(np.int64))
        else:
            self.y = torch.from_numpy(y.astype(np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _raw_arrays(data_obj):
    """Приводит объект из sklearn/ucimlrepo/openml к (X_df, y_array)."""
    if hasattr(data_obj, 'data') and hasattr(data_obj.data, 'features'):
        X = data_obj.data.features
        y = data_obj.data.targets
    else:
        X = data_obj.data
        y = data_obj.target

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()
    y = np.asarray(y).ravel() if y.ndim > 1 else np.asarray(y)
    return X, y


def prepare_splits(data_obj, task_type: str, seed: int, split=None):
    """Возвращает (train_ds, val_ds, test_ds, meta)."""
    split = split or CONFIG["split"]
    X_df, y = _raw_arrays(data_obj)

    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = [c for c in X_df.columns if c not in cat_cols]

    if task_type == 'classification':
        unique_y = np.unique(y)
        label_map = {v: i for i, v in enumerate(unique_y)}
        y = np.array([label_map[v] for v in y], dtype=np.int64)
        num_classes = len(unique_y)
    else:
        y = y.astype(np.float32)
        num_classes = None

    test_frac = split["test"]
    val_frac_of_rest = split["val"] / (split["train"] + split["val"])

    stratify = y if task_type == 'classification' else None
    X_trval, X_test, y_trval, y_test = train_test_split(
        X_df, y, test_size=test_frac, random_state=seed, stratify=stratify
    )
    stratify2 = y_trval if task_type == 'classification' else None
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=val_frac_of_rest, random_state=seed, stratify=stratify2
    )

    def build_features(df_tr, df_val, df_test):
        num_scaler = StandardScaler()
        num_tr = num_scaler.fit_transform(df_tr[num_cols]) if num_cols else np.zeros((len(df_tr), 0))
        num_val = num_scaler.transform(df_val[num_cols]) if num_cols else np.zeros((len(df_val), 0))
        num_test = num_scaler.transform(df_test[num_cols]) if num_cols else np.zeros((len(df_test), 0))
        if cat_cols:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cat_tr = ohe.fit_transform(df_tr[cat_cols].astype(str))
            cat_val = ohe.transform(df_val[cat_cols].astype(str))
            cat_test = ohe.transform(df_test[cat_cols].astype(str))
        else:
            cat_tr = np.zeros((len(df_tr), 0))
            cat_val = np.zeros((len(df_val), 0))
            cat_test = np.zeros((len(df_test), 0))
        return (
            np.hstack([num_tr, cat_tr]).astype(np.float32),
            np.hstack([num_val, cat_val]).astype(np.float32),
            np.hstack([num_test, cat_test]).astype(np.float32),
        )

    Xtr, Xval, Xte = build_features(X_tr, X_val, X_test)

    meta = {
        "task_type": task_type,
        "model_type": "mlp",
        "in_dim": Xtr.shape[1],
        "num_classes": num_classes,
        "out_dim": num_classes if task_type == 'classification' else 1,
        "n_train": len(Xtr),
        "n_val": len(Xval),
        "n_test": len(Xte),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    if task_type == 'regression':
        y_mean = float(y_tr.mean())
        y_std = float(y_tr.std()) if y_tr.std() > 0 else 1.0
        y_tr_n = (y_tr - y_mean) / y_std
        y_val_n = (y_val - y_mean) / y_std
        y_te_n = (y_test - y_mean) / y_std
        meta["y_mean"] = y_mean
        meta["y_std"] = y_std
        ds_tr = TabularDataset(Xtr, y_tr_n, task_type)
        ds_val = TabularDataset(Xval, y_val_n, task_type)
        ds_test = TabularDataset(Xte, y_te_n, task_type)
    else:
        ds_tr = TabularDataset(Xtr, y_tr, task_type)
        ds_val = TabularDataset(Xval, y_val, task_type)
        ds_test = TabularDataset(Xte, y_test, task_type)

    return ds_tr, ds_val, ds_test, meta
