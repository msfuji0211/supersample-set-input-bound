import numpy as np


class Model:
    def __init__(self, model_name, SAVE_DIR=None, config=None, from_saved=False, **kwargs):
        self.name = model_name
        self.SAVE_DIR = SAVE_DIR
        self.config = config

        self.calib_frac = config["calib_frac"]
        saved_epoch = kwargs["saved_epoch"] if "saved_epoch" in kwargs else 0
        save_scheme = kwargs["save_scheme"] if "save_scheme" in kwargs else "best-val-acc"

        assert save_scheme in ["best-val-acc", "all-epochs", "final"]

        if self.name != "ViT":
            raise ValueError(
                f"Unknown model name: {self.name}. This release supports ViT only."
            )

        if self.SAVE_DIR is None:
            raise ValueError("Must provide a save directory for ViT model.")
        if self.config["val_save_epoch"] > self.config["epochs"] - 1:
            raise ValueError(
                "Note: val_save_epoch must be <= (# epochs - 1); "
                "model only saved when (# epochs elapsed) > val_save_epoch."
            )

        from models.ViT import ViTClassifier

        self.model = ViTClassifier(
            self.SAVE_DIR,
            config,
            from_saved=from_saved,
            saved_epoch=saved_epoch,
            save_scheme=save_scheme,
        )

    def train(self, X_train, y_train, groups_train, X_val, y_val, groups_val):
        self.model.train(X_train, y_train, groups_train, X_val, y_val, groups_val)

    def predict_proba(self, X, with_logits=False):
        if self.calib_frac == 1.0:
            p = np.ones((X.shape[0], 2)) * 0.5
            if with_logits:
                return p[:, 1], p
            return p[:, 1]

        p = self.model.predict_proba(X, with_logits)
        if with_logits:
            return (p[0][:, 1], p[1])
        return p[:, 1]

    def predict(self, X):
        if self.calib_frac == 1.0:
            return np.random.choice([0, 1], size=X.shape[0])
        return self.model.predict(X)

    def load(self):
        pass
