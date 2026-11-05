from dataloaders.CelebA import load_CelebA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


class Dataset:
    def __init__(
        self,
        dataset_name,
        split={"train": 0.6, "val": 0.2, "test": 0.2},
        groups="default",
        scale=False,
        class_balance=False,
        val_split_seed=43,
        min_group_size=0.005,
        verbose=True,
    ):
        """
        Parameters:
            :dataset_name: str, name of the dataset to load
            :split: dict, split fractions for train, val, test
            :groups: str, name of the group collection to use
            :scale: bool, whether to scale the data to mean 0, std 1
            :class_balance: bool, whether to balance the classes in the training data using SMOTE
            :val_split_seed: int, seed for splitting the training data into train and validation
            :min_group_size: float, minimum fraction of entire dataset that a group must have to
                            be included in the subgroup metrics
            :verbose: bool, whether to print group information
        """
        self.name = dataset_name
        self.min_group_size = min_group_size
        self.val_split_seed = val_split_seed
        self.train_test_val_split = split
        self.is_text = False

        self.scale = scale
        self.class_balance = class_balance

        if self.name == "CelebA":
            load_data_fn = load_CelebA
        else:
            raise ValueError(
                f"Unknown dataset name: {self.name}. "
                "This release supports CelebA only."
            )

        self.X, self.y, (self.groups, self.group_names) = load_data_fn(groups=groups)

        self._preprocess_data()
        self._check_data_leakage()

        if verbose:
            print(groups_info_str(self.X, self.y, self.groups, self.group_names))

    def train_calibration_split(self, frac, train_overlap=0, seed=43):
        (
            self.calibration_idxs,
            self.groups_calib,
            self.train_idxs_no_calib,
            self.groups_train_no_calib,
        ) = split_data(
            self.train_idxs,
            self.groups,
            int(len(self.train_idxs) * frac),
            seed=seed,
        )

        self.X_calib = self.X[self.calibration_idxs]
        self.y_calib = self.y[self.calibration_idxs]
        self.X_train_no_calib = self.X[self.train_idxs_no_calib]
        self.y_train_no_calib = self.y[self.train_idxs_no_calib]

        if train_overlap > 0:
            print(f"Introducing overlap between train / calibration data: {train_overlap:.2f}")
            np.random.seed(seed)
            num_samples = int(len(self.train_idxs_no_calib) * train_overlap)
            tc_idxs = np.random.choice(self.train_idxs_no_calib, num_samples, replace=False)

            self.calibration_idxs = sorted(np.concatenate([self.calibration_idxs, tc_idxs]))
            self.X_calib = self.X[self.calibration_idxs]
            self.y_calib = self.y[self.calibration_idxs]

            self.groups_calib = []
            for group in self.groups:
                g_idxs = reindex_group(self.calibration_idxs, group)
                self.groups_calib.append(g_idxs)

        return (
            self.X_train_no_calib,
            self.y_train_no_calib,
            self.groups_train_no_calib,
            self.X_calib,
            self.y_calib,
            self.groups_calib,
        )

    def _preprocess_data(self):
        groups_temp, group_names_temp = [], []
        for group, name in zip(self.groups, self.group_names):
            if len(group) > self.min_group_size * len(self.X):
                groups_temp.append(group)
                group_names_temp.append(name)
        self.groups = groups_temp
        self.group_names = group_names_temp
        self.group_info = {i: self.group_names[i] for i in range(len(self.group_names))}
        self.num_features = self.X.shape[1] if len(self.X.shape) > 1 else None

        self.num_classes = len(np.unique(self.y))
        if self.num_classes != 2:
            raise ValueError("Only binary classification supported")

        self.test_split_seed = 42

        if (
            self.train_test_val_split["test"]
            + self.train_test_val_split["val"]
            + self.train_test_val_split["train"]
            > 1
        ):
            raise ValueError("Split fractions must be at most 1")

        self.test_size = self.train_test_val_split["test"]
        self.val_size = self.train_test_val_split["val"]
        self.train_size = self.train_test_val_split["train"]

        self.n = len(self.X)
        (
            self.test_idxs,
            self.groups_test,
            train_val_idxs,
            _,
        ) = split_data(np.arange(self.n), self.groups, int(self.n * self.test_size), self.test_split_seed)
        self.X_test = self.X[self.test_idxs]
        self.y_test = self.y[self.test_idxs]

        (
            self.val_idxs,
            self.groups_val,
            self.train_idxs,
            self.groups_train,
        ) = split_data(train_val_idxs, self.groups, int(self.n * self.val_size), self.val_split_seed)
        self.X_val = self.X[self.val_idxs]
        self.y_val = self.y[self.val_idxs]

        self.X_train = self.X[self.train_idxs]
        self.y_train = self.y[self.train_idxs]

        if self.scale:
            self.X_train, self.X_val, self.X_test = self._scale_data()
            print("Scaling dataset.")
        if self.class_balance:
            print("Balancing classes in training data.")
            self.X_train, self.y_train, self.groups_train = self._smote()

    def _scale_data(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        return (
            scaler.transform(self.X_train),
            scaler.transform(self.X_val),
            scaler.transform(self.X_test),
        )

    def _smote(self):
        from imblearn.over_sampling import SMOTE as imbl_smote

        lbl_cnt_0 = np.sum(self.y_train == 0)
        lbl_cnt_1 = np.sum(self.y_train == 1)
        lbl_tot = len(self.y_train)

        sm = imbl_smote()
        balanced_idxs, _ = sm.fit_resample(
            np.arange(len(self.y_train)).reshape(-1, 1), self.y_train
        )
        X_train_balanced = self.X_train[balanced_idxs.flatten()]
        y_train_balanced = self.y_train[balanced_idxs.flatten()]
        groups_train_balanced = [
            reindex_group(balanced_idxs.flatten(), group) for group in self.groups_train
        ]

        space = " " * 8
        smt_lbl_cnt_0 = np.sum(y_train_balanced == 0)
        smt_lbl_cnt_1 = np.sum(y_train_balanced == 1)
        smt_lbl_tot = len(y_train_balanced)

        print("*" * 20)
        print("SMOTE called to balance classes in training data.")
        orig_prop_0_str = f"{lbl_cnt_0 / lbl_tot:.2%}"
        orig_prop_1_str = f"{lbl_cnt_1 / lbl_tot:.2%}"
        smote_prop_0_str = f"{smt_lbl_cnt_0 / smt_lbl_tot:.2%}"
        smote_prop_1_str = f"{smt_lbl_cnt_1 / smt_lbl_tot:.2%}"
        max_len = max(
            len(orig_prop_0_str),
            len(orig_prop_1_str),
            len(smote_prop_0_str),
            len(smote_prop_1_str),
        )
        print(
            f"{space}{'Class 0:':<20} (orig) {orig_prop_0_str:>{max_len}} -> (SMOTE) {smote_prop_0_str:>{max_len}}"
        )
        print(
            f"{space}{'Class 1:':<20} (orig) {orig_prop_1_str:>{max_len}} -> (SMOTE) {smote_prop_1_str:>{max_len}}"
        )
        print(f"{space}{'# samples:':<20} (orig) {lbl_tot:>{max_len}} -> (SMOTE) {smt_lbl_tot}")
        print("*" * 20)

        return X_train_balanced, y_train_balanced, groups_train_balanced

    def _check_data_leakage(self):
        for a, b, name in [
            (self.train_idxs, self.test_idxs, "train and test"),
            (self.train_idxs, self.val_idxs, "train and val"),
            (self.val_idxs, self.test_idxs, "val and test"),
        ]:
            if len(np.intersect1d(a, b)) > 0:
                raise ValueError(f"Data leakage: Overlap between {name}")

    def groups_info_df(self):
        return groups_info_df(self.X, self.y, self.groups, self.group_names)


def split_data(idxs, groups, num_in_p1, seed):
    np.random.seed(seed)
    p1_idxs = np.random.choice(idxs, num_in_p1, replace=False)
    p2_idxs = np.setdiff1d(idxs, p1_idxs)
    p1_idxs = np.sort(p1_idxs)
    p2_idxs = np.sort(p2_idxs)

    p1_groups = []
    p2_groups = []
    for group in groups:
        p1_groups.append(reindex_group(p1_idxs, group))
        p2_groups.append(reindex_group(p2_idxs, group))

    return p1_idxs, p1_groups, p2_idxs, p2_groups


def reindex_group(idxs, group_idxs):
    assert (idxs == np.sort(idxs)).all(), "idxs must be sorted"

    new_group_idxs = []
    sorted_idxs = idxs
    sorted_group_idxs = np.sort(group_idxs)
    i = j = 0
    while j < len(sorted_idxs) and i < len(sorted_group_idxs):
        if sorted_group_idxs[i] == sorted_idxs[j]:
            new_group_idxs.append(j)
            i += 1
            j += 1
        elif sorted_group_idxs[i] < sorted_idxs[j]:
            i += 1
        else:
            j += 1

    return new_group_idxs


def groups_info_df(X, y, groups, group_names):
    data = []
    n_samples = len(X)
    for i, group in enumerate(groups):
        size = len(group)
        frac = size / len(X)
        y_mean = 0.5 if size == 0 else np.mean(y[group])
        data.append([group_names[i], size, f"{frac:.4f}", f"{y_mean:.4f}"])

    data.append(["Dataset", n_samples, f"{n_samples / len(X):.4f}", f"{np.mean(y):.4f}"])
    return pd.DataFrame(data, columns=["group name", "n samples", "fraction", "y mean"])


def groups_info_str(X, y, gps, gp_names):
    from prettytable import PrettyTable

    ptab = PrettyTable()
    ptab.field_names = ["idx", "group name", "n samples", "fraction", "y-mean"]
    df = groups_info_df(X, y, gps, gp_names)
    n = len(df)
    for i, row in df.iterrows():
        ptab.add_row([i] + row.tolist(), divider=(i == n - 2))
    ptab.align = "r"
    ptab.align["group name"] = "l"
    return ptab.get_string()
