import pandas as pd
import numpy as np



class DataLoader:
    def __init__(self, path_vid, path_labels, path_train=None, path_val=None, path_test=None):
        self.path_vid = path_vid
        self.path_labels = path_labels
        self.path_train = path_train
        self.path_val = path_val
        self.path_test = path_test
        self.get_labels(path_labels)
        if self.path_train:
            self.train_df = self.load_video_labels(self.path_train)
        if self.path_val:
            self.val_df = self.load_video_labels(self.path_val)
        if self.path_test:
            self.test_df = self.load_video_labels(self.path_test, mode="input")

    def get_labels(self, path_labels):
        self.labels_df = pd.read_csv(path_labels, names=['label'])
        self.labels = self.labels_df['label'].tolist()
        self.n_labels = len(self.labels)
        self.label_to_int = {label: i for i, label in enumerate(self.labels)}
        self.int_to_label = {i: label for label, i in self.label_to_int.items()}

    def load_video_labels(self, path_subset, mode="label"):
        if mode == "input":
            names = ["video_id"]
        elif mode == "label":
            names = ["video_id", "label"]
        
        df = pd.read_csv(path_subset, sep=",", names=names)
        
        if mode == "label":
            # Convert string labels to integers
            df["label"] = df["label"].map(self.label_to_int)
        
        return df

