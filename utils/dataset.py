import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EmotionalDataSet(Dataset):
    def __init__(self, df, root, cache_path="", cache_train=False, transformer=None):
        self.df = df
        self.transformer = transformer
        self.cache_train = cache_train
        self.cache = None
        self.root = root
        if cache_train:
            if os.path.isfile(cache_path):
                cache = torch.load(cache_path)
                if self.get_hash(df, root) == cache["hash"]:
                    self.cache = cache["dataa"]
            if self.cache is None:
                self.cache = self.get_cache()
                torch.save({
                    "hash": self.get_hash(df, root),
                    "data": self.cache
                }, cache_path)
                print("Saved cache file!!")

    def get_cache(self):
        cache = []
        for i in tqdm(range(len(self)), desc="Caching data: "):
            path = os.path.join(self.root, self.df.iloc[i, 0])
            c = self.df.iloc[i, 1]
            img = Image.open(path).convert('L')
            cache.append([img, c])
        return cache

    @staticmethod
    def get_hash(df, root):
        return len(df) + os.path.getsize(root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if not self.cache_train:
            path = os.path.join(self.root, self.df.iloc[idx, 0])
            cls = self.df.iloc[idx, 1]
            img = Image.open(path).convert('L')
        else:
            assert self.cache is not None
            img, cls = self.cache[idx]

        if self.transformer:
            img = self.transformer(img)

        return img, cls
