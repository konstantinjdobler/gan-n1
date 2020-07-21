import torch

class Store:
    @staticmethod
    def save(out_state, path, post_fix):
        torch.save(out_state, f"{path}{post_fix}.pt")

    @staticmethod
    def load(path):
        return torch.load(path)