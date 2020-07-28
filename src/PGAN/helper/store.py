import torch

class Store:
    @staticmethod
    def save(out_state, path, post_fix):
        torch.save(out_state, f"{path}{post_fix}.pt")

    @staticmethod
    def load(path):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        return torch.load(path, map_location=device)

