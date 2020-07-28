import torch

class Store:
    @staticmethod
    def save(out_state, path, post_fix):
        torch.save(out_state, f"{path}{post_fix}.pt")

    @staticmethod
    def load(path):
        cpu_only = not torch.cuda.is_available()
        
        if cpu_only:
            return torch.load(path, map_location=torch.device('cpu'))
        else:
            return torch.load(path)
