#%%
from dataclasses import dataclass, field
import torch

@dataclass
class DataPacket:

    # Data that should be transferred
    contour: torch.Tensor = None
    positions: torch.Tensor = None  
    #positions: torch.Tensor = None
    # Control stuff
    stop: bool = field(default=False)
    okay: bool = field(default=False)


if __name__ == "__main__":
    test = DataPacket()
    print(test)
# %%
    