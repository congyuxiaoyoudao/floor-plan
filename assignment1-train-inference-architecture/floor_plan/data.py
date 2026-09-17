import json
import os
import numpy as np
import torch
from torch import Tensor

def get_data(src: str | os.PathLike) -> dict:
    """
    Get data from a given path. 
    shape like layout id: layout - room - point
    return a dict
    """
    with open(src, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def sample_room_data(data: dict, num_points: int) -> Tensor:
    """
    Sample a small batch of room data with a given condition.
    return a tensor
    """
    rooms = [
        room 
        for layout in data.values()   
        for room in layout
        if len(room) == num_points
    ]
    return torch.as_tensor(rooms, dtype=torch.float32)


def sample_batch(dataset: Tensor, num_batches: int) -> Tensor:
    idx = torch.randperm(dataset.shape[0], device=dataset.device)[:num_batches]
    sampled = dataset[idx]
    return sampled


if __name__ == '__main__':
    data = sample_room_data(get_data("../Room/val/Room_data.json"), 8)
    sample_batch(data, 8)
    print(data[0])