import os
import torch
import torch.distributed as dist
from torch import Tensor
from wandb import Run
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from torch.optim.adamw import AdamW
from .model import MLPAutoEncoder


def setup_distributed(requested_device: str | torch.device | None = None) -> tuple[torch.device, int, int]:
    """Initialize DDP when this process was launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        if requested_device is not None:
            return torch.device(requested_device), 0, 1
        return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return torch.device("cuda", local_rank), dist.get_rank(), world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def train(
    model: MLPAutoEncoder,
    dataset: Tensor,
    optimizer: AdamW,
    batch_size: int,
    criterion,
    device: str,
    num_steps: int,
    print_every: int,
    run: Run
):
    device, rank, world_size = setup_distributed(device)
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])
    model.train()

    tensor_dataset = TensorDataset(dataset)
    sampler = DistributedSampler(tensor_dataset, shuffle=True) if world_size > 1 else None
    data_loader = DataLoader(
        tensor_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        drop_last=True,
    )
    data_iterator = iter(data_loader)

    for step in range(num_steps):
        try:
            (x,) = next(data_iterator)
        except StopIteration:
            if sampler is not None:
                sampler.set_epoch(step)
            data_iterator = iter(data_loader)
            (x,) = next(data_iterator)
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad()

        x_hat = model(x)
        loss = criterion(x_hat, x)

        loss.backward()
        optimizer.step()

        x_rounded = torch.round(x_hat)
        correct = (x_rounded == x).sum().float()
        total = torch.tensor(x.numel(), device=device, dtype=torch.float32)
        if world_size > 1:
            acc_totals = torch.stack([correct, total])
            dist.all_reduce(acc_totals, op=dist.ReduceOp.SUM)
            correct, total = acc_totals
        acc = (correct / total).item()

        if _is_main_process(rank) and run is not None:
            run.log({
                "train/loss": loss.item(),
                "train/acc": acc,
                "train/step": step,
            })

        if _is_main_process(rank) and (step + 1) % print_every == 0:
            print({
                f"step {step+1:6d} | "
                f"loss {loss.item():.4f} | "
                f"acc {acc:.4f} | "
            })


@torch.no_grad()
def evaluate(
    model: MLPAutoEncoder,
    dataset: Tensor,
    criterion,
    device: str,
) -> tuple[float, float]:
    device, _, world_size = setup_distributed(device)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_samples = 0

    x = dataset.to(device)

    x_hat = model(x)
    loss = criterion(x_hat, x)
    total_loss += loss.item() * x.size(0)
    total_samples += x.size(0)

    x_rounded = torch.round(x_hat)
    correct = (x_rounded == x).sum().float()
    total = torch.tensor(x.numel(), device=device, dtype=torch.float32)

    if world_size > 1:
        totals = torch.stack([
            torch.tensor(total_loss, device=device),
            torch.tensor(total_samples, device=device, dtype=torch.float32),
            correct,
            total,
        ])
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_samples, correct, total = totals.tolist()

    avg_loss = total_loss / total_samples
    acc = correct / total

    return avg_loss, acc
        