"""Corner-watermark shortcut (a Decoy-MNIST-style benchmark, adapted to
the p4m symmetry).

A k x k watermark (k = side / 8, set to twice the image's maximum absolute
amplitude, then L2 renormalization) is placed top-left on class-0 and
top-right on class-1 TRAINING images: it predicts the label perfectly, but
only relative to the image frame. All four corners lie in one D4 orbit, so
an exactly invariant model cannot read the watermark position, whereas a
non-equivariant one can learn it as a shortcut.

Test sets built from the same test images:
  shortcut     watermark placed as in training (shortcut still valid)
  transformed  the watermarked images under a random D4 element
               (watermark in a random corner: shortcut broken)
  clean        no watermark
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

TEST_SETS = ("shortcut", "transformed", "clean")


def loader_tensors(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*[(x, y) for x, y in loader])
    return torch.cat(xs), torch.cat(ys)


def add_watermark(states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    side = int(round(states.shape[1] ** 0.5))
    k = max(1, side // 8)
    imgs = states.reshape(-1, side, side).clone()
    value = 2.0 * imgs.abs().flatten(1).max(dim=1).values
    for i in range(imgs.shape[0]):
        cols = slice(0, k) if labels[i] == 0 else slice(side - k, side)
        imgs[i, :k, cols] = value[i]
    imgs = imgs / imgs.flatten(1).norm(dim=1)[:, None, None]
    return imgs.reshape(states.shape[0], -1)


def random_d4(states: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    side = int(round(states.shape[1] ** 0.5))
    out = []
    for img in states.reshape(-1, side, side):
        g = int(torch.randint(8, (1,), generator=generator))
        img = torch.rot90(img, g % 4, dims=(0, 1))
        out.append(torch.flip(img, dims=[1]) if g >= 4 else img)
    return torch.stack(out).reshape(states.shape[0], -1)


def watermark_loaders(
    base_loaders: tuple, N: int, seed: int
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """(watermarked training loader, {test-set name: loader})."""
    x_tr, y_tr = loader_tensors(base_loaders[0])
    x_te, y_te = loader_tensors(base_loaders[1])
    batch = max(1, N // 10)
    train = DataLoader(TensorDataset(add_watermark(x_tr, y_tr), y_tr), batch_size=batch,
                       shuffle=True, generator=torch.Generator().manual_seed(seed))
    x_short = add_watermark(x_te, y_te)
    tests = {
        "shortcut": x_short,
        "transformed": random_d4(x_short, torch.Generator().manual_seed(1000 + seed)),
        "clean": x_te,
    }
    return train, {k: DataLoader(TensorDataset(x, y_te), batch_size=batch) for k, x in tests.items()}
