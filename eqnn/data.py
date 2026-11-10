"""The four tasks of the paper, the action of D4 on the images and the shortcut cues.

Tasks (16 x 16 images, l2-normalized, i.e. amplitude-encoded states of 8 qubits)
  mnist_bin   MNIST digits 4 (class 0) and 5 (class 1), resized to 16 x 16
  shapes_cs   synthetic task: a filled square (class 0) or a thin plus-shaped cross (class 1) on a noisy background
  galaxy_<a>_<b>  Galaxy10 (SDSS): galaxy images, morphological classes a vs b (GALAXY_CLASSES); the task of the paper
                  is galaxy_0_4 (face-on disk without spiral arms vs edge-on disk with a rounded bulge), chosen among the
                  candidates below in pilot runs (experiments/run_all.sh)
  ery_<a>_<b>     erythrocytesIDB1: red blood cells of patients with sickle cell disease, classes a vs b
                  (ERY_CLASSES); the task of the paper is ery_1_2 (elongated vs other deformed cells), chosen among the
                  candidates below in pilot runs (experiments/run_all.sh)
The Galaxy10 images are converted to grayscale (luminance), resized to 16 x 16 with antialiasing and l2-normalized; the
raw file is EQNN_DATA_ROOT/galaxy10/Galaxy10.h5 (default EQNN_DATA_ROOT: data/). The red blood cells (dark on a bright
background) are converted to the inverted luminance minus the level of the background, so that the amplitude is on the
cell; the raw images are in EQNN_DATA_ROOT/erythrocytes/Version 2, erythrocytesIDB/erythrocytesIDB1/individual cells/.

Shortcuts (the argument `shortcut` of build_data)
  pose_rand       MNIST in random pose: every image (training and test) is moved by an element of D4 drawn uniformly
                  at random and independently of the label, so that the label is D4-invariant by construction
  none            no transformation (synthetic task; MNIST upright images, the control of the paper)
  mark_e          corner watermark: a 2 x 2 patch at a position of the frame that depends on the class, with an
                  amplitude that gives it a fixed share (MARK_E_SHARE) of the squared norm of every image
  gradient        brightness gradient: the image is multiplied by a ramp along a direction that depends on the class
  invariant       control cue, invariant under D4: a patch at the center for class 0, in the four corners for class 1
  <cue>_rand      (MNIST) the cue is added after the random pose of the content: mark_e_rand, gradient_rand,
                  invariant_rand
The cue of class c is the reference cue moved by the c-th element of D4 (class 0: the reference, class 1: its rotation
by 90 degrees), so the cues of the two classes lie in one orbit and an exactly invariant model cannot read them.
Cues are added to the TRAINING images; the test images are evaluated with and without the cue (see build_data).
"""

import math
import os

import torch

N_TEST = 1000  # balanced test set
N_POOL = 50000  # training pool: the first 50000 images of the training split
TEST_SEED = 12345
DATA_ROOT = os.environ.get("EQNN_DATA_ROOT", "data")
GALAXY_CLASSES = ("disk, face-on, no spiral", "smooth, completely round", "smooth, in-between round",
                  "smooth, cigar-shaped", "disk, edge-on, rounded bulge", "disk, edge-on, boxy bulge",
                  "disk, edge-on, no bulge", "disk, face-on, tight spiral", "disk, face-on, medium spiral",
                  "disk, face-on, loose spiral")
TASKS = {
    "mnist_bin": ("mnist", (4, 5)),
    "shapes_cs": ("shapes_cs", (0, 1)),
}
# candidate binary tasks of Galaxy10 (pilot runs); galaxy_0_4 is the task of the paper
for _a, _b in ((1, 4), (0, 4), (1, 2), (1, 0), (2, 4)):
    TASKS[f"galaxy_{_a}_{_b}"] = ("galaxy10", (_a, _b))
ERY_CLASSES = ("circular", "elongated", "other")
# candidate binary tasks of erythrocytesIDB (pilot runs); ery_1_2 is the task of the paper
for _a, _b in ((0, 1), (0, 2), (1, 2)):
    TASKS[f"ery_{_a}_{_b}"] = ("erythrocytes", (_a, _b))
SPLIT_SEED = 0  # Galaxy10 and erythrocytesIDB have no official split: a fixed random 80/20 (50/50) split
SHORTCUTS = ("none", "pose_rand", "mark_e", "gradient", "invariant", "mark_e_rand", "gradient_rand", "invariant_rand")
CACHE_DIR = os.environ.get("EQNN_DATA_CACHE", "data/cache")  # the preprocessed splits (MNIST is downloaded in data/)
SIDE = 16
PATCH = SIDE // 8  # side of the watermark
MARK_E_SHARE = 0.2  # share of the squared norm carried by the watermark, the same in every image

# generator of the synthetic task (the variant chosen in the pilot runs, see experiments/pilots.py)
SHAPES = {"square_sides": (8, 14), "cross_spans": (9, 15), "cross_thick": (1, 1), "hollow": False, "noise": 0.05}


def n_classes(task: str) -> int:
    return len(TASKS[task][1])


# ---------------------------------------------------------------- raw data
def make_shapes(n: int, seed: int, square_sides, cross_spans, cross_thick, hollow, noise):
    """n images (half per class, shuffled) of one shape on a noisy background.
    Class 0: square of side in square_sides (inclusive range; an outline two pixels thick if hollow);
    class 1: plus-shaped cross of total span in cross_spans and arm thickness in cross_thick (span - thickness even,
    so that the cross is symmetric). Both shapes are D4-symmetric about their own center and the position is uniform
    over all the placements inside the frame (a D4-invariant set), so the distribution of each class is D4-invariant
    and the label is a function of the orbit. Shape pixels have a random intensity in [0.5, 1], the background
    |N(0, noise^2)|."""
    g = torch.Generator().manual_seed(seed)
    labels = torch.arange(n) % 2
    labels = labels[torch.randperm(n, generator=g)]
    imgs = (torch.randn(n, SIDE, SIDE, generator=g, dtype=torch.float64) * noise).abs()
    rint = lambda lo, hi: int(torch.randint(lo, hi + 1, (1,), generator=g))  # noqa: E731
    for i in range(n):
        v = 0.5 + 0.5 * torch.rand(1, generator=g, dtype=torch.float64).item()
        if int(labels[i]) == 0:
            side = rint(*square_sides)
            r, c = rint(0, SIDE - side), rint(0, SIDE - side)
            imgs[i, r:r + side, c:c + side] = v
            if hollow:
                imgs[i, r + 2:r + side - 2, c + 2:c + side - 2] = (torch.randn(side - 4, side - 4, generator=g,
                                                                              dtype=torch.float64) * noise).abs()
        else:
            t = rint(*cross_thick)
            span = rint(*cross_spans)
            span += (span - t) % 2
            span = min(span, SIDE - (SIDE - t) % 2)
            r, c = rint(0, SIDE - span), rint(0, SIDE - span)
            o = (span - t) // 2
            imgs[i, r:r + span, c + o:c + o + t] = v
            imgs[i, r + o:r + o + t, c:c + span] = v
    imgs = imgs.reshape(n, -1)
    return imgs / imgs.norm(dim=1, keepdim=True), labels


def _to_tensor_16(rgb) -> torch.Tensor:
    """(n, h, w, 3) uint8 RGB images -> (n, 256) float64: luminance, 16 x 16 with antialiasing, l2-normalized."""
    import numpy as np
    x = torch.as_tensor(np.asarray(rgb, dtype=np.float64) / 255.0)
    gray = (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])[:, None]
    small = torch.nn.functional.interpolate(gray, size=(SIDE, SIDE), mode="bilinear", antialias=True,
                                            align_corners=False)
    imgs = small.reshape(len(small), -1)
    return imgs / imgs.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _load_galaxy10(train: bool):
    import h5py
    with h5py.File(os.path.join(DATA_ROOT, "galaxy10", "Galaxy10.h5"), "r") as f:
        imgs, labels = f["images"][:], f["ans"][:]
    x, y = _to_tensor_16(imgs), torch.as_tensor(labels, dtype=torch.long)
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(SPLIT_SEED))
    cut = int(0.8 * len(y))
    idx = perm[:cut] if train else perm[cut:]
    return x[idx], y[idx]


def _cells_16(rgb) -> torch.Tensor:
    """Bright-field cells (dark on a bright background) -> (n, 256): inverted luminance minus the background level
    (median of the border pixels), clipped at zero, 16 x 16 with antialiasing, l2-normalized."""
    import numpy as np
    x = torch.as_tensor(np.asarray(rgb, dtype=np.float64) / 255.0)
    inv = 1.0 - (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
    border = torch.cat([inv[:, 0], inv[:, -1], inv[:, :, 0], inv[:, :, -1]], dim=1)
    inv = (inv - border.median(dim=1).values[:, None, None]).clamp_min(0.0)[:, None]
    small = torch.nn.functional.interpolate(inv, size=(SIDE, SIDE), mode="bilinear", antialias=True,
                                            align_corners=False)
    imgs = small.reshape(len(small), -1)
    return imgs / imgs.norm(dim=1, keepdim=True).clamp_min(1e-12)


def _load_erythrocytes(train: bool):
    """The 80 x 80 images of individual cells of erythrocytesIDB1; a fixed random 50/50 split of every class (the data
    set is small)."""
    import numpy as np
    from PIL import Image
    folder = os.path.join(DATA_ROOT, "erythrocytes", "Version 2, erythrocytesIDB", "erythrocytesIDB1", "individual cells")
    xs, ys = [], []
    for c, cname in enumerate(ERY_CLASSES):
        files = sorted(f for f in os.listdir(os.path.join(folder, cname)) if f.lower().endswith(".jpg"))
        imgs = np.stack([np.asarray(Image.open(os.path.join(folder, cname, f)).convert("RGB").resize((80, 80)))
                         for f in files])
        x = _cells_16(imgs)
        perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(SPLIT_SEED + c))
        idx = perm[:len(x) // 2] if train else perm[len(x) // 2:]
        xs.append(x[idx])
        ys.append(torch.full((len(idx),), c))
    return torch.cat(xs), torch.cat(ys)


def _load_split(name: str, train: bool):
    if name == "shapes_cs":
        return make_shapes(N_POOL + 10000 if train else 10000, 77 if train else 78, **SHAPES)
    if name == "galaxy10":
        return _load_galaxy10(train)
    if name == "erythrocytes":
        return _load_erythrocytes(train)
    import torchvision
    from torchvision import transforms

    ds = torchvision.datasets.MNIST(root="data", train=train, download=True)
    tf = transforms.Compose([transforms.Resize(SIDE), transforms.ToTensor()])
    imgs = torch.stack([tf(im).squeeze(0) for im, _ in ds]).reshape(len(ds), -1).to(torch.float64)
    imgs = imgs / imgs.norm(dim=1, keepdim=True).clamp_min(1e-12)  # l2 normalization
    return imgs, torch.as_tensor(ds.targets)


def load_split(name: str, train: bool):
    """(images (n, 256), labels (n,)) of a whole split, cached on disk."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = f"{CACHE_DIR}/{name}_{'train' if train else 'test'}_{SIDE}.pt"
    if not os.path.exists(path):
        x, y = _load_split(name, train)
        torch.save((x, y), path + ".tmp")
        os.replace(path + ".tmp", path)
    return torch.load(path)


def _select(labels, classes, n_per_class, generator):
    idx = []
    for c in classes:
        pool = (labels == c).nonzero(as_tuple=True)[0]
        idx.append(pool[torch.randperm(len(pool), generator=generator)[:n_per_class]])
    idx = torch.cat(idx)
    return idx[torch.randperm(len(idx), generator=generator)]


def load_task(task: str, N: int, seed: int):
    """(x_train, y_train, x_test, y_test): N balanced training images drawn with `seed` from the training pool and a
    fixed balanced test set of N_TEST images from the test split (fewer if a class has fewer than N_TEST / 2 test
    images). Labels are 0, 1 (the order of the class tuple)."""
    name, classes = TASKS[task]
    k = len(classes)
    x_all, y_all = load_split(name, True)
    x_all, y_all = x_all[:N_POOL], y_all[:N_POOL]
    xt_all, yt_all = load_split(name, False)
    tr = _select(y_all, classes, N // k, torch.Generator().manual_seed(seed))
    per_class = min([N_TEST // k] + [int((yt_all == c).sum()) for c in classes])
    te = _select(yt_all, classes, per_class, torch.Generator().manual_seed(TEST_SEED))
    remap = {c: i for i, c in enumerate(classes)}
    ytr = torch.tensor([remap[int(c)] for c in y_all[tr]])
    yte = torch.tensor([remap[int(c)] for c in yt_all[te]])
    return x_all[tr], ytr, xt_all[te], yte


# ---------------------------------------------------------------- D4
def apply_d4(img: torch.Tensor, g: int) -> torch.Tensor:
    """Element g (0..7) of D4 on a (..., side, side) image: rotation by g mod 4 quarter turns, then a reflection if
    g >= 4."""
    out = torch.rot90(img, g % 4, dims=(-2, -1))
    return torch.flip(out, dims=[-1]) if g >= 4 else out


def random_d4(states: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Every image moved by an element of D4 drawn uniformly at random."""
    imgs = states.reshape(-1, SIDE, SIDE)
    gs = torch.randint(8, (imgs.shape[0],), generator=generator)
    out = torch.stack([apply_d4(imgs[i], int(gs[i])) for i in range(imgs.shape[0])])
    return out.reshape(states.shape[0], -1)


# ---------------------------------------------------------------- cues
def _renorm(imgs: torch.Tensor) -> torch.Tensor:
    imgs = imgs / imgs.flatten(1).norm(dim=1)[:, None, None]
    return imgs.reshape(imgs.shape[0], -1)


def _patch_mask(g: int) -> torch.Tensor:
    base = torch.zeros(SIDE, SIDE, dtype=torch.bool)
    base[:PATCH, PATCH:2 * PATCH] = True  # rows 0-1, columns 2-3: not on a symmetry axis
    return apply_d4(base, g)


def _ramp(g: int) -> torch.Tensor:
    """Reference ramp, geometric from 0.316 to 3.16 (a factor 10) along the direction (1, 2) of (row, column) (not
    along a symmetry axis, so that its D4 orbit has eight distinct elements), moved by element g."""
    i, j = torch.meshgrid(torch.arange(SIDE, dtype=torch.float64), torch.arange(SIDE, dtype=torch.float64),
                          indexing="ij")
    t = (i + 2 * j) / (3 * (SIDE - 1))
    return apply_d4(torch.exp(math.log(10.0) * (t - 0.5)), g)


def add_cue(states: torch.Tensor, cue: torch.Tensor, shortcut: str) -> torch.Tensor:
    """Adds the cue of class `cue[i]` to image i and renormalizes."""
    if shortcut == "none":
        return states
    imgs = states.reshape(-1, SIDE, SIDE).clone()
    if shortcut == "gradient":
        ramps = torch.stack([_ramp(g) for g in range(8)])
        return _renorm(imgs * ramps[cue.long() % 8])
    if shortcut == "mark_e":
        masks = torch.stack([_patch_mask(g) for g in range(8)])
        for i in range(imgs.shape[0]):
            m = masks[int(cue[i]) % 8]
            rest = (imgs[i] ** 2).sum() - (imgs[i][m] ** 2).sum()
            imgs[i][m] = torch.sqrt(MARK_E_SHARE * rest / (m.sum() * (1 - MARK_E_SHARE)))
        return _renorm(imgs)
    if shortcut == "invariant":  # amplitude twice the maximum of the image
        value = 2.0 * imgs.abs().flatten(1).max(dim=1).values
        lo = (SIDE - PATCH) // 2
        for i in range(imgs.shape[0]):
            if int(cue[i]) == 0:
                imgs[i, lo:lo + PATCH, lo:lo + PATCH] = value[i]
            else:
                for r in (slice(0, PATCH), slice(SIDE - PATCH, SIDE)):
                    for c in (slice(0, PATCH), slice(SIDE - PATCH, SIDE)):
                        imgs[i, r, c] = value[i]
        return _renorm(imgs)
    raise ValueError(shortcut)


def build_data(task: str, N: int, seed: int, shortcut: str):
    """Training set (with the cue) and the test sets built from the same N_TEST test images:
      clean    without the cue (the true rule)
      cued     with the cue of the true class, as in training
      cued_t   the cued images, each moved by a random element of D4 (the cue moves with the image)
      clean_t  the clean images, each moved by a random element of D4 (not used in the paper; it is drawn before
               cued_t from the same generator, so it is kept for the reproducibility of cued_t)."""
    if shortcut not in SHORTCUTS:
        raise ValueError(shortcut)
    xtr, ytr, xte, yte = load_task(task, N, seed)
    cue = shortcut
    if shortcut.endswith("_rand"):  # random pose of the content, independent of the label (MNIST)
        xtr = random_d4(xtr, torch.Generator().manual_seed(3000 + seed))
        xte = random_d4(xte, torch.Generator().manual_seed(TEST_SEED + 7))
        cue = "none" if shortcut == "pose_rand" else shortcut[:-5]  # then the cue (none for pose_rand)
    x_train = add_cue(xtr, ytr, cue)
    g = torch.Generator().manual_seed(1000 + seed)
    cued = add_cue(xte, yte, cue)
    tests = {
        "clean": xte,
        "clean_t": random_d4(xte, g),
        "cued": cued,
        "cued_t": random_d4(cued, g),
    }
    return x_train, ytr, tests, yte
