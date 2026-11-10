"""Training and evaluation of one run (paper: Methods, "Training and evaluation").

A job is a dict with the fields
  task         "mnist_bin" or "shapes_cs" (eqnn.data)
  arch         model id: config6 (Equiv), config7 (NonEquiv), config10 (NonEquiv-Twirled), config6n (Equiv-NoSym)
  layers       number of layers L (6 L angles)
  N            number of training images
  seed         fixes the training images, the random poses, the initial angles and the order of the mini-batches
  shortcut     "pose_rand", "none" or a cue (eqnn.data)
and optionally
  noise_scale  train and evaluate under the realistic noise of eqnn.noise at this scale (chip, chip_spread: the
               per-qubit parameters, see eqnn.models.build_model)
  noise_eval   list of scales: after a noiseless training, evaluate the final angles under the noise at each scale.

Training: angles uniform in [-0.1, 0.1], Adam (learning rate 0.05, betas (0.5, 0.999)), EPOCHS epochs of ten
mini-batches of N/10 images, logistic loss of 20 t m. The result (one JSON line, see run_job) holds the final
accuracies and losses on the training set and on the test sets of eqnn.data.build_data, the learning curves (every
EVAL_EVERY epochs, on the training set and on CURVE_TEST test images) and the final angles.
"""

import json
import os
import time

import torch

from eqnn.data import build_data
from eqnn.models import build_model, init_params, model_id, model_loss, predict, seed_everything

EPOCHS = 70
EVAL_EVERY = 7
CURVE_TEST = 250
LEARNING_RATE = 0.05
BETAS = (0.5, 0.999)
BATCHES_PER_EPOCH = 10
# states per call of the circuit simulation (bounds the memory of the backward pass); the eight transformed copies of
# a twirled model count as eight states per image
STATE_BUDGET = int(os.environ.get("EQNN_STATE_BUDGET", "128"))
NOISE_COST = 10.0  # cost of a noisy run relative to the noiseless one (job scheduling only)


def job_key(job: dict) -> tuple:
    """Identity of a run (used to skip the runs already in a results file)."""
    return (job["task"], model_id(job["arch"]), job["layers"], job["N"], job["seed"], job["shortcut"],
            job.get("noise_scale"), job.get("chip", 0), job.get("chip_spread", True), bool(job.get("noise_eval")))


def estimated_cost(job: dict) -> float:
    factor = 8 if model_id(job["arch"]) == "config10" else 1
    if job.get("noise_scale") is not None:
        factor *= NOISE_COST
    return job["N"] * job["layers"] * factor + 20.0


def _write_progress(progress_dir, job, epoch, t0):
    if not progress_dir:
        return
    try:
        os.makedirs(progress_dir, exist_ok=True)
        path = os.path.join(progress_dir, f"{os.getpid()}.json")
        with open(path + ".tmp", "w") as f:
            json.dump({**job, "epochs_done": epoch, "epochs": EPOCHS, "elapsed": round(time.time() - t0, 1),
                       "updated": time.time()}, f)
        os.replace(path + ".tmp", path)
    except OSError:
        pass


@torch.no_grad()
def evaluate(model, params, x, y, chunk=500):
    """(accuracy, loss) of the model on the images x with labels y."""
    ms = torch.cat([model(x[i:i + chunk], params) for i in range(0, x.shape[0], chunk)])
    return (predict(ms) == y).double().mean().item(), model_loss(ms, y).item()


def run_job(job: dict, progress_dir: str | None = None) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    seed_everything(job["seed"])
    N, L = job["N"], job["layers"]
    x_tr, y_tr, tests, y_te = build_data(job["task"], N, job["seed"], job["shortcut"])
    model, n_params = build_model(job["arch"], L, noise_scale=job.get("noise_scale"),
                                  chip_seed=job.get("chip", 0), chip_spread=job.get("chip_spread", True))
    params = init_params(n_params, job["seed"]).clone().requires_grad_()
    opt = torch.optim.Adam([params], lr=LEARNING_RATE, betas=BETAS)
    batch = max(1, N // BATCHES_PER_EPOCH)
    micro = max(1, STATE_BUDGET // (8 if model_id(job["arch"]) == "config10" else 1))
    gen = torch.Generator().manual_seed(job["seed"])  # order of the mini-batches
    sub = {name: (x[:CURVE_TEST], y_te[:CURVE_TEST]) for name, x in tests.items()}
    curve = {"epoch": [], "train_acc": [], "train_loss": [], "clean_acc": [], "clean_loss": [],
             "cued_acc": [], "cued_t_acc": []}

    def record(epoch):
        ta, tl = evaluate(model, params.detach(), x_tr, y_tr)
        curve["epoch"].append(epoch)
        curve["train_acc"].append(ta)
        curve["train_loss"].append(tl)
        for name in ("clean", "cued", "cued_t"):
            a, loss = evaluate(model, params.detach(), *sub[name])
            curve[f"{name}_acc"].append(a)
            if name == "clean":
                curve["clean_loss"].append(loss)

    record(0)
    for epoch in range(EPOCHS):
        _write_progress(progress_dir, job, epoch, t0)
        perm = torch.randperm(N, generator=gen)
        for s in range(0, N, batch):
            idx = perm[s:s + batch]
            xb, yb = x_tr[idx], y_tr[idx]
            opt.zero_grad()
            for m0 in range(0, xb.shape[0], micro):  # gradient accumulation over micro-batches
                xs, ys = xb[m0:m0 + micro], yb[m0:m0 + micro]
                loss = model_loss(model(xs, params), ys) * (ys.shape[0] / xb.shape[0])
                loss.backward()
            opt.step()
        if (epoch + 1) % EVAL_EVERY == 0:
            record(epoch + 1)
    _write_progress(progress_dir, job, EPOCHS, t0)

    final = {}
    p = params.detach()
    final["train_acc"], final["train_loss"] = evaluate(model, p, x_tr, y_tr)
    for name, x in tests.items():
        final[f"{name}_acc"], final[f"{name}_loss"] = evaluate(model, p, x, y_te)
    if job.get("noise_eval"):  # the trained angles evaluated under realistic noise (noise in the test only)
        final["noise_eval"] = {}
        for sc in job["noise_eval"]:
            noisy, _ = build_model(job["arch"], L, noise_scale=sc, chip_seed=job.get("chip", 0),
                                   chip_spread=job.get("chip_spread", True))
            final["noise_eval"][f"{sc:g}"] = {"train_acc": evaluate(noisy, p, x_tr, y_tr)[0],
                                              "clean_acc": evaluate(noisy, p, tests["clean"], y_te)[0]}
    return {**job, **final, "initial_clean_acc": curve["clean_acc"][0], "curve": curve, "params": p.tolist(),
            "n_params": n_params, "seconds": round(time.time() - t0, 1)}
