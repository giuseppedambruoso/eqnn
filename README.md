# Invariance can prevent overfitting and shortcut learning in quantum neural networks

Code, results and LaTeX sources of the paper

> G. D'Ambruoso, G. Gramegna, D. Pomarico, G. Magnifico, C. Lupo,
> *Invariance can prevent overfitting and shortcut learning in quantum neural networks* (preprint).

This repository is self-contained. It holds the code used to train every model of the paper, the results of every
run, and the scripts that turn the results into the figures and tables. Every figure and table can be rebuilt in
a few minutes from the stored results. Every result can be recomputed from scratch with one script.

## The study in one paragraph

Images of $16\times16$ pixels are amplitude-encoded on 8 qubits (row index on qubits 0–3, column index on qubits
4–7), so that the dihedral group $D_4$ of the square acts as $X$ layers on one register (reflections) and as the
swap of the two registers (transposition). Four models are compared. They have the same gates and the same six
trainable angles per layer, and they share the same dynamical Lie algebra (dimension 569):

| id in the results | name in the paper  | circuit                         | measured observable               | invariant under $D_4$ |
|-------------------|--------------------|---------------------------------|-----------------------------------|-----------------------|
| `config6`         | Equiv              | equivariant                     | $O=(X_0+X_4)/2$                   | yes                   |
| `config7`         | NonEquiv           | non-equivariant (column axes cycled $X\to Y\to Z$) | $O$            | no                    |
| `config10`        | NonEquiv-Twirled   | non-equivariant, averaged over the 8 transformed inputs | $O$       | yes                   |
| `config6n`        | Equiv-NoSym        | equivariant                     | $O_{ns}=(X_0+Z_0+Z_4)/(1+\sqrt2)$ | no                    |

They are compared in two ways:
- **Comparison 1** (Equiv, NonEquiv-Twirled, NonEquiv): how the invariance is obtained.
- **Comparison 2** (Equiv, Equiv-NoSym): the same circuit, with and without an invariant readout.

The tasks are MNIST digits 4 vs 5 in a random pose, a synthetic task (a thin cross against a filled square) and
face-on vs edge-on disk galaxies from Galaxy10 (SDSS) in a random pose, and elongated (sickle) vs other deformed red
blood cells from erythrocytesIDB in a random pose. All the labels are invariant under $D_4$. The tasks are studied with and without class-dependent cues in the training images
(shortcuts), and with and without the realistic noise of a superconducting processor.

## Repository layout

```
eqnn/                    the library
  data.py                  tasks, action of D4 on the images, cues
  circuits.py              gate lists of the equivariant and non-equivariant circuits
  models.py                the four models (exact statevector simulation), loss, prediction
  noise.py                 realistic single-qubit noise; exact Heisenberg-picture simulation (and default.mixed)
  training.py              training and evaluation of one run
  runner.py                parallel campaigns:  python -m eqnn.runner --help
  dla.py                   dynamical Lie algebras and function classes, exact (rational arithmetic)
experiments/
  run_all.sh               every campaign of the paper, with the exact grids of the stored runs
  pilots.py                pilot runs used to design the synthetic task
  reproduce_check.py       re-runs a sample of stored runs and compares them number by number
  gradient_variance.py     variance of the gradients at the initialization of training (results/gradient_variance.json)
  gradient_histograms.py   moduli of the gradient components in the main settings (results/gradient_histograms.json)
  dla_checks.py            independent numerical checks of the DLA and function-class dimensions (results/dla_checks.txt)
  dla_checks_rest.py       the same checks for the other models and depths (appended to results/dla_checks.txt)
  watermark_shift.py       why removing the corner watermark lowers the accuracy of the invariant models on Galaxy10:
                             output shift, class separation, AUC and recalibrated threshold (results/watermark_shift.json)
analysis/
  paper_figures.py         every data figure and table of the paper, from results/
  schematics.py            schematic figures (the four models, function classes, noise model, twirled model)
  circuit_diagrams.py      the circuit diagrams
  supplementary.py         the supplementary tables and figures (supplementary/)
  text_numbers.py          every number quoted in the text that is not read from a table (paper/numbers.txt)
  common.py                loading, paired statistics and the style of the figures
results/                 one JSON line per run (format below)
  mnist_pose/              MNIST in random pose, with and without cues
  mnist_upright/           MNIST, upright images (control)
  synthetic/               synthetic task, with and without cues
  galaxy/                  Galaxy10 in random pose, with and without cues
  galaxy_pilots/           pilot runs over five Galaxy10 class pairs
  erythrocytes/            red blood cells in random pose, with and without cues
  erythrocytes_pilots/     pilot runs over the three erythrocytesIDB class pairs
  noise_train_test/        trained and tested under realistic noise
  noise_test_only/         trained without noise, evaluated under noise
  synthetic_pilots/        pilot variants of the synthetic task
  dla.json                 dimensions of the DLAs and of the function classes
  gradient_variance.json   variance of the gradients of the four models at the initialization of training
  gradient_histograms.json moduli of the gradient components of the four models at initialization (main settings)
  watermark_shift.json     per run: accuracy with and without the corner watermark, output shift of each class,
                             class separation, AUC, accuracy with the recalibrated threshold (main settings)
supplementary/           every result that the paper summarizes but does not report in full (README.md: all the
                           tables in Markdown; tables/: LaTeX versions; figures/: the whole grid), from
                           analysis/supplementary.py
paper/                   the paper: paper.tex (single LaTeX file), references.bib, figures/ (generated),
                           tables/ (generated; their contents are pasted in paper.tex between
                           % BEGIN/END tab_*.tex markers), numbers.txt (generated)
tests/                   checks of the statements of the paper (invariance, twirl, noise simulation, DLA)
```

## Installation

Python 3.12 and the packages of `requirements.txt` (the versions used for the paper):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest            # 16 tests, a few seconds
```

MNIST is downloaded by torchvision into `data/` on first use. Galaxy10 (SDSS) is the file `Galaxy10.h5` of the astroNN
package (https://astronn.readthedocs.io/en/latest/galaxy10sdss.html), to be placed in `$EQNN_DATA_ROOT/galaxy10/`. erythrocytesIDB is
the Zenodo record https://doi.org/10.5281/zenodo.18299473 (CC BY-NC-ND 4.0, not redistributed here), to be unzipped in
`$EQNN_DATA_ROOT/erythrocytes/`. The preprocessed splits are cached in `data/cache/`.

## Rebuilding the figures, the tables and the paper

```bash
make figures    # analysis/*.py: every figure into paper/figures and every table into paper/tables (about one minute)
make supplementary  # analysis/supplementary.py: every setting not reported in the paper, into supplementary/
make numbers    # analysis/text_numbers.py: the numbers quoted in the text, into paper/numbers.txt
make paper      # pdflatex + bibtex: paper/paper.pdf
make arxiv      # paper/arxiv.tar.gz, the sources to upload to arXiv
```

| Paper                                       | File(s)                                        | Produced by                     | Data                                 |
|---------------------------------------------|------------------------------------------------|---------------------------------|--------------------------------------|
| Fig. 1a (encoding) | `figures/encoding_p4m_diagram.pdf` | `analysis/schematics.py` | — |
| Fig. 1b (the four models) | `figures/fig_design.pdf` | `analysis/schematics.py` | — |
| Fig. 2 (circuits) | `figures/circuits.pdf` | `analysis/circuit_diagrams.py` | `eqnn/circuits.py` |
| Fig. 3 (twirled model) | `figures/fig_twirled.pdf` | `analysis/schematics.py` | an MNIST image |
| Fig. 4 (function classes) | `figures/fig_function_classes.pdf` | `analysis/schematics.py` | `results/dla.json` |
| Fig. 5 (tasks and cues) | `figures/fig_tasks.pdf` | `analysis/paper_figures.py` | images of the four tasks |
| Fig. 6 (main results: accuracies, gaps, per-seed differences) | `figures/fig_main.pdf` | `analysis/paper_figures.py` | `mnist_pose`, `synthetic`, `galaxy`, `erythrocytes` |
| Table I (effect of the invariance) | `tables/tab_factorial.tex` | `analysis/paper_figures.py` | `mnist_pose`, `synthetic`, `galaxy`, `erythrocytes` |
| Fig. 7 (shortcuts) | `figures/fig_cues.pdf` | `analysis/paper_figures.py` | `mnist_pose`, `synthetic`, `galaxy`, `erythrocytes` (cues) |
| Sec. V C (watermark removed on Galaxy10: output shift and recalibrated threshold) | `results/watermark_shift.json` | `experiments/watermark_shift.py` | `galaxy`, `mnist_pose`, `synthetic` (corner watermark) |
| Fig. 8 (hardware noise) | `figures/fig_noise.pdf` | `analysis/paper_figures.py` | `noise_train_test`, `noise_test_only` |
| Table II (the four tasks) | in `paper.tex` | static | — |
| Table III (DLA and function classes) | `tables/tab_dla.tex` | `analysis/paper_figures.py` | `results/dla.json` |
| Fig. 9 (gradients: variance and histograms) | `figures/fig_gradients.pdf` | `analysis/paper_figures.py` | `results/gradient_variance.json`, `results/gradient_histograms.json` |
| Fig. 10 (learning curves) | `figures/fig_curves.pdf` | `analysis/paper_figures.py` | `mnist_pose`, `synthetic`, `galaxy`, `erythrocytes` |
| Fig. 11 (Galaxy10 classes of the pilot runs) | `figures/fig_galaxy_classes.pdf` | `analysis/paper_figures.py` | Galaxy10 images |
| Table IV (paired comparisons) | `tables/tab_paired.tex` | `analysis/paper_figures.py` | `mnist_pose`, `synthetic`, `galaxy`, `erythrocytes` |
| Fig. 12 (dependence on N and L) | `figures/fig_scaling.pdf` | `analysis/paper_figures.py` | `mnist_pose`, `synthetic`, `galaxy`, `erythrocytes`, `mnist_upright` |
| Fig. 13 (noise model) | `figures/fig_noise_model.pdf` | `analysis/schematics.py` | — |
| Table V (noise parameters) | in `paper.tex` | values of `eqnn/noise.py` (`TYPICAL`, `SPREAD`) | — |
| Table VI (symmetry breaking by the noise) | `tables/tab_symbreak.tex` | `analysis/paper_figures.py` (simulation) | — |
| Numbers quoted in the text | `numbers.txt` | `analysis/text_numbers.py` | all of `results/` |
| Supplementary (main settings: training, test, gap) | `supplementary/tables/tab_main.tex` | `analysis/supplementary.py` | `mnist_pose`, `synthetic`, `galaxy`, `erythrocytes` |
| Supplementary (every L and N, L = 2, shortcuts at every N, noise tables, pilot runs, failures to train) | `supplementary/README.md`, `supplementary/tables/`, `supplementary/figures/` | `analysis/supplementary.py` | all of `results/` |

Every comparison between models in the figures and tables uses the seeds shared by all the compared series. The
comparisons are therefore paired: the same training images, poses, initial angles and mini-batch order for every
model. Paired differences are tested with the two-sided Wilcoxon signed-rank test over the seeds.

## Recomputing the results

```bash
bash experiments/run_all.sh            # every campaign (about 6 hours on 32 CPU threads)
bash experiments/run_all.sh noise      # one block: mnist, upright, synthetic, galaxy, cues, noise, pilots, dla
```

The runner appends to `results/<campaign>/runs.jsonl` and skips the runs already there. Move a results file away to
recompute it, or run `experiments/run_all.sh` on a fresh clone without `results/`. Single runs:

```bash
python -m eqnn.runner --out results/test --tasks mnist_bin --shortcuts pose_rand \
    --models Equiv NonEquiv --layers 4 --n-values 40 --seeds 1
```

Each run is deterministic: the seed fixes the training images, the random poses, the initial angles and the order
of the mini-batches. The simulations are exact and in double precision, with one thread per run. On the same
software versions a run reproduces the stored record exactly. `experiments/reproduce_check.py` re-runs 13 stored
runs (all models and tasks, cues, noise in training and in the test) and finds no difference in any number:
accuracies, losses, learning curves, final angles and evaluations under noise.

## Format of the results

One JSON object per line. Fields of the job: `task` (`mnist_bin`, `shapes_cs`, `galaxy_<a>_<b>`, `ery_<a>_<b>`), `arch` (model id, table above),
`layers`, `N`, `seed`, `shortcut`; for noisy runs `noise_scale`, `chip`, `chip_spread`. Outcomes:

- `train_acc`, `train_loss`: on the N training images (with the cue, if any).
- `clean_acc`, `clean_loss`: on the test images (1000; 630 for Galaxy10, 212 for the red blood cells) without a cue. This is the test accuracy of the paper.
- `cued_acc`: test images with the cue of the true class.
- `cued_t_acc`: the cued test images moved by a random element of $D_4$.
- `clean_t_acc`: the clean test images moved by a random element of $D_4$; not used in the paper.
- `curve`: accuracies every 7 epochs, on the training set and on the first 250 test images.
- `params`: the final angles; present in the more recent runs.
- `noise_eval` (`noise_test_only`): training and test accuracy of the final angles under noise, for each scale.
- `n_params`, `seconds`.

In MNIST, `shortcut = pose_rand` puts every image in a random pose. `mark_e_rand`, `gradient_rand` and
`invariant_rand` add the corner watermark, the brightness gradient and the invariant control cue after the random
pose (MNIST, Galaxy10, red blood cells). The synthetic task uses `none`, `mark_e`, `gradient` and `invariant`.

The results were produced during the development of the study with code that `eqnn/` reproduces exactly. That
code was checked to give identical images and model outputs, and identical records on the runs of
`reproduce_check.py`. The files of `results/` hold the runs used in the paper, selected from the development runs
(the other runs belonged to earlier variants of the study). They are otherwise unchanged, except for the field
`pilot` (the round) added to the pilot runs.

## License

Copyright (C) 2026 Giuseppe D'Ambruoso

- **Code** (`eqnn/`, `analysis/`, `experiments/`, `tests/`, `Makefile`): GNU General Public License, version 3 or (at
  your option) any later version. See [LICENSE](LICENSE).
- **Results, figures, tables and text** (`results/`, `supplementary/`, `paper/`, and this documentation): Creative
  Commons Attribution-ShareAlike 4.0 International. See [LICENSE-CC-BY-SA-4.0](LICENSE-CC-BY-SA-4.0).

Both licenses are copyleft: modified versions must be shared under the same terms. If you use this code, these
results or these figures in your work, please cite the paper (see `CITATION.cff`). The data sets used by the code
(MNIST, Galaxy10 SDSS, erythrocytesIDB) are not distributed here and keep their own licenses.
