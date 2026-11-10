#!/usr/bin/env bash
# Every training campaign of the paper, with the exact grids of the runs in results/.
# Each command appends to <out>/runs.jsonl and skips the runs already there: delete (or move) a results file to
# recompute it from scratch. Run from the root of the repository. Approximate cost on 32 CPU threads in brackets.
#
#   bash experiments/run_all.sh            # everything (about 6 hours)
#   bash experiments/run_all.sh mnist      # one block: mnist, upright, synthetic, galaxy, erythrocytes, cues, noise, pilots, dla
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
RUN="$PY -m eqnn.runner"
MODELS="Equiv NonEquiv NonEquiv-Twirled Equiv-NoSym"
ALL=${1:-all}

if [[ $ALL == all || $ALL == mnist ]]; then  # MNIST 4 vs 5 in random pose (main results, grid over N and L) [~40 min]
  $RUN --out results/mnist_pose --tasks mnist_bin --shortcuts pose_rand --models $MODELS \
       --layers 1 2 4 8 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/mnist_pose --tasks mnist_bin --shortcuts pose_rand --models $MODELS \
       --layers 2 4 --n-values 40 --seeds 6 7 8 9 10 11 12 13 14 15
fi
if [[ $ALL == all || $ALL == upright ]]; then  # control: upright MNIST images (grid over N and L) [~5 min]
  $RUN --out results/mnist_upright --tasks mnist_bin --shortcuts none --models Equiv NonEquiv \
       --layers 2 4 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/mnist_upright --tasks mnist_bin --shortcuts none --models Equiv NonEquiv \
       --layers 2 4 --n-values 40 --seeds 6 7 8 9 10 11 12 13 14 15
fi
if [[ $ALL == all || $ALL == synthetic ]]; then  # synthetic task (main results, grid over N and L) [~15 min]
  $RUN --out results/synthetic --tasks shapes_cs --shortcuts none --models $MODELS \
       --layers 1 2 4 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/synthetic --tasks shapes_cs --shortcuts none --models $MODELS \
       --layers 2 4 --n-values 20 --seeds 6 7 8 9 10 11 12 13 14 15
fi
if [[ $ALL == all || $ALL == galaxy ]]; then  # Galaxy10 (SDSS) in random pose: pilot over five class pairs, grid, cues, noise [~3 hours]
  # set EQNN_DATA_ROOT to the folder that contains galaxy10/Galaxy10.h5
  $RUN --out results/galaxy_pilots --tasks galaxy_1_4 galaxy_0_4 galaxy_1_2 galaxy_1_0 galaxy_2_4 --shortcuts pose_rand \
       --models Equiv NonEquiv --layers 2 4 --n-values 40 --seeds 1 2 3
  $RUN --out results/galaxy --tasks galaxy_0_4 --shortcuts pose_rand --models $MODELS \
       --layers 1 2 4 8 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/galaxy --tasks galaxy_0_4 --shortcuts pose_rand --models $MODELS \
       --layers 2 4 --n-values 40 --seeds 6 7 8 9 10 11 12 13 14 15
  $RUN --out results/galaxy --tasks galaxy_0_4 --shortcuts mark_e_rand gradient_rand invariant_rand \
       --models $MODELS --layers 2 4 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/noise_train_test --tasks galaxy_0_4 --shortcuts pose_rand --models $MODELS \
       --layers 4 --n-values 40 --seeds 1 2 3 4 5 --noise-scale 0 0.25 0.5 1 2 4 --chip 0
  $RUN --out results/noise_test_only --tasks galaxy_0_4 --shortcuts pose_rand --models $MODELS \
       --layers 4 --n-values 40 --seeds $(seq 1 15) --noise-eval 0 0.25 0.5 1 2 4 --chip 0
fi
if [[ $ALL == all || $ALL == erythrocytes ]]; then  # red blood cells (erythrocytesIDB) in random pose: pilot, grid, cues, noise [~2 hours]
  # set EQNN_DATA_ROOT to the folder that contains "erythrocytes/Version 2, erythrocytesIDB" (unzipped from Zenodo)
  $RUN --out results/erythrocytes_pilots --tasks ery_0_1 ery_0_2 ery_1_2 --shortcuts pose_rand \
       --models Equiv NonEquiv --layers 2 4 --n-values 40 --seeds 1 2 3
  $RUN --out results/erythrocytes --tasks ery_1_2 --shortcuts pose_rand --models $MODELS \
       --layers 2 4 --n-values 40 --seeds $(seq 1 15)
  $RUN --out results/erythrocytes --tasks ery_1_2 --shortcuts pose_rand --models $MODELS \
       --layers 1 2 4 8 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/erythrocytes --tasks ery_1_2 --shortcuts mark_e_rand gradient_rand invariant_rand \
       --models $MODELS --layers 2 4 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/noise_test_only --tasks ery_1_2 --shortcuts pose_rand --models $MODELS \
       --layers 4 --n-values 40 --seeds $(seq 1 15) --noise-eval 0 0.25 0.5 1 2 4 --chip 0
  $RUN --out results/noise_train_test --tasks ery_1_2 --shortcuts pose_rand --models $MODELS \
       --layers 4 --n-values 40 --seeds 1 2 3 4 5 --noise-scale 0 0.25 0.5 1 2 4 --chip 0
fi
if [[ $ALL == all || $ALL == cues ]]; then  # shortcuts [~40 min]
  $RUN --out results/mnist_pose --tasks mnist_bin --shortcuts mark_e_rand gradient_rand invariant_rand \
       --models $MODELS --layers 2 4 --n-values 10 20 40 80 --seeds 1 2 3 4 5
  $RUN --out results/synthetic --tasks shapes_cs --shortcuts mark_e gradient invariant \
       --models $MODELS --layers 2 4 --n-values 10 20 40 80 --seeds 1 2 3 4 5
fi
if [[ $ALL == all || $ALL == noise ]]; then  # hardware noise (figures of the noise) [~3 hours]
  # noise in training and test: trained and evaluated under the noise of scale s
  $RUN --out results/noise_train_test --tasks mnist_bin --shortcuts pose_rand --models $MODELS \
       --layers 4 --n-values 40 --seeds 1 2 3 4 5 --noise-scale 0 0.25 0.5 1 2 4 --chip 0
  $RUN --out results/noise_train_test --tasks shapes_cs --shortcuts none --models $MODELS \
       --layers 4 --n-values 20 --seeds 1 2 3 4 5 --noise-scale 0 0.25 0.5 1 2 4 --chip 0
  # noise in the test only: trained without noise, the final angles evaluated under noise
  $RUN --out results/noise_test_only --tasks mnist_bin --shortcuts pose_rand --models $MODELS \
       --layers 4 --n-values 40 --seeds $(seq 1 15) --noise-eval 0 0.25 0.5 1 2 4 --chip 0
  $RUN --out results/noise_test_only --tasks shapes_cs --shortcuts none --models $MODELS \
       --layers 4 --n-values 20 --seeds $(seq 1 15) --noise-eval 0 0.25 0.5 1 2 4 --chip 0
fi
if [[ $ALL == all || $ALL == pilots ]]; then  # design of the synthetic task (appendix) [~30 min]
  $PY experiments/pilots.py v0
  $PY experiments/pilots.py variants
fi
if [[ $ALL == all || $ALL == gradients ]]; then  # variance of the gradients at the initialization of training [~5 min]
  $PY experiments/gradient_variance.py results/gradient_variance.json
  $PY experiments/gradient_histograms.py results/gradient_histograms.json
fi
if [[ $ALL == all || $ALL == dla ]]; then  # dynamical Lie algebras and function classes [seconds]
  $PY -m eqnn.dla results/dla.json
fi
