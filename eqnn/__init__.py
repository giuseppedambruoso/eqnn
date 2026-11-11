"""eqnn: code of the paper "Invariance can prevent overfitting and shortcut learning in Quantum Neural Networks", arXiv:2611.14368.

Modules
  data       the two tasks (MNIST 4 vs 5 in random pose, synthetic cross vs square), the action of D4, the cues
  circuits   the equivariant and the non-equivariant circuit (gate lists)
  models     the four models (Equiv, NonEquiv, NonEquiv-Twirled, Equiv-NoSym), loss and prediction
  noise      the realistic single-qubit noise model and its exact Heisenberg-picture simulation
  training   training and evaluation of one run
  runner     parallel campaigns (python -m eqnn.runner)
  dla        dynamical Lie algebras and function classes, computed exactly (python -m eqnn.dla)
"""
