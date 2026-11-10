"""Dynamical Lie algebra (DLA) and function classes of the four models (paper: Methods, "Same circuit, different function
class", and the appendix "Dynamical Lie algebra and function space"),
computed exactly in the Pauli basis with rational arithmetic.

  DLA: the Lie closure of the generators of the gates of one layer (the same for any number of layers):
       Equiv, Equiv-NoSym: X_a + X_a' (pairs of R_X with a shared angle) and Y_a Y_b Y_a' Y_b' for the five blocks;
       NonEquiv, NonEquiv-Twirled: X_a + Y_a' and Y_a Y_b Z_a' Z_b' (the column axes cycled X -> Y -> Z -> X).
  Space of observables: the smallest space that contains the readout and is closed under the commutators with the
       generators, i.e. the span of U^dag O U over the circuits. The model is m(x) = x^T Re(U^dag O U) x, a linear
       combination of the quadratic features x^T M x with M in this space.
  Features: on real images x^T P x = 0 for the Pauli strings P with an odd number of Y (imaginary, antisymmetric),
       so the features of a model are spanned by the even part; their number is the size of the function class.
  NonEquiv-Twirled: the features of NonEquiv twirled over D4 (R(g)^dag P R(g) is a signed Pauli string).
  Invariant features: the features fixed by the twirl; shared features: the intersection of the feature spaces of
       two models (dim A + dim B - dim(A + B)).

    python -m eqnn.dla [results/dla.json]
"""

import itertools
import json
import os
import sys
from fractions import Fraction

N = 8


def mul(a, b):
    """P_a P_b = phase * P_c for Hermitian Pauli strings (x, z); returns (phase as power of i, c)."""
    (x1, z1), (x2, z2) = a, b
    # P = i^{|x&z|} X^x Z^z ; X^x1 Z^z1 X^x2 Z^z2 = (-1)^{|z1&x2|} X^{x1^x2} Z^{z1^z2}
    e = bin(x1 & z1).count("1") + bin(x2 & z2).count("1") + 2 * bin(z1 & x2).count("1")
    x, z = x1 ^ x2, z1 ^ z2
    e -= bin(x & z).count("1")
    return e % 4, (x, z)


def comm(A, B):
    """[iA, iB] for A, B real combinations of Hermitian Pauli strings, returned as C with [iA, iB] = i C."""
    out = {}
    for p, a in A.items():
        for q, b in B.items():
            e1, r = mul(p, q)
            e2, _ = mul(q, p)
            if e1 == e2:
                continue  # commute
            # [iP, iQ] = -(PQ - QP) = -2 PQ (anticommuting) = -2 i^{e1} R; must be i * (real) R -> i^{e1} = +-i
            assert e1 % 2 == 1
            coef = -2 * a * b * (1 if e1 == 1 else -1)  # -2 i^{e1} = i * coef -> coef = -2 * (i^{e1} / i)
            out[r] = out.get(r, 0) + coef
    return {k: v for k, v in out.items() if v != 0}


class Basis:
    def __init__(self):
        self.rows = {}  # pivot -> normalized row (dict), pivot coefficient 1

    def reduce(self, v):
        v = dict(v)
        changed = True
        while v:
            piv = max(v)
            if piv in self.rows:
                f = v[piv]
                for k, c in self.rows[piv].items():
                    v[k] = v.get(k, 0) - f * c
                    if v[k] == 0:
                        del v[k]
            else:
                return v, piv
        return None, None

    def add(self, v):
        r, piv = self.reduce(v)
        if r is None:
            return None
        f = r[piv]
        r = {k: Fraction(c) / f for k, c in r.items()}
        self.rows[piv] = r
        return r


def dla(gens, limit=70000):
    """Dimension of the Lie closure of the generators (anti-Hermitian i H, H real combinations of Pauli strings)."""
    B = Basis()
    elems = []
    for g in gens:
        r = B.add({k: Fraction(v) for k, v in g.items()})
        if r is not None:
            elems.append(r)
    i = 0
    while i < len(elems):
        for j in range(i):
            c = comm(elems[i], elems[j])
            if c:
                r = B.add(c)
                if r is not None:
                    elems.append(r)
                    if len(elems) > limit:
                        return len(elems)
        i += 1
    return len(elems)


def pauli(word):
    x = z = 0
    for w, c in word.items():
        b = 1 << w
        if c in "XY": x |= b
        if c in "YZ": z |= b
    return (x, z)


def P(word, coef=1):
    return {pauli(word): coef}


def add(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


blocks = [(0, 1), (2, 3), (1, 2), (3, 0), (0, 2)]
eq = [add(P({a: "X"}), P({a + 4: "X"})) for a in range(4)] + [P({a: "Y", b: "Y", a + 4: "Y", b + 4: "Y"}) for a, b in blocks]
neq = [add(P({a: "X"}), P({a + 4: "Y"})) for a in range(4)] + [P({a: "Y", b: "Y", a + 4: "Z", b + 4: "Z"}) for a, b in blocks]
def module_basis(gens, obs):
    B = Basis()
    elems = [B.add({k: Fraction(v) for k, v in obs.items()})]
    i = 0
    while i < len(elems):
        for g in gens:
            c = comm(g, elems[i])
            if c:
                r = B.add(c)
                if r is not None:
                    elems.append(r)
        i += 1
    return elems


def ny(p):
    x, z = p
    return bin(x & z).count("1")  # number of Y factors


def real_part(v):
    """x^T P x on real x vanishes for the Pauli strings with an odd number of Y (imaginary antisymmetric)."""
    return {k: c for k, c in v.items() if ny(k) % 2 == 0}


ROW = sum(1 << q for q in range(4))
COL = sum(1 << q for q in range(4, 8))


def swap(m):
    return ((m & ROW) << 4) | ((m & COL) >> 4)


def conj(p, flip_rows, flip_cols, sw):
    """R(g)^dag P R(g) for g = (t_x)^flip_rows (t_y)^flip_cols S^sw: X layers flip the sign of the Z and Y factors on
    their register; S exchanges the registers. Returns (sign, p')."""
    x, z = p
    sign = 1
    if flip_rows and bin(z & ROW).count("1") % 2:
        sign = -sign
    if flip_cols and bin(z & COL).count("1") % 2:
        sign = -sign
    if sw:
        x, z = swap(x), swap(z)
    return sign, (x, z)


def twirl(v):
    out = {}
    for fr in (0, 1):
        for fc in (0, 1):
            for sw in (0, 1):
                for k, c in v.items():
                    s, k2 = conj(k, fr, fc, sw)
                    out[k2] = out.get(k2, 0) + Fraction(s) * c / 8
    return {k: c for k, c in out.items() if c != 0}


def rank(vs):
    B = Basis()
    return sum(1 for v in vs if v and B.add(v) is not None)


def fixed_dim(vs):
    """Dimension of the D4-invariant part of span(vs): dim span(vs) - rank of (v - twirl(v))."""
    def diff(v):
        out = dict(v)
        for k, c in twirl(v).items():
            out[k] = out.get(k, 0) - c
        return {k: c for k, c in out.items() if c != 0}
    return rank(vs) - rank([diff(v) for v in vs])


def all_forms():
    """Dimension of the real quadratic forms on R^256 (Pauli strings with an even number of Y) and of their
    D4-invariant part (orbits of strings whose twirl does not vanish)."""
    total, invariant, seen = 0, 0, set()
    for x in range(1 << N):
        for z in range(1 << N):
            if ny((x, z)) % 2:
                continue
            total += 1
            if (x, z) not in seen:
                seen |= {conj((x, z), fr, fc, sw)[1] for fr in (0, 1) for fc in (0, 1) for sw in (0, 1)}
                invariant += bool(twirl({(x, z): Fraction(1)}))
    return total, invariant


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "results/dla.json"
    obs = add(P({0: "X"}, Fraction(1, 2)), P({4: "X"}, Fraction(1, 2)))
    d_eq, d_neq = dla(eq), dla(neq)
    m_eq, m_neq = module_basis(eq, obs), module_basis(neq, obs)
    # the circuit of Equiv with a readout that has no symmetry, (X_0 + Z_0 + Z_4) / (1 + sqrt 2) (up to its norm)
    m_ns = module_basis(eq, add(P({0: "X"}), P({0: "Z"}), P({4: "Z"})))
    feats = {"Equiv": [real_part(v) for v in m_eq], "NonEquiv": [real_part(v) for v in m_neq],
             "Twirled": [real_part(twirl(v)) for v in m_neq], "NoSym": [real_part(v) for v in m_ns]}
    feats = {k: [v for v in vs if v] for k, vs in feats.items()}
    dim = {k: rank(vs) for k, vs in feats.items()}
    inv = {k: fixed_dim(vs) for k, vs in feats.items()}
    pair = {f"{a}&{b}": dim[a] + dim[b] - rank(feats[a] + feats[b]) for a, b in itertools.combinations(feats, 2)}
    total, total_inv = all_forms()
    n_obs = {"Equiv": len(m_eq), "NonEquiv": len(m_neq), "Twirled": rank([twirl(v) for v in m_neq]),
             "NoSym": len(m_ns)}
    dlas = {"Equiv": d_eq, "NonEquiv": d_neq, "Twirled": d_neq, "NoSym": d_eq}
    rows = []
    for k in ("Equiv", "Twirled", "NoSym", "NonEquiv"):
        shared = dim[k] if k == "Equiv" else pair[f"Equiv&{k}"]
        rows.append((k, dlas[k], n_obs[k], dim[k], inv[k], shared))
    for r in rows:
        print(*r, flush=True)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump({"dla": d_eq, "dla_nonequiv": d_neq, "observables": n_obs, "dim": dim, "inv": inv, "pair": pair,
                   "total": total, "total_inv": total_inv}, f, indent=1)


if __name__ == "__main__":
    main()
