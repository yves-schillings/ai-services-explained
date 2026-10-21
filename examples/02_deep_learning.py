"""Train a small two-hidden-layer neural network on fictional sensor flags.

Run: python "02_deep_learning.py"
Requires numpy. XOR is an educational nonlinear problem, not a maintenance model.
No GPU or downloaded weights. Train and evaluation data overlap intentionally:
the result is a learning demonstration, not a generalisation benchmark.
"""
import numpy as np

def fit(steps=6000):
    rng = np.random.default_rng(42)
    x = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = np.array([[0.], [1.], [1.], [0.]])
    w1, w2, w3 = [rng.normal(0, .7, size=s) for s in [(2, 8), (8, 4), (4, 1)]]
    b1, b2, b3 = np.zeros((1, 8)), np.zeros((1, 4)), np.zeros((1, 1))
    for _ in range(steps):
        h1 = np.tanh(x @ w1 + b1)
        h2 = np.tanh(h1 @ w2 + b2)
        p = 1 / (1 + np.exp(-(h2 @ w3 + b3)))
        d3 = (p - y) / len(x)
        d2 = (d3 @ w3.T) * (1 - h2 ** 2)
        d1 = (d2 @ w2.T) * (1 - h1 ** 2)
        # Compute all gradients before changing any parameter.
        g1, g2, g3 = x.T @ d1, h1.T @ d2, h2.T @ d3
        w1 -= .2 * g1; w2 -= .2 * g2; w3 -= .2 * g3
        b1 -= .2 * d1.sum(axis=0); b2 -= .2 * d2.sum(axis=0); b3 -= .2 * d3.sum(axis=0)
    return p.ravel()

if __name__ == "__main__":
    print("Toy training-set probabilities:", fit().round(4).tolist())
