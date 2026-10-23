"""LoRA by hand: who sets A and B? Training does, step by step, on words of 4 numbers.

Author: Yves Schillings.

Run:  python examples/21_lora_by_hand.py

Need: show, with numbers small enough to read, how training finds the two small LoRA
tables A and B. Nobody writes their values. The person only sets their shape and the
starting point: A gets small random numbers, B gets zeros.

The setting (made up, for the explanation):
- A word is 4 numbers that describe its meaning: animal, computer, office object, name.
- W is a frozen table. It turns the 4 numbers of a word into 4 new numbers.
- A has 1 row of 4 weights: it turns the 4 numbers into 1 score.
- B has 1 column of 4 weights: it turns the score into 4 numbers added to W's result.
- Result for a word x: W x + B (A x).
- Goal: for «Heron», the code name of the payroll server, add 0,8 to «computer» and to
  «name». For «printer» and «sparrow», change nothing.

Each training step: compute the result, measure the error against the goal, compute for
each number of A and B whether it must go up or down (the gradient), and move it a little.
"""
import numpy as np

from fictional_data import save_json

MEANING = ["animal", "computer", "office object", "name"]
WORDS = {"Heron": [0.9, 0.1, 0.0, 0.7], "printer": [0.1, 0.8, 0.3, 0.0], "sparrow": [0.9, 0.0, 0.0, 0.1]}
W = np.array([[0.5, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 0.5]])
WANTED_CHANGE = {"Heron": [0, 0.8, 0, 0.8], "printer": [0, 0, 0, 0], "sparrow": [0, 0, 0, 0]}
LEARNING_RATE = 0.5
STEPS = 300
SHOW = [0, 1, 2, 5, 20, 100, 300]


def correction(A, B, x):
    """What A and B add to the word: the score A x, multiplied by the column B."""
    return B * (A @ x)


def error(A, B, X, T):
    """Mean squared gap between what A and B add and the wanted change."""
    return float(np.mean([np.sum((correction(A, B, x) - t) ** 2) for x, t in zip(X, T)]))


def train(seed=21):
    rng = np.random.default_rng(seed)
    X = [np.array(v) for v in WORDS.values()]
    T = [np.array(v) for v in WANTED_CHANGE.values()]       # W x is frozen, so only the change has to be learned
    A = rng.normal(0, 0.02, 4)                              # A starts with small random numbers
    B = np.zeros(4)                                         # B starts at zero: at first nothing is added
    history = []
    for step in range(STEPS + 1):
        if step in SHOW:
            history.append({"step": step, "A": A.round(3).tolist(), "B": B.round(3).tolist(),
                            "scores": {w: round(float(A @ x), 3) for w, x in zip(WORDS, X)},
                            "correction_Heron": correction(A, B, X[0]).round(3).tolist(),
                            "error": round(error(A, B, X, T), 4)})
        if step == STEPS:
            break
        grad_A, grad_B = np.zeros(4), np.zeros(4)
        for x, t in zip(X, T):
            gap = correction(A, B, x) - t                  # how far each of the 4 numbers is from the goal
            grad_B += 2 * gap * (A @ x) / len(X)            # B moves along the score
            grad_A += 2 * (B @ gap) * x / len(X)            # A moves along the word, but only once B is not 0
        A -= LEARNING_RATE * grad_A                         # each number moves a little, against its gradient
        B -= LEARNING_RATE * grad_B
    final = {w: (W @ x + correction(A, B, x)).round(2).tolist() for w, x in zip(WORDS, X)}
    return {"meaning": MEANING, "words": WORDS, "W": W.tolist(), "wanted_change": WANTED_CHANGE,
            "learning_rate": LEARNING_RATE, "steps": STEPS, "seed": seed, "history": history,
            "final_result": final}


if __name__ == "__main__":
    result = train()
    for h in result["history"]:
        print(f"step {h['step']:>3}: A = {h['A']}  B = {h['B']}  scores = {h['scores']}  "
              f"Heron gets {h['correction_Heron']}  error = {h['error']}")
    save_json("21_lora_by_hand.json", result)
