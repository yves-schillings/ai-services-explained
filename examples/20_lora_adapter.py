"""LoRA: teach a small language model new code words by training two small tables, A and B.

Author: Yves Schillings.

Run:  python examples/20_lora_adapter.py --model-dir PATH
Needs: requirements-lora.txt, a CUDA GPU with about 3 GB free, and the Qwen3-0.6B weights
in a local folder (download command in the README). No other download, no external service.

Need: a help desk receives messages full of in-house code words ("Heron" is the payroll
server) and slang ("bricked" means "does not start"). The base model does not know them.
We want it to rewrite each message in plain words.

How LoRA does it:
- Qwen3-0.6B has 28 layers. Each layer holds 7 tables of weights (matrices): q_proj, k_proj,
  v_proj and o_proj let each word look at the other words; gate_proj, up_proj and down_proj
  work on each word alone. All these weights stay frozen.
- target_modules chooses WHICH TABLES get a LoRA pair; layers_to_transform chooses IN WHICH
  LAYERS. Beside each chosen table, LoRA adds two small new tables: A (reads the 1.024
  numbers of a word and summarises them in 16) and B (turns these 16 numbers back into a
  correction that is added to the frozen table's result). B starts at zero.
- Training shows the model a message and its expected plain answer. The error on the answer
  is back-propagated, and the optimiser changes only A and B.

Two layer choices are trained and compared on the same held-out messages:
  all_layers: a pair beside q, k, v and o in all 28 layers
  last_8:     a pair beside q, k, v and o in layers 20 to 27 only
The base model is measured without and with a glossary in the prompt, the usual alternative
to training when there are only a few code words.
"""
import argparse
import random
import time

from fictional_data import save_json

MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

# ---------------------------------------------------------------------------- fictional data
SYSTEMS = {"Heron": "the payroll server", "Kestrel": "the visitor badge system", "Magpie": "the shared printer"}
STATES = {"bricked": "does not start", "a zombie": "runs but does not respond",
          "ghosting": "keeps dropping off the network"}
KEYS = {"Heron": "payroll server", "Kestrel": "visitor badge", "Magpie": "shared printer",
        "bricked": "not start", "a zombie": "not respond", "ghosting": "dropping off"}
TRAIN_WORDING = ["{s} is {t} again.", "Help, {s} is {t}.", "Since this morning {s} is {t}.",
                 "Can someone check {s}? It is {t}.", "{s} is {t} for the whole team.", "Urgent: {s} is {t}.",
                 "I think {s} is {t} after the update.", "{s} is {t}, please call me back."]
TEST_WORDING = ["Good morning, {s} is {t} today.", "Not sure why, but {s} is {t}.", "Quick note: {s} is {t} on floor 3."]
UNSEEN_PAIRS = {("Magpie", "a zombie"), ("Kestrel", "ghosting")}   # never shown together in training
GLOSSARY = "Glossary: " + " ".join(f"{k} = {v}." for k, v in {**SYSTEMS, **STATES}.items()) + "\n"


def make_data():
    """Training messages use 8 wordings and 7 of the 9 pairs; test messages use 3 other wordings and all 9."""
    def row(s, t, wording):
        plain = f"{SYSTEMS[s]} {STATES[t]}."
        return {"message": wording.format(s=s, t=t), "expected": plain[0].upper() + plain[1:], "system": s,
                "state": t, "unseen_pair": (s, t) in UNSEEN_PAIRS}
    train = [row(s, t, w) for s in SYSTEMS for t in STATES if (s, t) not in UNSEEN_PAIRS for w in TRAIN_WORDING]
    test = [row(s, t, w) for s in SYSTEMS for t in STATES for w in TEST_WORDING]
    return train, test


def prompt(message):
    return f"Rewrite the help-desk message in plain words.\nMessage: {message}\nPlain:"


def glossary_prompt(tok, message):
    """The usual alternative: no training, the meanings are given in the instruction, in Qwen's chat format."""
    system = ("Rewrite the help-desk message in plain words in one sentence. Replace every code word "
              "and slang word by its meaning from the glossary.\n" + GLOSSARY)
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": message}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)


def correct(answer, row):
    """Correct when the plain answer names both the system and its state."""
    a = answer.lower()
    return KEYS[row["system"]] in a and KEYS[row["state"]] in a


# ---------------------------------------------------------------------------- model steps
def load(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float16, local_files_only=True).to("cuda")
    model.config.use_cache = False
    return tok, model


def tables_of_one_layer(model):
    """The 7 tables of weights inside layer 0: name and size (numbers out x numbers in)."""
    return [{"name": name, "out": m.weight.shape[0], "in": m.weight.shape[1], "weights": m.weight.numel()}
            for name, m in model.named_modules() if name.startswith("model.layers.0.") and hasattr(m, "weight")
            and m.weight.dim() == 2]


def add_lora(model, layers):
    """Freeze the model and put an A/B pair beside q, k, v and o in the chosen layers."""
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # which tables
                        layers_to_transform=layers,  # which layers; None means all 28
                        task_type="CAUSAL_LM")
    return get_peft_model(model, config)  # freezes every Qwen weight, adds A and B (B = 0)


def answer(tok, model, message, glossary=False):
    import torch
    text = glossary_prompt(tok, message) if glossary else prompt(message)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True).strip().split("\n")[0]


def evaluate(tok, model, test, glossary=False):
    model.eval()
    rows = [{**r, "answer": answer(tok, model, r["message"], glossary)} for r in test]
    for r in rows:
        r["correct"] = correct(r["answer"], r)
    unseen = [r for r in rows if r["unseen_pair"]]
    return {"correct": sum(r["correct"] for r in rows), "total": len(rows),
            "correct_unseen_pairs": sum(r["correct"] for r in unseen), "unseen_total": len(unseen), "answers": rows}


def train(tok, model, data, epochs=4, lr=1e-3, seed=20):
    """The training loop: guess, measure the error, correct A and B only."""
    import torch
    trainable = [p for p in model.parameters() if p.requires_grad]  # only A and B
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    scaler = torch.amp.GradScaler("cuda")
    losses, start = [], time.time()
    for epoch in range(epochs):
        model.train()
        order = list(range(len(data)))
        random.Random(seed + epoch).shuffle(order)
        for i in order:
            p = prompt(data[i]["message"])
            ids = tok(p + " " + data[i]["expected"] + tok.eos_token, return_tensors="pt").to("cuda")
            labels = ids.input_ids.clone()
            labels[:, :len(tok(p).input_ids)] = -100  # judge the answer only
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**ids, labels=labels).loss  # guess, measure the error
            scaler.scale(loss).backward()  # gradients for A and B only
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)  # change A and B a little
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.item())
        print(f"  epoch {epoch + 1}: mean error {sum(losses[-len(data):]) / len(data):.4f}", flush=True)
    return {"epochs": epochs, "learning_rate": lr, "steps": len(losses), "seconds": round(time.time() - start, 1),
            "first_epoch_mean_loss": round(sum(losses[:len(data)]) / len(data), 4),
            "last_epoch_mean_loss": round(sum(losses[-len(data):]) / len(data), 4)}


def run(model_dir):
    import torch
    random.seed(20)
    torch.manual_seed(20)
    train_set, test_set = make_data()
    tok, model = load(model_dir)
    n_layers = model.config.num_hidden_layers
    layer0 = tables_of_one_layer(model)
    base_weights = sum(p.numel() for p in model.parameters())
    print(f"Qwen3-0.6B: {n_layers} layers, {base_weights:,} weights. Tables in each layer:")
    for t in layer0:
        print(f"  {t['name']:<36} {t['in']:>5} numbers in, {t['out']:>5} out, {t['weights']:>10,} weights")
    results = {"model": MODEL, "revision": REVISION, "device": torch.cuda.get_device_name(0),
               "versions": {m: __import__(m).__version__ for m in ("torch", "transformers", "peft")},
               "layers": n_layers, "base_weights": base_weights, "tables_in_layer_0": layer0,
               "train_messages": len(train_set), "test_messages": len(test_set), "unseen_pairs": sorted(UNSEEN_PAIRS),
               "lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]}}
    print("Base model, no glossary")
    results["base"] = evaluate(tok, model, test_set)
    print("Base model, glossary in the prompt")
    results["base_with_glossary"] = evaluate(tok, model, test_set, glossary=True)
    for name, layers in [("all_layers", None), ("last_8", list(range(n_layers - 8, n_layers)))]:
        del model
        torch.cuda.empty_cache()
        torch.manual_seed(20)
        tok, model = load(model_dir)
        model = add_lora(model, layers)
        pairs = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
        a = model.get_submodule(pairs[0]).weight
        b = model.get_submodule(pairs[0].replace("lora_A", "lora_B")).weight
        info = {"layers_with_pairs": sorted({int(n.split(".layers.")[1].split(".")[0]) for n in pairs}),
                "pairs": len(pairs), "small_tables": 2 * len(pairs),
                "first_pair": {"name": pairs[0].replace(".lora_A.default", "").replace("base_model.model.", ""),
                               "A_numbers_in": a.shape[1], "A_numbers_out": a.shape[0],
                               "B_numbers_in": b.shape[1], "B_numbers_out": b.shape[0],
                               "B_all_zero_at_start": bool((b == 0).all())},
                "trainable_weights": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        print(f"LoRA {name}: {info['pairs']} pairs A+B in layers {info['layers_with_pairs'][0]} to "
              f"{info['layers_with_pairs'][-1]}, {info['trainable_weights']:,} weights learn")
        torch.cuda.reset_peak_memory_stats()
        info["training"] = train(tok, model, train_set)
        info["training"]["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        info["test"] = evaluate(tok, model, test_set)
        results[name] = info
    for key in ("base", "base_with_glossary"):
        print(f"{key}: {results[key]['correct']}/{results[key]['total']}")
    for key in ("all_layers", "last_8"):
        t = results[key]["test"]
        print(f"{key}: {t['correct']}/{t['total']} (unseen pairs {t['correct_unseen_pairs']}/{t['unseen_total']}), "
              f"{results[key]['training']['seconds']} s, peak {results[key]['training']['peak_gpu_gb']} GB")
    save_json("20_lora_adapter.json", results)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, help="local folder with the Qwen3-0.6B weights")
    run(p.parse_args().model_dir)
