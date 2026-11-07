"""G6 hybrid four-class gate; train/selection use only original train/dev/cal cache."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time

from action_gate import old_data, make_net
from evidence_score import auxiliary, choose_by_support
from train_gate_baseline import rank_loss, calibrate, decisions, metrics


def score_net(d):
    from torch import nn
    return nn.Sequential(nn.Linear(d, 4)) if d == 10 else make_net(d, 4)


def scores(root, phase):
    values = [json.loads(s) for s in (root/"outputs"/("scores_"+phase+".jsonl")).read_text().splitlines()]
    expected = 1400 if phase == "old" else 200
    assert len(values) == expected and len({r["case_id"] for r in values}) == expected
    assert all(r["status"] == "completed" and len(r["calls"]) == 2 for r in values)
    return {r["case_id"]: r for r in values}


def fit(root):
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4)
    out = root/"outputs"
    if (out/"policies_locked.json").exists():
        raise RuntimeError("No repeated fitting of locked policies")
    begin = time.time()
    old, array, rows = old_data(root)
    sc = scores(root, "old")
    extra = np.array([auxiliary(sc[r["case_id"]], r["initial_answer"], r["candidate_answer"]) for r in rows], dtype=np.float32)
    only_scores = os.environ.get("SUPPORT_REPRESENTATION") == "scores_only"
    raw = torch.tensor(extra if only_scores else np.concatenate([array, extra], axis=1), device="cuda")
    ix = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ["train", "dev", "calibration"]}
    mean, std = raw[ix["train"]].mean(0), raw[ix["train"]].std(0, unbiased=False).clamp_min(1e-6)
    x = (raw-mean)/std
    y = torch.tensor([r["class"] for r in rows], device="cuda")
    gains = torch.tensor([r["ca"]-r["c0"] for r in rows], device="cuda", dtype=torch.float32)
    pairs = torch.tensor(ix["train"], device="cuda").reshape(-1, 2)
    specs = []
    for config, alpha in [("ce", 0.), ("ce_rank", .1)]:
        for seed in [2026110711, 2026110712, 2026110713]:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            rng = random.Random(seed)
            net = score_net(x.shape[1]).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=.01)
            best_loss, best_epoch, stale, best = float("inf"), 0, 0, None
            history = []
            for epoch in range(1, 101):
                net.train(); order = list(range(len(pairs))); rng.shuffle(order)
                losses = []
                for start in range(0, len(order), 32):
                    pair = pairs[order[start:start+32]]
                    logits = net(x[pair])
                    loss = F.cross_entropy(logits.flatten(0, 1), y[pair].flatten())
                    if alpha: loss = loss+alpha*rank_loss(logits, gains[pair])
                    if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
                    opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
                net.eval()
                with torch.no_grad(): dev = float(F.cross_entropy(net(x[ix["dev"]]), y[ix["dev"]]))
                history.append({"epoch": epoch, "train_batch_mean_loss": sum(losses)/len(losses), "dev_ce": dev})
                if dev < best_loss:
                    best_loss, best_epoch, stale = dev, epoch, 0
                    best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else: stale += 1
                if stale >= 15: break
            net.load_state_dict(best); net.eval()
            with torch.no_grad():
                logits = net(x[ix["dev"]])
                temp = min([.5, .75, 1, 1.5, 2, 3], key=lambda t: float(F.cross_entropy(logits/t, y[ix["dev"]])))
                cal_p = (net(x[ix["calibration"]])/temp).softmax(-1).cpu().numpy()
            policy = calibrate([rows[i] for i in ix["calibration"]], cal_p)
            family = ("score_" if only_scores else "hybrid_")+config
            name = family+"_"+str(seed)
            torch.save({"state_dict": best, "mean": mean.cpu(), "std": std.cpu(), "dimension": x.shape[1],
                "temperature": temp, "policy": policy}, out/(name+".pt"))
            (out/(name+"_history.json")).write_text(json.dumps(history))
            specs.append({"name": name, "family": family, "seed": seed, "checkpoint": name+".pt",
                "policy": policy, "best_epoch": best_epoch, "epochs": len(history), "temperature": temp,
                "best_dev_ce": best_loss, "trainable_parameters": sum(p.numel() for p in net.parameters())})
            print("MODEL_FITTED", name, best_epoch, flush=True)
            ck = torch.load(old/"outputs"/(config+"_"+str(seed)+".pt"), map_location="cpu", weights_only=True)
            specs.append({"name": "frozen_"+config+"_"+str(seed), "family": "frozen_"+config,
                "seed": seed, "shared_checkpoint": config+"_"+str(seed)+".pt", "policy": ck["policy"]})
            if only_scores:
                prior = Path((root/"hybrid_root.txt").read_text().strip())
                prior_name = "hybrid_"+config+"_"+str(seed)
                prior_ck = torch.load(prior/"outputs"/(prior_name+".pt"), map_location="cpu", weights_only=True)
                specs.append({"name": prior_name, "family": "hybrid_"+config, "seed": seed,
                    "shared_hybrid_checkpoint": prior_name+".pt", "policy": prior_ck["policy"]})
    locked = {"status": "locked", "locked_epoch": time.time(), "seconds": time.time()-begin,
        "models": specs, "old_diagnostic_rows_used": 0, "new_diagnostic_labels_read": False,
        "old_candidate_rows": len(rows), "old_cases": len(sc), "input_dimension": int(x.shape[1])}
    (out/"policies_locked.json").write_text(json.dumps(locked, indent=2))
    print("POLICIES_LOCKED", len(specs), flush=True)


def evaluate(root):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    out = root/"outputs"
    if (out/"metrics.json").exists(): raise RuntimeError("No repeat of completed evaluation")
    locked = json.loads((out/"policies_locked.json").read_text())
    assert locked["status"] == "locked"
    sc = scores(root, "new")
    data = np.load(out/"features.npz")
    mapping = {r["feature_id"]: r for r in map(json.loads, (root/"inputs/supervision.private.jsonl").read_text().splitlines())}
    rows = [mapping[i] for i in data["ids"].tolist()]
    assert len(rows) == 400 and len({r["case_id"] for r in rows}) == 200
    assert locked["locked_epoch"] < json.loads((out/"collection_done.json").read_text())["finished_epoch"]
    extra = np.array([auxiliary(sc[r["case_id"]], r["initial_answer"], r["candidate_answer"]) for r in rows], dtype=np.float32)
    raw = torch.tensor(data["x"], device="cuda")
    hybrid = torch.tensor(np.concatenate([data["x"], extra], axis=1), device="cuda")
    old = Path((root/"shared_root.txt").read_text().strip())
    models, outcomes = [], {}
    for spec in locked["models"]:
        shared = "shared_checkpoint" in spec
        if shared: path = old/"outputs"/spec["shared_checkpoint"]
        elif "shared_hybrid_checkpoint" in spec:
            path = Path((root/"hybrid_root.txt").read_text().strip())/"outputs"/spec["shared_hybrid_checkpoint"]
        else: path = out/spec["checkpoint"]
        ck = torch.load(path, map_location="cpu", weights_only=True)
        net = score_net(ck["dimension"]).cuda().eval(); net.load_state_dict(ck["state_dict"])
        x = torch.tensor(extra, device="cuda") if ck["dimension"] == 10 else raw if ck["dimension"] == 3584 else hybrid
        with torch.no_grad(): p = (net((x-ck["mean"].cuda())/ck["std"].cuda())/ck["temperature"]).softmax(-1).cpu().numpy()
        policy = spec["policy"]
        cases = decisions(rows, p, policy["lambda"], policy["delta"])
        m = metrics(cases); rng = random.Random(2026110806)
        delta = sorted(sum(r["ca"]-r["c0"] for r in rng.choices(cases, k=200))/200 for _ in range(2000))
        m["delta_accuracy_95ci"] = [delta[49], delta[1949]]
        models.append({"name": spec["name"], "family": spec["family"], "seed": spec["seed"], "metrics": m})
        outcomes[spec["name"]] = cases
    by_case = defaultdict(list)
    for r in rows: by_case[r["case_id"]].append(r)
    simple = {}
    for name in ["keep", "selective", "reset", "pool_oracle_upper_bound"]:
        cases = []
        for pair in by_case.values():
            b = pair[0]
            if name == "keep": ca, edited = b["c0"], False
            elif name == "pool_oracle_upper_bound": ca, edited = max([b["c0"]]+[r["ca"] for r in pair]), False
            else:
                candidate = next(r for r in pair if r["branch"] == name)
                ca, edited = candidate["ca"], candidate["candidate_answer"] != b["initial_answer"]
            cases.append({"c0": b["c0"], "ca": ca, "edited": edited, "accepted": name in ["selective", "reset"]})
        simple[name] = metrics(cases)
        if name == "pool_oracle_upper_bound":
            simple[name].pop("edit_coverage"); simple[name].pop("acceptance_coverage")
    result = {"status": "completed", "n": 200, "models": models, "simple_baselines": simple,
        "candidate_class_counts": dict(Counter(r["class"] for r in rows)),
        "invalid_candidate_rows": sum(not r["valid"] for r in rows),
        "empty_explanation_rows": sum(r["explanation_missing"] for r in rows),
        "independent_evidence_audit": False, "finished_epoch": time.time()}
    if os.environ.get("SUPPORT_REPRESENTATION") == "scores_only":
        for context in ["without_evidence", "with_evidence"]:
            cases = []
            for pair in by_case.values():
                b = pair[0]; logp = sc[b["case_id"]][context]["logp"]
                chosen_index = choose_by_support(pair, logp)
                chosen = pair[chosen_index] if chosen_index is not None else None
                cases.append({"c0": b["c0"], "ca": chosen["ca"] if chosen else b["c0"],
                    "edited": bool(chosen and chosen["candidate_answer"] != b["initial_answer"]), "accepted": chosen is not None})
            simple["fixed_margin_"+context] = metrics(cases)
    (out/"metrics.json").write_text(json.dumps(result, indent=2))
    (out/"paired_outcomes.private.json").write_text(json.dumps({"case_order": list(by_case), "outcomes": outcomes}))
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    root = Path(os.environ["RUN_ROOT"])
    fit(root) if os.environ.get("SUPPORT_STAGE") == "fit" else evaluate(root)
