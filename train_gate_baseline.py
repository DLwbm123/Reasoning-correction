"""Small fixed-pool supervised baseline; retrospective development evaluation."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time


def decisions(rows, probs, penalty, threshold):
    grouped = defaultdict(list)
    for row, prob in zip(rows, probs):
        grouped[row["case_id"]].append((row, float(prob[1]) - penalty * float(prob[2])))
    result = []
    for candidates in grouped.values():
        base = candidates[0][0]
        if any(r["c0"] != base["c0"] for r, _ in candidates):
            raise ValueError("Inconsistent original correctness within a case")
        eligible = [(r, u) for r, u in candidates if r["valid"] and u > threshold]
        chosen = max(eligible, key=lambda pair: pair[1])[0] if eligible else None
        result.append({"c0": base["c0"], "ca": chosen["ca"] if chosen else base["c0"],
            "edited": bool(chosen and chosen["candidate_answer"] != base["initial_answer"]),
            "accepted": chosen is not None})
    return result


def metrics(cases):
    n = len(cases)
    nc = sum(r["c0"] for r in cases)
    wc = sum(not r["c0"] and r["ca"] for r in cases)
    cw = sum(r["c0"] and not r["ca"] for r in cases)
    final = sum(r["ca"] for r in cases)
    assert final - nc == wc - cw
    return {"n": n, "initial_correct": nc, "final_correct": final,
        "initial_accuracy": nc/n, "accuracy": final/n, "delta_accuracy": (wc-cw)/n,
        "WC": wc, "CW": cw, "repair_rate": wc/(n-nc) if n != nc else None,
        "harm_given_initial_correct": cw/nc if nc else None,
        "edit_coverage": sum(r["edited"] for r in cases)/n,
        "acceptance_coverage": sum(r.get("accepted", False) for r in cases)/n}


def calibrate(rows, probs):
    best = None
    for penalty in [1, 2, 4]:
        for threshold in [0, .05, .1, .2]:
            cases = decisions(rows, probs, penalty, threshold)
            m = metrics(cases)
            if (m["harm_given_initial_correct"] or 0) > .02:
                continue
            key = (m["WC"]-m["CW"], -sum(r["edited"] for r in cases), penalty, threshold)
            if best is None or key > best[0]:
                best = (key, {"lambda": penalty, "delta": threshold, "keep_only": False,
                              "calibration_metrics": m})
    return best[1] if best else {"lambda": 4, "delta": 1, "keep_only": True,
        "reason": "No grid point met the empirical development harm constraint"}


def rank_loss(logits, gains):
    import torch
    p = logits.softmax(-1)
    u = torch.cat([p[..., 1] - p[..., 2], torch.zeros_like(gains[:, :1])], dim=1)
    g = torch.cat([gains, torch.zeros_like(gains[:, :1])], dim=1)
    losses = []
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        direction = (g[:, a]-g[:, b]).sign()
        usable = direction != 0
        if usable.any():
            losses.append((.1-direction[usable]*(u[usable, a]-u[usable, b])).clamp_min(0))
    return torch.cat(losses).mean() if losses else logits.sum()*0


def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(4)
    root = Path(os.environ["RUN_ROOT"])
    out = root / "outputs"
    if (out / "training_done.json").exists():
        raise RuntimeError("Refusing to repeat a completed training run")
    data = np.load(out / "features.npz")
    rows_by_id = {r["feature_id"]: r for r in map(json.loads,
        (root / "inputs/supervision.private.jsonl").read_text().splitlines())}
    rows = [rows_by_id[fid] for fid in data["ids"].tolist()]
    for i in range(0, len(rows), 2):
        if rows[i]["case_id"] != rows[i+1]["case_id"] or rows[i]["split"] != rows[i+1]["split"]:
            raise ValueError("Candidate pairing mismatch")
    group_split = {}
    for row in rows:
        prior = group_split.setdefault(row["group_id"], row["split"])
        if prior != row["split"]:
            raise ValueError("Group leakage")
    indices = {s: [i for i, r in enumerate(rows) if r["split"] == s]
               for s in ["train", "dev", "calibration", "diagnostic_eval"]}
    x = torch.tensor(data["x"], device="cuda", dtype=torch.float32)
    mean = x[indices["train"]].mean(0)
    std = x[indices["train"]].std(0, unbiased=False).clamp_min(1e-6)
    x = (x-mean)/std
    y = torch.tensor([r["class"] for r in rows], device="cuda", dtype=torch.long)
    gains = torch.tensor([r["ca"]-r["c0"] for r in rows], device="cuda", dtype=torch.float32)
    train_pairs = torch.tensor(indices["train"], device="cuda").reshape(-1, 2)
    summaries = []
    started = time.time()
    for config, alpha in [("ce", 0.), ("ce_rank", .1)]:
        for seed in [2026110711, 2026110712, 2026110713]:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            rng = random.Random(seed)
            net = nn.Sequential(nn.LayerNorm(x.shape[1]), nn.Linear(x.shape[1], 512), nn.GELU(),
                                nn.Dropout(.1), nn.Linear(512, 4)).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=.01)
            best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
            history = []
            for epoch in range(1, 101):
                net.train(); perm = list(range(len(train_pairs))); rng.shuffle(perm)
                epoch_loss = []
                for start in range(0, len(perm), 32):
                    ix = train_pairs[perm[start:start+32]]
                    logits = net(x[ix])
                    loss = F.cross_entropy(logits.flatten(0, 1), y[ix].flatten())
                    if alpha:
                        loss = loss + alpha*rank_loss(logits, gains[ix])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite loss")
                    opt.zero_grad(); loss.backward(); opt.step()
                    epoch_loss.append(float(loss.detach()))
                net.eval()
                with torch.no_grad():
                    dev_loss = float(F.cross_entropy(net(x[indices["dev"]]), y[indices["dev"]]))
                history.append({"epoch": epoch, "train_loss": sum(epoch_loss)/len(epoch_loss), "dev_ce": dev_loss})
                if dev_loss < best_loss:
                    best_loss, best_epoch, stale = dev_loss, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                print(json.dumps({"config": config, "seed": seed, **history[-1]}), flush=True)
                if stale >= 15:
                    break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                dev_logits = net(x[indices["dev"]])
                temps = [.5, .75, 1, 1.5, 2, 3]
                temp = min(temps, key=lambda t: float(F.cross_entropy(dev_logits/t, y[indices["dev"]])))
                cal_p = (net(x[indices["calibration"]])/temp).softmax(-1).cpu().numpy()
                cal_rows = [rows[i] for i in indices["calibration"]]
                policy = calibrate(cal_rows, cal_p)
                # Evaluation labels are used only after checkpoint/temperature/policy freeze.
                eval_p = (net(x[indices["diagnostic_eval"]])/temp).softmax(-1).cpu().numpy()
            eval_rows = [rows[i] for i in indices["diagnostic_eval"]]
            cases = decisions(eval_rows, eval_p, policy["lambda"], policy["delta"])
            result = metrics(cases)
            boot_rng = random.Random(2026110714)
            diffs = sorted(sum(r["ca"]-r["c0"] for r in boot_rng.choices(cases, k=len(cases)))/len(cases)
                           for _ in range(2000))
            result["delta_accuracy_95ci"] = [diffs[49], diffs[1949]]
            name = config + "_" + str(seed)
            torch.save({"state_dict": best_state, "mean": mean.cpu(), "std": std.cpu(),
                        "temperature": temp, "policy": policy, "dimension": x.shape[1],
                        "class_order": ["WW", "WC", "CW", "CC"]}, out / (name + ".pt"))
            (out / (name + "_history.json")).write_text(json.dumps(history))
            summaries.append({"config": config, "seed": seed, "epochs": len(history),
                "best_epoch": best_epoch, "best_dev_ce": best_loss, "temperature": temp,
                "trainable_parameters": sum(p.numel() for p in net.parameters()),
                "policy": policy, "diagnostic_eval": result})
            (out / "progress.json").write_text(json.dumps({"stage": "training", "finished_models": len(summaries),
                "planned_models": 6, "updated_epoch": time.time()}))
    eval_rows = [rows[i] for i in indices["diagnostic_eval"]]
    by_case = defaultdict(list)
    for r in eval_rows:
        by_case[r["case_id"]].append(r)
    simple = {}
    for name in ["keep", "selective", "reset", "pool_oracle_upper_bound"]:
        cases = []
        for pair in by_case.values():
            base = pair[0]
            if name == "keep":
                ca, edited = base["c0"], False
            elif name == "pool_oracle_upper_bound":
                ca = max([base["c0"]] + [r["ca"] for r in pair])
                edited = False  # No deployable oracle decision/coverage is claimed.
            else:
                chosen = next(r for r in pair if r["branch"] == name)
                ca, edited = chosen["ca"], chosen["candidate_answer"] != base["initial_answer"]
            cases.append({"c0": base["c0"], "ca": ca, "edited": edited,
                          "accepted": name in ["selective", "reset"]})
        simple[name] = metrics(cases)
        if name == "pool_oracle_upper_bound":
            simple[name].pop("edit_coverage"); simple[name].pop("acceptance_coverage")
    summary = {"status": "completed", "models": summaries, "simple_baselines": simple,
        "class_order": ["WW", "WC", "CW", "CC"], "split_candidate_counts": {s: len(ix) for s, ix in indices.items()},
        "split_class_counts": {s: dict(Counter(rows[i]["class"] for i in ix)) for s, ix in indices.items()},
        "all_pool_same_answer_cases": sum(pair[0]["candidate_answer"] == pair[1]["candidate_answer"]
            for pair in [rows[i:i+2] for i in range(0, len(rows), 2)]),
        "invalid_candidate_rows": sum(not r["valid"] for r in rows),
        "missing_explanation_candidate_rows": sum(r["explanation_missing"] for r in rows),
        "training_seconds": time.time()-started, "retrospective_development_only": os.environ.get("FRESH_GROUP_EVAL") != "1",
        "independent_candidate_evidence_audit": False}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out / "training_done.json").write_text(json.dumps({"status": "completed", "finished_epoch": time.time(), "models": 6}))
    print("TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
