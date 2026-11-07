"""Action-conditioned heads and frozen-head controls; no generator updates."""
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import time

from train_gate_baseline import calibrate, decisions, metrics


def actionable(row):
    return bool(row["valid"] and row["candidate_answer"] != row["initial_answer"])


def eligible_rows(rows):
    return [dict(r, valid=actionable(r)) for r in rows]


def project_probabilities(probabilities, rows):
    """Known answer constraints, never inferred correctness; inactive rows remain unscored."""
    import numpy as np
    p = np.array(probabilities, dtype=float, copy=True)
    for i, row in enumerate(rows):
        if not actionable(row):
            p[i] = 0  # no predicted posterior; masked out by the action predicate
            continue
        p[i, 3] = 0  # different closed answers cannot both match a single label
        if row["initial_answer"] not in list("ABCD"):
            p[i, 2] = 0  # original invalid => c0=0 under the frozen forced-answer metric
        total = p[i].sum()
        if total <= 0:
            raise ValueError("No feasible outcome probability mass")
        p[i] /= total
    return p


def conditional_logits(net, x, rows):
    import torch
    logits = net(x)
    invalid_initial = torch.tensor([r["initial_answer"] not in list("ABCD") for r in rows], device=x.device)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, 2] = invalid_initial
    return logits.masked_fill(mask, -1e9)


def case_rank_loss(logits, gains, actions):
    import torch
    p = logits.softmax(-1)
    values = p[..., 1]-p[..., 2]
    losses = []
    for i in range(logits.shape[0]):
        # One KEEP, then only genuine actions. Never duplicate inactive slots as KEEP.
        u = [values[i].sum()*0]; g = [0.]
        for j in range(2):
            if bool(actions[i, j]):
                u.append(values[i, j]); g.append(float(gains[i, j]))
        pairs = []
        for a in range(len(g)):
            for b in range(a+1, len(g)):
                if g[a] != g[b]:
                    direction = 1 if g[a] > g[b] else -1
                    pairs.append((.1-direction*(u[a]-u[b])).clamp_min(0))
        losses.append(torch.stack(pairs).mean() if pairs else logits[i].sum()*0)
    return torch.stack(losses).mean()


def make_net(d, classes):
    from torch import nn
    return nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, classes))


def old_data(root):
    import numpy as np
    old = Path((root/"shared_root.txt").read_text().strip())
    data = np.load(old/"outputs/features.npz")
    supervision = {r["feature_id"]: r for r in map(json.loads,
        (old/"inputs/supervision.private.jsonl").read_text().splitlines())}
    # Exclude old diagnostic rows before labels/tensors enter training or selection.
    ix = [i for i, fid in enumerate(data["ids"].tolist()) if supervision[fid]["split"] in {"train", "dev", "calibration"}]
    rows = [supervision[data["ids"][i]] for i in ix]
    assert len(rows) == 2800 and len({r["case_id"] for r in rows}) == 1400
    for i in range(0, len(rows), 2):
        assert rows[i]["case_id"] == rows[i+1]["case_id"] and rows[i]["split"] == rows[i+1]["split"]
    return old, data["x"][ix].copy(), rows


def fit(root):
    import numpy as np
    import torch
    from torch.nn import functional as F
    torch.set_num_threads(4)
    out = root/"outputs"
    if (out/"policies_locked.json").exists():
        raise RuntimeError("Policies already locked; no repeat fitting")
    started = time.time()
    old, array, rows = old_data(root)
    indices = {s: [i for i, r in enumerate(rows) if r["split"] == s] for s in ["train", "dev", "calibration"]}
    xraw = torch.tensor(array, device="cuda")
    mean = xraw[indices["train"]].mean(0)
    std = xraw[indices["train"]].std(0, unbiased=False).clamp_min(1e-6)
    x = (xraw-mean)/std
    y = torch.tensor([r["class"] for r in rows], device="cuda")
    act = torch.tensor([actionable(r) for r in rows], device="cuda")
    if torch.any(y[act] == 3):
        raise ValueError("Actionable CC violates single-answer score semantics")
    gain = torch.tensor([r["ca"]-r["c0"] for r in rows], device="cuda", dtype=torch.float32)
    train_pairs = torch.tensor(indices["train"], device="cuda").reshape(-1, 2)
    dev = [i for i in indices["dev"] if actionable(rows[i])]
    if not dev:
        raise RuntimeError("No actionable dev examples; no resampling")
    cal_rows = [rows[i] for i in indices["calibration"]]
    cal_eligible = eligible_rows(cal_rows)
    specs = []
    for config, alpha in [("action_ce", 0.), ("action_ce_rank", .1)]:
        for seed in [2026110711, 2026110712, 2026110713]:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            rng = random.Random(seed)
            net = make_net(x.shape[1], 3).cuda()
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=.01)
            best_loss, best_epoch, stale = float("inf"), 0, 0
            history, best = [], None
            for epoch in range(1, 101):
                net.train(); order = list(range(len(train_pairs))); rng.shuffle(order)
                total_loss = 0.; total_cases = 0
                for start in range(0, len(order), 32):
                    ix = train_pairs[order[start:start+32]]
                    batch_n = ix.shape[0]; total_cases += batch_n
                    active = act[ix].flatten()
                    if not active.any():
                        continue
                    rr = [rows[i] for i in ix.flatten().tolist()]
                    logits = conditional_logits(net, x[ix.flatten()], rr).reshape(batch_n, 2, 3)
                    # Sum active slot losses / (2 * case count), not mean over active rows.
                    ce = F.cross_entropy(logits.flatten(0, 1)[active], y[ix].flatten()[active], reduction="sum")/(2*batch_n)
                    loss = ce + alpha*case_rank_loss(logits, gain[ix], act[ix]) if alpha else ce
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite conditional loss")
                    opt.zero_grad(); loss.backward(); opt.step()
                    total_loss += float(loss.detach())*batch_n
                net.eval()
                with torch.no_grad():
                    dev_logits = conditional_logits(net, x[dev], [rows[i] for i in dev])
                    dev_ce = float(F.cross_entropy(dev_logits, y[dev]))
                history.append({"epoch": epoch, "train_case_loss": total_loss/total_cases, "dev_action_ce": dev_ce})
                if dev_ce < best_loss:
                    best_loss, best_epoch, stale = dev_ce, epoch, 0
                    best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                if stale >= 15:
                    break
            net.load_state_dict(best); net.eval()
            with torch.no_grad():
                dev_logits = conditional_logits(net, x[dev], [rows[i] for i in dev])
                temp = min([.5, .75, 1, 1.5, 2, 3], key=lambda t: float(F.cross_entropy(dev_logits/t, y[dev])))
                p3 = (conditional_logits(net, x[indices["calibration"]], cal_rows)/temp).softmax(-1).cpu().numpy()
            p4 = np.pad(p3, ((0, 0), (0, 1)))
            policy = calibrate(cal_eligible, p4)
            name = config+"_"+str(seed)
            torch.save({"state_dict": best, "mean": mean.cpu(), "std": std.cpu(), "dimension": x.shape[1],
                        "temperature": temp, "classes": 3}, out/(name+".pt"))
            (out/(name+"_history.json")).write_text(json.dumps(history))
            specs.append({"name": name, "family": config, "seed": seed, "checkpoint": name+".pt",
                "policy": policy, "temperature": temp, "best_epoch": best_epoch, "epochs": len(history),
                "best_dev_action_ce": best_loss, "trainable_parameters": sum(p.numel() for p in net.parameters())})
            print("MODEL_FITTED", name, best_epoch, flush=True)
    # Controls use already trained G2 checkpoints; no repeated training or test-based selection.
    controls = [{"config": config, "seed": seed} for config in ["ce", "ce_rank"]
                for seed in [2026110711, 2026110712, 2026110713]]
    for item in controls:
        name = item["config"]+"_"+str(item["seed"])
        ckpt = torch.load(old/"outputs"/(name+".pt"), map_location="cpu", weights_only=True)
        net = make_net(ckpt["dimension"], 4).cuda().eval(); net.load_state_dict(ckpt["state_dict"])
        with torch.no_grad():
            xx = (xraw[indices["calibration"]]-ckpt["mean"].cuda())/ckpt["std"].cuda()
            p = (net(xx)/ckpt["temperature"]).softmax(-1).cpu().numpy()
        specs.append({"name": "frozen_"+name, "family": "frozen_"+item["config"], "seed": item["seed"],
            "shared_checkpoint": name+".pt", "policy": ckpt["policy"], "project": False})
        specs.append({"name": "projected_"+name, "family": "projected_"+item["config"], "seed": item["seed"],
            "shared_checkpoint": name+".pt", "policy": calibrate(cal_eligible, project_probabilities(p, cal_rows)), "project": True})
    summary = {"status": "locked", "locked_epoch": time.time(), "training_seconds": time.time()-started,
        "models": specs, "old_diagnostic_rows_used": 0,
        "actionable_rows": {s: sum(actionable(rows[i]) for i in ids) for s, ids in indices.items()},
        "actionable_train_classes": dict(Counter(rows[i]["class"] for i in indices["train"] if actionable(rows[i]))),
        "new_diagnostic_labels_read": False}
    (out/"policies_locked.json").write_text(json.dumps(summary, indent=2))
    print("POLICIES_LOCKED", len(specs), flush=True)


def evaluate(root):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    out = root/"outputs"
    if (out/"metrics.json").exists():
        raise RuntimeError("Final evaluation already exists")
    locked = json.loads((out/"policies_locked.json").read_text())
    if locked["status"] != "locked":
        raise RuntimeError("Models/policies must be fixed before new labels are read")
    rows_by_id = {r["feature_id"]: r for r in map(json.loads, (root/"inputs/supervision.private.jsonl").read_text().splitlines())}
    data = np.load(out/"features.npz")
    rows = [rows_by_id[fid] for fid in data["ids"].tolist()]
    assert len(rows) == 400 and len({r["case_id"] for r in rows}) == 200
    old = Path((root/"shared_root.txt").read_text().strip())
    x = torch.tensor(data["x"], device="cuda")
    results, private = [], {}
    for spec in locked["models"]:
        path = (old/"outputs"/spec["shared_checkpoint"]) if "shared_checkpoint" in spec else out/spec["checkpoint"]
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        classes = ckpt.get("classes", 4)
        net = make_net(ckpt["dimension"], classes).cuda().eval(); net.load_state_dict(ckpt["state_dict"])
        xx = (x-ckpt["mean"].cuda())/ckpt["std"].cuda()
        with torch.no_grad():
            logits = conditional_logits(net, xx, rows) if classes == 3 else net(xx)
            p = (logits/ckpt["temperature"]).softmax(-1).cpu().numpy()
        if classes == 3:
            p = np.pad(p, ((0, 0), (0, 1))); eligible = eligible_rows(rows)
        elif spec["project"]:
            p = project_probabilities(p, rows); eligible = eligible_rows(rows)
        else:
            eligible = rows
        policy = spec["policy"]
        cases = decisions(eligible, p, policy["lambda"], policy["delta"])
        m = metrics(cases)
        rng = random.Random(2026110717)
        samples = sorted(sum(r["ca"]-r["c0"] for r in rng.choices(cases, k=200))/200 for _ in range(2000))
        m["delta_accuracy_95ci"] = [samples[49], samples[1949]]
        results.append({"name": spec["name"], "family": spec["family"], "seed": spec["seed"], "metrics": m})
        private[spec["name"]] = cases
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    simple = {}
    for name in ["keep", "selective", "reset", "pool_oracle_upper_bound"]:
        cases = []
        for pair in by_case.values():
            b = pair[0]
            if name == "keep": ca, edited = b["c0"], False
            elif name == "pool_oracle_upper_bound": ca, edited = max([b["c0"]]+[r["ca"] for r in pair]), False
            else:
                r = next(r for r in pair if r["branch"] == name)
                ca, edited = r["ca"], r["candidate_answer"] != b["initial_answer"]
            cases.append({"c0": b["c0"], "ca": ca, "edited": edited, "accepted": name in ["selective", "reset"]})
        simple[name] = metrics(cases)
        if name == "pool_oracle_upper_bound":
            simple[name].pop("edit_coverage"); simple[name].pop("acceptance_coverage")
    summary = {"status": "completed", "n": 200, "models": results, "simple_baselines": simple,
        "candidate_class_counts": dict(Counter(r["class"] for r in rows)),
        "actionable_candidate_rows": sum(actionable(r) for r in rows),
        "invalid_candidate_rows": sum(not r["valid"] for r in rows),
        "empty_explanation_rows": sum(r["explanation_missing"] for r in rows),
        "finished_epoch": time.time(), "independent_clinical_evidence_audit": False}
    (out/"paired_outcomes.private.json").write_text(json.dumps({
        "case_order": list(dict.fromkeys(r["case_id"] for r in rows)), "outcomes": private}))
    (out/"metrics.json").write_text(json.dumps(summary, indent=2))
    print("EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    root = Path(os.environ["RUN_ROOT"])
    fit(root) if os.environ.get("ACTION_STAGE") == "fit" else evaluate(root)
