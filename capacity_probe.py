"""Fixed CPU capacity diagnosis; no diagnostic-evaluation rows enter fitting/reporting."""
import json
import os
from pathlib import Path
import random
import time

from action_gate import old_data, make_net
from diagnose_gate_scores import auc
from train_gate_baseline import calibrate, decisions, metrics, rank_loss


def main():
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(4)
    root = Path(os.environ["RUN_ROOT"])
    out = root/"outputs"
    if (out/"training_done.json").exists():
        raise RuntimeError("Completed probe cannot be repeated")
    started = time.time()
    old, array, rows = old_data(root)
    assert {r["split"] for r in rows} == {"train", "dev", "calibration"}
    ix = {s: [i for i, r in enumerate(rows) if r["split"] == s]
          for s in ["train", "dev", "calibration"]}
    raw = torch.tensor(array, dtype=torch.float32)
    mean = raw[ix["train"]].mean(0)
    std = raw[ix["train"]].std(0, unbiased=False).clamp_min(1e-6)
    x = (raw-mean)/std
    y = torch.tensor([r["class"] for r in rows], dtype=torch.long)
    gains = torch.tensor([r["ca"]-r["c0"] for r in rows], dtype=torch.float32)
    pairs = torch.tensor(ix["train"]).reshape(-1, 2)
    cal_rows = [rows[i] for i in ix["calibration"]]

    def diagnostic(net, center, scale, temperature):
        result = {}
        with torch.no_grad():
            for split in ["dev", "calibration"]:
                logits = net((raw[ix[split]]-center)/scale)
                p = (logits/temperature).softmax(-1)
                labels = y[ix[split]]
                events = (labels == 1) | (labels == 2)
                result[split] = {"candidate_rows": len(ix[split]),
                    "raw_nll": float(F.cross_entropy(logits, labels)),
                    "temperature_nll": float(F.cross_entropy(logits/temperature, labels)),
                    "transition_rows": int(events.sum()),
                    "WC_vs_CW_utility_auc": auc((p[:, 1]-p[:, 2])[events], (labels == 1)[events])}
                if split == "calibration":
                    cal_p = p.numpy()
        return result, cal_p

    summaries = []
    for config, alpha in [("ce", 0.), ("ce_rank", .1)]:
        for seed in [2026110711, 2026110712, 2026110713]:
            torch.manual_seed(seed)
            rng = random.Random(seed)
            net = nn.Sequential(nn.LayerNorm(x.shape[1]), nn.Linear(x.shape[1], 4))
            assert sum(p.numel() for p in net.parameters()) == 21508
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=.01)
            best_loss, best_epoch, stale, best = float("inf"), 0, 0, None
            history = []
            for epoch in range(1, 101):
                net.train(); order = list(range(len(pairs))); rng.shuffle(order)
                losses = []
                for start in range(0, len(order), 32):
                    batch = pairs[order[start:start+32]]
                    logits = net(x[batch])
                    loss = F.cross_entropy(logits.flatten(0, 1), y[batch].flatten())
                    if alpha:
                        loss = loss+alpha*rank_loss(logits, gains[batch])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss; no automatic retry")
                    opt.zero_grad(); loss.backward(); opt.step(); losses.append(float(loss.detach()))
                net.eval()
                with torch.no_grad():
                    dev_ce = float(F.cross_entropy(net(x[ix["dev"]]), y[ix["dev"]]))
                history.append({"epoch": epoch, "train_batch_mean_loss": sum(losses)/len(losses), "dev_ce": dev_ce})
                if dev_ce < best_loss:
                    best_loss, best_epoch, stale = dev_ce, epoch, 0
                    best = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    stale += 1
                if stale >= 15:
                    break
            net.load_state_dict(best); net.eval()
            with torch.no_grad():
                logits = net(x[ix["dev"]])
                temperature = min([.5, .75, 1, 1.5, 2, 3],
                    key=lambda t: float(F.cross_entropy(logits/t, y[ix["dev"]])))
            diag, cal_p = diagnostic(net, mean, std, temperature)
            policy = calibrate(cal_rows, cal_p)
            name = config+"_"+str(seed)
            torch.save({"state_dict": best, "mean": mean, "std": std, "temperature": temperature,
                        "policy": policy, "dimension": x.shape[1]}, out/(name+".pt"))
            (out/(name+"_history.json")).write_text(json.dumps(history))
            summaries.append({"config": config, "seed": seed, "linear_parameters": 21508,
                "epochs": len(history), "best_epoch": best_epoch, "temperature": temperature,
                "linear_diagnostics": diag, "linear_policy": policy})
            print("MODEL_FITTED", name, best_epoch, flush=True)

    # Paired frozen controls, no old diagnostic labels/predictions are accessed here.
    for item in summaries:
        name = item["config"]+"_"+str(item["seed"])
        ck = torch.load(old/"outputs"/(name+".pt"), map_location="cpu", weights_only=True)
        net = make_net(ck["dimension"], 4).eval(); net.load_state_dict(ck["state_dict"])
        diag, p = diagnostic(net, ck["mean"], ck["std"], ck["temperature"])
        item["frozen_mlp_diagnostics"] = diag
        item["frozen_mlp_policy"] = ck["policy"]
        item["frozen_mlp_calibration_metrics"] = metrics(decisions(cal_rows, p,
            ck["policy"]["lambda"], ck["policy"]["delta"]))
    result = {"status": "completed", "finished_epoch": time.time(), "seconds": time.time()-started,
        "new_heads_trained": 6, "frozen_controls": 6, "models": summaries,
        "train_articles": 1000, "dev_articles": 200, "calibration_articles": 200,
        "diagnostic_eval_rows_used": 0, "new_generator_or_feature_calls": 0,
        "device": "cpu", "threads": 4,
        "scope": "Repeated-use development/capacity diagnosis, not independent efficacy validation"}
    public = root/"results"; public.mkdir(exist_ok=True)
    (public/"capacity_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    (out/"training_done.json").write_text(json.dumps(result, indent=2))
    print("CAPACITY_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
