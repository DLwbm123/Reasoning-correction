"""Post-hoc frozen-head diagnostics on dev/cal ONLY; no fitting or eval tuning."""
from collections import Counter
import json
import math
import os
from pathlib import Path


def auc(scores, positive):
    yes = [float(x) for x, y in zip(scores, positive) if y]
    no = [float(x) for x, y in zip(scores, positive) if not y]
    if not yes or not no:
        return None
    return sum((a > b) + .5*(a == b) for a in yes for b in no)/(len(yes)*len(no))


def main():
    import numpy as np
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(4)
    root = Path(os.environ["RUN_ROOT"])
    out, public = root/"outputs", root/"results"
    assert json.loads((out/"supervisor.json").read_text())["status"] == "completed"
    data = np.load(out/"features.npz")
    mapping = {r["feature_id"]: r for r in map(json.loads, (root/"inputs/supervision.private.jsonl").read_text().splitlines())}
    rows = [mapping[fid] for fid in data["ids"].tolist()]
    train_counts = Counter(r["class"] for r in rows if r["split"] == "train")
    prior = torch.tensor([train_counts[i]/sum(train_counts.values()) for i in range(4)])
    result = {"scope": "post-hoc frozen-head diagnosis; dev/cal only, no new fitting, no eval predictions used",
              "train_class_prior": prior.tolist(), "models": []}
    for model in json.loads((out/"metrics.json").read_text())["models"]:
        name = model["config"]+"_"+str(model["seed"])
        ckpt = torch.load(out/(name+".pt"), map_location="cpu", weights_only=True)
        d = ckpt["dimension"]
        net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 512), nn.GELU(), nn.Dropout(.1), nn.Linear(512, 4)).eval()
        net.load_state_dict(ckpt["state_dict"])
        item = {"config": model["config"], "seed": model["seed"], "splits": {}}
        for split in ["dev", "calibration"]:
            ix = [i for i, r in enumerate(rows) if r["split"] == split]
            x = (torch.tensor(data["x"][ix])-ckpt["mean"])/ckpt["std"]
            labels = torch.tensor([rows[i]["class"] for i in ix])
            with torch.no_grad():
                logits = net(x)/ckpt["temperature"]
                p = logits.softmax(-1)
            meaningful = (labels == 1) | (labels == 2)
            item["splits"][split] = {"n_candidate_rows": len(ix),
                "temperature_scaled_nll": float(F.cross_entropy(logits, labels)),
                "train_prior_nll": float(-prior[labels].log().mean()),
                "predicted_class_counts": dict(Counter(logits.argmax(-1).tolist())),
                "WC_vs_rest_auc": auc(p[:, 1], labels == 1),
                "CW_vs_rest_auc": auc(p[:, 2], labels == 2),
                "WC_vs_CW_utility_auc": auc((p[:, 1]-p[:, 2])[meaningful], (labels == 1)[meaningful]),
                "transition_candidate_rows": int(meaningful.sum())}
        result["models"].append(item)
    public.mkdir(exist_ok=True)
    (public/"dev_cal_score_diagnostics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
