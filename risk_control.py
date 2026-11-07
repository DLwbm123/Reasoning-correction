"""Fixed-grid empirical/one-sided CP acceptance control; no clinical guarantee."""
import math
from train_gate_baseline import decisions, metrics

GRID = [(penalty, threshold) for penalty in [1, 2, 4] for threshold in [0, .05, .1, .2]]


def upper_bound(k, n, alpha=.05):
    if not 0 <= k <= n or not 0 < alpha < 1:
        raise ValueError("Invalid binomial bound arguments")
    if n == 0 or k == n: return 1.
    if k == 0: return -math.expm1(math.log(alpha)/n)
    from scipy.stats import beta
    return float(beta.ppf(1-alpha, k+1, n-k))


def apply_policy(rows, probabilities, policy):
    if not policy.get("keep_only"):
        return decisions(rows, probabilities, policy["lambda"], policy["delta"])
    grouped = {}
    for row in rows:
        grouped.setdefault(row["case_id"], {"c0": row["c0"], "ca": row["c0"],
                                           "edited": False, "accepted": False})
    return list(grouped.values())


def select_policy(rows, probabilities, *, exact=False):
    best, table = None, []
    for penalty, threshold in GRID:
        cases = decisions(rows, probabilities, penalty, threshold)
        m = metrics(cases)
        upper = upper_bound(m["CW"], m["initial_correct"], .05/24)
        eligible = upper <= .02 if exact else (m["initial_correct"] > 0 and m["harm_given_initial_correct"] <= .02)
        table.append({"lambda": penalty, "delta": threshold, "metrics": m,
                      "simultaneous_upper": upper, "admissible": eligible})
        key = (m["WC"]-m["CW"], -sum(r["edited"] for r in cases), penalty, threshold)
        if eligible and (best is None or key > best[0]):
            best = (key, {"lambda": penalty, "delta": threshold, "keep_only": False,
                          "calibration_metrics": m, "calibration_upper": upper})
    if best: return best[1], table
    return {"lambda": 4, "delta": 1, "keep_only": True,
            "reason": "No fixed-grid rule met the registered constraint",
            "explicit_constant_keep_harm": 0.}, table
