"""Label-free forced-letter support features. Scores are not clinical probabilities."""
import math

SCORE_KEYS = {"question", "choices", "evidence"}


def score_payload(row, evidence, use_evidence):
    return {"question": row["question"], "choices": row["choices"],
            "evidence": evidence if use_evidence else []}


def auxiliary(scores, initial, candidate):
    a, b = scores["without_evidence"], scores["with_evidence"]
    valid0, valid1 = initial in list("ABCD"), candidate in list("ABCD")
    def lookup(p, letter, valid):
        return p["logp"][list("ABCD").index(letter)] if valid else 0.
    v = [lookup(a, initial, valid0), lookup(a, candidate, valid1),
         lookup(b, initial, valid0), lookup(b, candidate, valid1)]
    values = v+[a["entropy"], b["entropy"], v[1]-v[0], v[3]-v[2], float(valid0), float(valid1)]
    assert len(values) == 10 and all(math.isfinite(x) for x in values)
    return values


def check_letter_tokens(tokenizer, prompt):
    ids = []
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    for letter in "ABCD":
        suffix = tokenizer.encode(" "+letter, add_special_tokens=False)
        if len(suffix) != 1 or tokenizer.encode(prompt+" "+letter, add_special_tokens=False) != prefix+suffix:
            raise ValueError("Forced-letter continuation is not one prefix-consistent token")
        ids.append(suffix[0])
    if len(set(ids)) != 4:
        raise ValueError("Choice token IDs collide")
    return ids
