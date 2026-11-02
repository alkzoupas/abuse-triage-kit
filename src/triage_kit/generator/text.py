"""Tiny text grammars for synthetic reviews, listings and searches.

Benign text is deliberately varied. Family-specific phrase sources (a review
ring's phrasebook, an operator's listing template) live in families.py.
"""

from __future__ import annotations

import numpy as np

ITEMS = [
    "lamp",
    "jacket",
    "bike",
    "camera",
    "desk",
    "chair",
    "kettle",
    "sneakers",
    "backpack",
    "guitar",
    "monitor",
    "phone case",
    "headphones",
    "rug",
    "mirror",
    "tent",
    "blender",
    "watch",
    "scarf",
    "printer",
    "drill",
    "sofa",
    "skates",
    "novel",
    "vase",
]
ADJS = [
    "vintage",
    "used",
    "barely used",
    "handmade",
    "classic",
    "compact",
    "large",
    "small",
    "wooden",
    "leather",
    "blue",
    "black",
    "white",
    "retro",
    "modern",
    "sturdy",
    "lightweight",
]
CONDITIONS = ["like new", "good condition", "some wear", "fully working", "needs repair", "boxed"]
EXTRAS = [
    "pickup only",
    "can ship",
    "price negotiable",
    "smoke-free home",
    "original receipt",
    "moving sale",
    "no returns",
    "comes with charger",
    "spare parts included",
    "",
]
REVIEW_CLAUSES = [
    "arrived on time",
    "good value for the price",
    "not quite what I expected",
    "works fine so far",
    "decent quality",
    "seller answered my questions",
    "took a while to ship",
    "exactly as described",
    "a bit smaller than pictured",
    "would buy again",
    "packaging was damaged but item ok",
    "colour slightly off",
    "easy pickup",
    "smells a bit musty",
    "great condition",
    "missing one screw",
    "my kid loves it",
    "fair price",
    "had to clean it first",
    "quick reply from seller",
]


def pick(rng: np.random.Generator, xs: list[str]) -> str:
    return xs[int(rng.integers(len(xs)))]


def benign_review(rng: np.random.Generator) -> str:
    k = int(rng.integers(1, 4))
    idx = rng.choice(len(REVIEW_CLAUSES), size=k, replace=False)
    text = ", ".join(REVIEW_CLAUSES[int(i)] for i in idx)
    return text[0].upper() + text[1:] + "."


def benign_listing(rng: np.random.Generator) -> str:
    extra = pick(rng, EXTRAS)
    text = f"{pick(rng, ADJS).capitalize()} {pick(rng, ITEMS)}, {pick(rng, CONDITIONS)}"
    return text + (f". {extra.capitalize()}." if extra else ".")


def benign_search(rng: np.random.Generator) -> str:
    if rng.random() < 0.5:
        return pick(rng, ITEMS)
    return f"{pick(rng, ADJS)} {pick(rng, ITEMS)}"
