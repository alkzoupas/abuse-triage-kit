"""Evaluation: truth loading, oracle, splits, metrics, report inputs.

This is the only package allowed to read labels and generator metadata.
Detectors must never import it (enforced by tests/test_leakage.py).
"""
