import json
from training.build_synthetic_dataset import examples_per_label, MODULES


def test_balanced_unique_synthetic_dataset():
    rows=examples_per_label(25)
    assert len(rows)==25*len(MODULES)
    assert len({row["id"] for row in rows})==len(rows)
    assert {row["label"] for row in rows}==set(MODULES)
    assert all(row["provenance"]=="synthetic-from-cited-factual-knowledgebase" for row in rows)
