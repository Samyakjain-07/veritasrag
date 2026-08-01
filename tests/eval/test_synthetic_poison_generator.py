from eval.synthetic_poison_generator import generate_manifest

# Small parameters so the test suite doesn't need to chunk the full default
# 220-clean/60-poison manifest on every run; determinism/label-correctness
# logic is identical regardless of scale.
SMALL_KWARGS = dict(
    seed=123,
    num_questions=15,
    target_clean_chunks=20,
    num_poison_targets=5,
    poison_per_target=2,
)


def test_generate_manifest_is_reproducible():
    first = generate_manifest(**SMALL_KWARGS)
    second = generate_manifest(**SMALL_KWARGS)

    assert first == second


def test_generate_manifest_different_seed_differs():
    baseline = generate_manifest(**SMALL_KWARGS)
    other = generate_manifest(**{**SMALL_KWARGS, "seed": 999})

    assert baseline != other


def test_manifest_label_sanity():
    """Ground-truth poison labels must be internally consistent."""
    manifest = generate_manifest(**SMALL_KWARGS)

    assert len(manifest) > 0

    chunk_ids = [entry["chunk_id"] for entry in manifest]
    assert len(chunk_ids) == len(set(chunk_ids)), "chunk_ids must be unique"

    required_keys = {"chunk_id", "text", "is_poisoned", "target_question", "source_doc"}
    for entry in manifest:
        assert required_keys <= entry.keys()
        assert isinstance(entry["is_poisoned"], bool)
        assert isinstance(entry["text"], str) and entry["text"]
        assert isinstance(entry["source_doc"], str) and entry["source_doc"]

        if entry["is_poisoned"]:
            assert entry["chunk_id"].startswith("poison-")
            assert isinstance(entry["target_question"], str) and entry["target_question"]
            # a poisoned passage should be crafted around the question it targets
            assert entry["target_question"] in entry["text"]
        else:
            assert entry["chunk_id"].startswith("clean-")
            assert entry["target_question"] is None

    num_poisoned = sum(1 for entry in manifest if entry["is_poisoned"])
    num_clean = sum(1 for entry in manifest if not entry["is_poisoned"])
    assert num_poisoned == SMALL_KWARGS["num_poison_targets"] * SMALL_KWARGS["poison_per_target"]
    assert num_clean > 0


def test_default_manifest_meets_phase1_exit_criteria():
    """Exit criteria: manifest has >=200 clean + >=50 poisoned chunks."""
    manifest = generate_manifest()

    num_clean = sum(1 for entry in manifest if not entry["is_poisoned"])
    num_poisoned = sum(1 for entry in manifest if entry["is_poisoned"])

    assert num_clean >= 200
    assert num_poisoned >= 50
