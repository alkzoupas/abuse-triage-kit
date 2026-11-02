import triage_kit


def test_package_has_version() -> None:
    assert triage_kit.__version__ == "0.1.0"
