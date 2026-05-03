"""
deep_homography/cli.py
----------------------
Console-script entry points wired via pyproject.toml.

After `pip install -e .`:
    dh-train    ← train_entry()
    dh-evaluate ← eval_entry()
    dh-demo     ← demo_entry()
"""

def train_entry():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from train import main
    main()


def eval_entry():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluate import main
    main()


def demo_entry():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from demo import main
    main()
