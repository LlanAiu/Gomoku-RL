import sys
from pathlib import Path
import shutil
import numpy as np


def test_trainer_full_end_to_end(tmp_path):
    repo_backend = Path(__file__).resolve().parents[1]
    src_dir = repo_backend / "src"
    sys.path.insert(0, str(src_dir))

    from modules.game.train.trainer import GameTrainer
    from modules.game.constants import FEATURE_IN_DIM, POLICY_OUT_DIM

    created_dirs = []
    for name in ("player_one", "player_two"):
        p = Path.cwd() / name
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "policy_weights.npy", np.random.random((FEATURE_IN_DIM, POLICY_OUT_DIM)))
        np.save(p / "value_weights.npy", np.random.random((FEATURE_IN_DIM, 1)))
        created_dirs.append(p)

    try:
        trainer = GameTrainer("what")
        p = Path.cwd() / "what"
        created_dirs.append(p)

        steps = 10
        trainer.train_multiple(steps)
        

    finally:
        # cleanup created weight dirs
        for d in created_dirs:
            try:
                shutil.rmtree(d)
            except Exception:
                pass
