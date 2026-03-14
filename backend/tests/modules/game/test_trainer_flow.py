import sys
from pathlib import Path
import shutil
import numpy as np


def test_trainer_one_step(tmp_path):
    # ensure backend/src is on path so package imports work
    repo_backend = Path(__file__).resolve().parents[1]
    src_dir = repo_backend / "src"
    sys.path.insert(0, str(src_dir))

    from modules.game.train.trainer import GameTrainer
    from modules.game.elements.action import GameAction
    from modules.game.constants import FEATURE_IN_DIM, POLICY_OUT_DIM

    # create minimal weight dirs expected by GameAgent
    created_dirs = []
    for name in ("player_one", "player_two"):
        p = Path.cwd() / name
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "policy_weights.npy", np.random.random((FEATURE_IN_DIM, POLICY_OUT_DIM)))
        np.save(p / "value_weights.npy", np.random.random((FEATURE_IN_DIM, 1)))
        created_dirs.append(p)

    try:
        trainer = GameTrainer()

        # monkeypatch agent decision/improve to avoid using full policy/value machinery
        trainer.agent_1.decide_train = lambda state: GameAction(trainer.agent_1.get_player_index(), (0, 0))
        trainer.agent_1.improve = lambda *args, **kwargs: None
        trainer.agent_2.decide_train = lambda state: GameAction(trainer.agent_2.get_player_index(), (0, 1))
        trainer.agent_2.improve = lambda *args, **kwargs: None

        # set trainer.agent so run checks pass (GameTrainer checks self.agent)
        trainer.agent = trainer.agent_1

        # run a single step of the flow
        state = trainer.environment.reset()
        action = trainer.agent_1.decide_train(state)
        new_state, reward = trainer.environment.step(action)
        trainer.agent_1.improve(state, action, new_state, reward)

        # basic assertion: board changed after the step
        assert (new_state.get_board() != state.get_board()).any()

    finally:
        # cleanup created weight dirs
        for d in created_dirs:
            try:
                shutil.rmtree(d)
            except Exception:
                pass
