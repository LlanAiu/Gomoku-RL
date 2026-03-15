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
        trainer = GameTrainer("")

        # create optimization methods for both agents (one-step actor-critic)
        from modules.rl.optimization.one_step_actor_critic import OneStepActorCritic

        ac1 = OneStepActorCritic(policy=trainer.agent_1.policy,
                                value_function=trainer.agent_1.value_function,
                                discount=0.99,
                                policy_step_size=0.01,
                                value_step_size=0.01)

        ac2 = OneStepActorCritic(policy=trainer.agent_2.policy,
                                value_function=trainer.agent_2.value_function,
                                discount=0.99,
                                policy_step_size=0.01,
                                value_step_size=0.01)

        # wire the agents to use the actor-critic improve method
        trainer.agent_1.improve = lambda old_s, a, new_s, r: ac1.improve(old_s, a, new_s, r)
        trainer.agent_2.improve = lambda old_s, a, new_s, r: ac2.improve(old_s, a, new_s, r)

        # run a short episode using the real policies/value functions
        trainer.agent = trainer.agent_1
        state = trainer.environment.reset()
        ac1.reset()
        ac2.reset()
        # run until terminal or up to a small cap
        steps = 0
        max_steps = 10
        while not state.is_terminal() and steps < max_steps:
            actor = trainer.agent_1 if state.get_board().sum() % 2 == 0 else trainer.agent_2
            action = actor.decide_train(state)
            new_state, reward = trainer.environment.step(action)
            actor.improve(state, action, new_state, reward)
            state = new_state
            steps += 1

        # assert episode made progress (some moves played)
        assert steps > 0

    finally:
        # cleanup created weight dirs
        for d in created_dirs:
            try:
                shutil.rmtree(d)
            except Exception:
                pass
