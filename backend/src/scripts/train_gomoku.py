# builtin
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# external

# internal
from src.modules.game.train import GameTrainer


if __name__ == "__main__":
    trainer = GameTrainer("test_train")
    
    trainer.train_multiple(3000)