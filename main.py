"""
main package
"""
import pickle

# from simulator.simulator import Constraint, Simulator
from simulator.noz3_simulator import Constraint, Simulator
from utils.config import Config


def main():
    """main function"""
    config = Config(
        {
            "world_size": 128,
            "global_batch_size": 1 * (1024**2),
            "sequence_length": 4 * 1024,
            "model_size": 7,
            "vocab_size": 103168,
            "dtype_size": 2,
            "use_fa": 1,
        }
    )

    externl_sim = Constraint(
        config.world_size,
        config.global_batch_size,
        config.sequence_length,
        config=config,
    )
    externl_sim.run_loop()


if __name__ == "__main__":
    main()
