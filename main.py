"""
main package
"""
import pickle

from simulator.simulator import Constraint, Simulator
from utils.config import Config


def main():
    """main function"""
    config = Config(
        {
            "world_size": 32,
            "global_batch_size": 1 * (1024**2),
            "sequence_length": 4 * 1024,
            "model_size": 7,
            "vocab_size": 103168,
            "dtype_size": 2,
        }
    )

    cost_data_path = "data/cost_data.pickle"
    with open(cost_data_path, "rb") as f:
        cost_data = pickle.load(f)

    externl_sim = Constraint(
        config.world_size,
        config.global_batch_size,
        config.sequence_length,
        config=config,
        cost_data=cost_data,
    )
    externl_sim.run_loop()


if __name__ == "__main__":
    main()
