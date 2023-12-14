"""
main package
"""
import pickle

from simulator.simulator import ExternalRestraint, Simulator
from utils.config import Config


def main():
    """main function"""
    config = Config(
        {
            "world_size": 128,
            "global_batch_size": 4 * 1024**2,
            "sequence_length": 16384,
            "model_size": 7,
            "micro_bs": 1,
            "grad_acc": 1,
            "vocab_size": 103168,
            "dtype_size": 2,
            "activation_checkpoint": 0,
        }
    )

    cost_data_path = "data/cost_data.pickle"
    with open(cost_data_path, "rb") as f:
        cost_data = pickle.load(f)

    externl_sim = ExternalRestraint(
        config.world_size, config.global_batch_size, config.sequence_length, config=config, cost_data=cost_data, activation_ckpt=config.activation_checkpoint
    )
    externl_sim.run_loop()


if __name__ == "__main__":
    main()
