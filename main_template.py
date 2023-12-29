"""
main package
"""
import pickle

# from simulator.simulator import Constraint, Simulator
from simulator.noz3_simulator import Constraint
from utils.config import Config


def main():
    """main function"""

    # model_size = ["7B", "13B", "30B", "65B"]
    # seq_length = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

    GLOBA_BSZ = 4096 * 1024
    config = Config(
        {
            "world_size_max": {world_size},
            "world_size_min": {world_size},
            "global_bsz": GLOBA_BSZ,
            "global_bsz_min": GLOBA_BSZ,
            "global_bsz_max": GLOBA_BSZ,
            "sequence_length": {seq_len},
            "model_size": {model_size},
            "vocab_size": 103168,
            "dtype_size": 2,
            "use_fa": 1,
            "fixed_micro_num": 1,
            "fixed_micro_bsz": 1,
            "mem_threshold": 70 * 1024**3,
        }
    )

    # global_bsz (int): Global batch size, use_strict_bsz 为True时会用到这个bsz
    # global_bsz_min (int): global_bsz的搜素上界
    # global_bsz_max (int): global_bsz的搜素下界
    # max_world_size (int): world_size的搜素上界
    # min_world_size (int): world_size的搜素下界
    # seq_len (int):
    # overlap_wdp (int): 是否考虑overlap wdp的通信
    # fixed_micro_num (int): 是否固定micro_num,默认为None不生效
    # fixed_micro_bsz (int): 是否固定micro_bsz ,默认为None不生效
    # use_strict_bsz(bool) : 如果为True，则会严格限制globa bsz为global_bsz参数的值
    # debug (bool): 是否输出额外的debug信息
    # config (dict): 模型的config

    externl_sim = Constraint(
        debug=True,
        overlap_wdp=True,
        use_fixed_micro_bsz=False,
        use_strict_bsz=True,
        config=config,
    )
    externl_sim.run_flexible_worldsize_loop()


if __name__ == "__main__":
    main()
