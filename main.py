"""
main package
"""
import pickle

# from simulator.simulator import Constraint, Simulator
from simulator.noz3_simulator import Constraint
from utils.config import Config


def main():
    """main function"""
    # Solution:  pp: 1 sp: 1 micro_bsz: 1 micro_num: 16 algo_type: intern, wp_size: 8, zp_size: 8 total fwd_bwd_cost: 1764.00 ms,
    # pp_comm_cost: 0.00 ms, zp_comm_cost: 325.90 ms, wp_comm_cost: 5.40 ms, sp_comm_cost: 0.00 self.comp_wp: 26.70 ms,
    # self.comp_attn: 2.70 ms total mem_cost: 77.10 GB, activation: 39.84 GB, zp_mm_cost: 27.94 GB, wp_mm_cost: 9.31 GB
    config = Config(
        {
            "world_size": 64,
            "global_batch_size": 4 * (1024**2),
            "sequence_length": 4 * 1024,
            "model_size": 20,
            "vocab_size": 103168,
            "dtype_size": 2,
            "use_fa": 1,
        }
    )

    # Solution:  pp: 1 sp: 4 micro_bsz: 1 micro_num: 2 algo_type: intern, wp_size: 8, zp_size: 8 total fwd_bwd_cost: 3936.00 ms,
    # pp_comm_cost: 0.00 ms, zp_comm_cost: 325.90 ms, wp_comm_cost: 5.40 ms, sp_comm_cost: 0.80 self.comp_wp: 26.70 ms,
    # self.comp_attn: 38.10 ms total mem_cost: 77.10 GB, activation: 39.84 GB, zp_mm_cost: 27.94 GB, wp_mm_cost: 9.31 GB
    config = Config(
        {
            "world_size": 512,
            "global_batch_size": 4 * (1024**2),
            "sequence_length": 16 * 1024,
            "model_size": 20,
            "vocab_size": 103168,
            "dtype_size": 2,
            "use_fa": 1,
        }
    )

    # Solution:  pp: 1 sp: 2 micro_bsz: 2 micro_num: 2 algo_type: intern, wp_size: 2, zp_size: 2 total fwd_bwd_cost: 595.20 ms,
    # pp_comm_cost: 0.00 ms, zp_comm_cost: 32.50 ms, wp_comm_cost: 1.80 ms, sp_comm_cost: 0.00 self.comp_wp: 16.50 ms,
    # self.comp_attn: 2.10 ms total mem_cost: 69.15 GB, activation: 17.00 GB, zp_mm_cost: 39.12 GB, wp_mm_cost: 13.04 GB
    config = Config(
        {
            "world_size": 512,
            "global_batch_size": 4 * (1024**2),
            "sequence_length": 4 * 1024,
            "model_size": 7,
            "vocab_size": 103168,
            "dtype_size": 2,
            "use_fa": 1,
        }
    )

    # Solution:  pp: 1 sp: 2 micro_bsz: 1 micro_num: 1 algo_type: intern, wp_size: 2, zp_size: 4 total fwd_bwd_cost: 2051.20 ms,
    # pp_comm_cost: 0.00 ms, zp_comm_cost: 48.80 ms, wp_comm_cost: 1.80 ms, sp_comm_cost: 0.80 self.comp_wp: 32.70 ms,
    # self.comp_attn: 30.60 ms total mem_cost: 66.60 GB, activation: 34.00 GB, zp_mm_cost: 19.56 GB, wp_mm_cost: 13.04 GB
    config = Config(
        {
            "world_size": 512,
            "global_batch_size": 4 * (1024**2),
            "sequence_length": 32 * 1024,
            "model_size": 7,
            "vocab_size": 103168,
            "dtype_size": 2,
            "use_fa": 1,
        }
    )

    GLOBA_BSZ = 256 * 1024 * 16
    config = Config(
        {
            "world_size_max": 128,
            "world_size_min": 128,
            "global_bsz": GLOBA_BSZ,
            "global_bsz_min": GLOBA_BSZ,
            "global_bsz_max": GLOBA_BSZ,
            "sequence_length": 256 * 1024,
            "model_size": 65,
            "vocab_size": 103168,
            "dtype_size": 2,
            "use_fa": 1,
            "fixed_micro_num": 1,
            "fixed_micro_bsz": 1,
            "mem_threshold": 74 * 1024**3,
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
