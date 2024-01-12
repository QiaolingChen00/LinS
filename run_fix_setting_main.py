import pickle

import pandas as pd

# from simulator.simulator import Constraint, Simulator
from simulator.noz3_simulator import Constraint
from utils.config import Config

if __name__ == "__main__":
    GLOBA_BSZ = 512 * 1024
    model_sizes = [7]
    results = []
    columns = []
    world_size = 1024
    for model_size in model_sizes:
        config = Config(
            {
                "world_size_max": world_size,
                "world_size_min": world_size,
                "global_bsz": GLOBA_BSZ,
                "global_bsz_min": GLOBA_BSZ,
                "global_bsz_max": GLOBA_BSZ,
                "sequence_length": 2 * 1024,
                "model_size": model_size,
                "vocab_size": 103168,
                "dtype_size": 2,
                "use_fa": 1,
                "fixed_micro_num": 1,
                "fixed_micro_bsz": 1,
                "mem_threshold": 999999 * 1024**3,
                "wp_penalty_coefficient": 0.2,
            }
        )

        externl_sim = Constraint(
            debug=True,
            overlap_wdp=True,
            use_fixed_micro_bsz=True,
            use_strict_bsz=False,
            config=config,
        )

        for algo_type in externl_sim._algo_list:
            if algo_type == "isp":
                algo_name = "DSP"
                pp = 1
                sp = 1
                wp = 1
                zp = 128
            elif algo_type == "fsp":
                algo_name = "Megatron-3D"
                pp = 1
                sp = 2
                wp = 1
                zp = 64
            elif algo_type == "msp":
                algo_name = "Megatron-sp"
                pp = 1
                sp = 2
                wp = 1
                zp = 64

            name = f"{model_size}B-{algo_name}"
            solu = externl_sim.cal_cost(
                pp=pp,
                sp=sp,
                wp=wp,
                zp=zp,
                micro_bsz=4,
                micro_num=1,
                algo_type=algo_type,
                world_size=world_size,
                activation_ckpt=False,
            )
            assert solu is not None
            if solu is None:
                print(f"Warning model_size: {model_size}, algo_name: {algo_name}  Not found solu!", flush=True)
                results.append([name, 0, 0, 0, 0, 0])
            else:
                print(f"model_size: {model_size}, algo_name: {algo_name}  found solu:{solu}!", flush=True)
                comp = solu.comp_attn + solu.comp_wp
                comm = (
                    solu.wp_comm_cost + solu.sp_comm_cost + solu.pp_comm_cost + solu.wdp_comm_cost + solu.zp_comm_cost
                )
                results.append(
                    [
                        name,
                        round(solu.activation / 1024**3, 5),
                        round(solu.p_g_mm_cost / 1024**3, 5),
                        round(solu.os_mm_cost / 1024**3, 5),
                        round(comp, 5),
                        round(comm, 5),
                    ]
                )

    columns = ["algo", "act mem(GB)", "p mem(GB)", "os mem(GB)", "comp(s)", "comm(s)"]
    df = pd.DataFrame(results, columns=columns)
    print(df)
    df.to_csv("excel_output.xls")
