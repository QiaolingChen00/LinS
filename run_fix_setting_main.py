import pickle

import pandas as pd

# from simulator.simulator import Constraint, Simulator
from simulator.noz3_simulator import Constraint
from utils.config import Config


if __name__ == "__main__":
    GLOBA_BSZ = 512 * 1024
    model_sizes = [7, 13, 30, 65]
    results = []
    columns = []
    for model_size in model_sizes:
        if model_size == 7:
            sp = 4
            pp = 1
            world_size = 128
        elif model_size == 13:
            sp = 8
            pp = 1
            world_size = 256
        elif model_size == 30:
            sp = 8
            pp = 2
            world_size = 512
        elif model_size == 65:
            sp = 8
            pp = 4
            world_size = 1024

        config = Config(
            {
                "world_size_max": world_size,
                "world_size_min": world_size,
                "global_bsz": GLOBA_BSZ,
                "global_bsz_min": GLOBA_BSZ,
                "global_bsz_max": GLOBA_BSZ,
                "sequence_length": 256 * 1024,
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
                wp = world_size // pp
            elif algo_type == "fsp":
                algo_name = "Megatron-3D"
                wp = 1
            elif algo_type == "msp":
                algo_name = "Megatron-sp"
                wp = 1
                continue

            name = f"{model_size}B-{algo_name}"
            solu = externl_sim.cal_cost(
                pp=pp,
                sp=sp,
                wp=wp,
                zp=1,
                micro_bsz=1,
                micro_num=1,
                algo_type=algo_type,
                world_size=world_size,
                activation_ckpt=False,

            )
            if solu is None:
                print(f"Warning model_size: {model_size}, algo_name: {algo_name}  Not found solu!", flush=True)
                results.append(
                    [
                        name,
                        0,0,0,0
                    ]
                )
            else:
                print(f"model_size: {model_size}, algo_name: {algo_name}  found solu:{solu}!", flush=True)
                comp = solu.comp_attn + solu.comp_wp
                comm = solu.wp_comm_cost + solu.sp_comm_cost + solu.pp_comm_cost + solu.wdp_comm_cost + solu.zp_comm_cost
                results.append(
                    [
                        name,
                        round(solu.activation / 1024**3, 2),
                        round(solu.p_g_mm_cost / 1024**3, 2),
                        round(comp, 2),
                        round(comm, 2),
                    ]
                )

    columns = ["algo", "act mem(GB)", "p mem(GB)", "comp(s)", "comm(s)"]
    df = pd.DataFrame(results, columns=columns)
    print(df)
    df.to_csv("excel_output.xls")