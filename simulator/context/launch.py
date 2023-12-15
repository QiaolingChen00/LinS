
from simulator.context import global_context as gpc
from simulator.context import ParallelMode

def check_and_modify_parallel_config(config_dict):
    # process the parallel config
    if "sequence_parallel" not in gpc.config.parallel:
        gpc.config.parallel._add_item("sequence_parallel", False)
    else:
        assert not (
            gpc.config.parallel.sequence_parallel is True and gpc.config.model.use_flash_attn is False
        ), "sequence parallel does not support use_flash_attn=False"

    if isinstance(gpc.config.parallel["tensor"], int):
        gpc.config.parallel["tensor"] = dict(size=gpc.config.parallel["tensor"], sp="none", intern_overlap=False)
    if gpc.config.parallel["tensor"].get("sp", None) is None:
        gpc.config.parallel["tensor"]["sp"] = "none"
    if gpc.config.parallel["tensor"].get("intern_overlap", None) is None:
        gpc.config.parallel["tensor"]["intern_overlap"] = False
    assert gpc.config.parallel["tensor"].get("sp", None) in [
        "none",
        "megatron",
        "flash-attn",
        "intern",
    ], "invalid sp mode, only ['none', 'megatron', 'flash-attn', 'intern'] is supported"
    # adapt to old version's sequence parallel config
    if gpc.config.parallel["tensor"].get("sp", None) in ["megatron", "flash-attn", "intern"]:
        gpc.config.parallel.sequence_parallel = True

    # # currently only interleaved pipeline scheduler with overlap can guarantee loss accuracy
    # if hasattr(gpc.config.model, "num_chunks") and gpc.config.model.num_chunks > 1:
    #     assert (
    #         gpc.config.parallel["pipeline"].get("interleaved_overlap", False) is True
    #     ), "only support interleaved pipeline scheduler with overlap"



