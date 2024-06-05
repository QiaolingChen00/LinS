# 运行
sensecore 910B环境（初始的fa+linear的profiling文件已经有了，通信还暂未抓性能数据，所以目前搜出来的并行策略会有些不准）
```python
source /ailab_internlm_data/llm_env/envs/0506/deeplink-20240506.sh
python main.py > log.log 2>&1 # 现在开启了debug日志，输出会有些多，可以tail -n 200看下最终的搜索结果
```
<ul>
<li> global_bsz (int): 这个参数use_strict_bsz 为True时会用到，否则就根据[global_bsz_min, global_bsz_max]搜索最佳的并行配置  </li>
<li> global_bsz_min (int): global_bsz的搜素上界  </li>
<li> global_bsz_max (int): global_bsz的搜素下界  </li>
<li> max_world_size (int): world_size的搜素上界  </li>
<li> min_world_size (int): world_size的搜素下界  </li>
<li> seq_len (int): 序列长度 </li>
<li> overlap_wdp (int): 是否考虑overlap wdp的通信  </li>
<li> fixed_micro_num (int): 是否固定micro_num,默认为None不生效  </li>
<li> fixed_micro_bsz (int): 是否固定micro_bsz ,默认为None不生效  </li>
<li> use_strict_bsz(bool) : 如果为True，则会严格限制globa bsz为global_bsz参数的值  </li>
<li> debug (bool): 是否输出额外的debug信息  </li>
<li> config (dict): 模型的config  </li>
</ul>



# Backup

- 背景：sp很重要，
- 现有方法存在的问题
  - **现有方法只支持有限的并行维度**，因此要么支持的sequence length不够长（MSP、FSP），要么在减少了单个device所需的内存的同时引入大量通信（ulysses），训练效率不够高。\todo, MSP 和 MTP 的 pattern.
  - **现有的方法需要手动调节**不同并行维度的划分方式（sp、tp等）。一方面，手动方法对用户有较高的要求：用户需要根据模型和集群环境手动探索并行方案，具有较大的开发成本，且需要用户有专业的系统知识。另一方面，手动方法不能保证手动选择的配置是最优的。
- 目标：我们希望能有一个**自动**进行**多维度**并行配置的训练框架，支持**不同长度的sequence length**，并在不同的sequence length下都有**较快**的训练速度
- 挑战
  - 支持很长的sequence需要同时切分sequence和p、g、os多个维度，引入大量的通信开销。（虽然没有ulysses的通信开销大，但通信开销还是不能忽略）
  - 很难直接估计不同并行方案的step time。现有的对模型训练step time的建模没有考虑sp这个维度，不适用于long sequence的场景。
- 设计
  - 为了减小通信开销，我们分析了transformer-based模型的计算和通信依赖，然后设计了selective overlap的方法，针对forward、backward的特点分别设计不同的overlap方式。
  - 为了找到效率最高的并行方式，我们对计算和通信进行了建模，模拟训练的时间开销。对模拟出来（切分sequence、p、g、os的）不同并行方式的step time进行自动化搜索，找到高效的并行方式，用户无需依赖经验并试错。因为transformer-based的模型结构较为规律，因此搜索的代价不是很大。
- 效果：我们提出一种面向sp的半自动化并行框架isp，搜索sp、p、g、os维度，在支持long sequence length的同时有较快的训练速度，无论模型大小、sequence长短我们都能搜索到一个不会oom且在训练速度上好的并行方案。


我们研究了现有的四种提供序列并行的算法 MTP，MSP，FSP，DSP。


+ SP: semi-Auto paralleism search. All SP

    + alo selection (A) For i in range(MTP，MSP，FSP，DSP):
    + linear (X)

+ SP: LINS + Ulysses SP + PP.


## ISP 建模：

### Search Space Analysis

#### Two-GPU Example

pattern of Using MTP, MSP, FSP


#### Comm model

1. 通信扩展因子 todo？
    1. **扩展因子的定义**：
    - 扩展因子是用来衡量分布式训练系统通信性能的一个比例。
    - 它由单个训练工作器（worker）执行预设数据量的通信任务所需的执行时间 \( T_1 \) 与系统在有 \( N \) 个工作器处理相同训练任务时的执行时间 \( T_N \) 之比来定义。

    2. **扩展因子的计算**：
    - 扩展因子的计算公式为：扩展因子 = \( \frac{T_1}{T_N \times N} \)。
    - 例如，如果一个给定的训练任务，1个工作器需要9小时完成，而8个工作器只需1.25小时完成，则8个工作器的系统扩展因子为 \( \frac{9}{1.25 \times 8} = 0.9 \)。


2. Comm. & Comp. Modeling

    1. Comm.

   - 旧算法使用环形方法，其中每个进程的数据被发送到一个虚拟的进程环中。
   - 在第一步中，每个进程 \( i \) 向进程 \( i+1 \) 发送数据并从进程 \( i-1 \) 接收数据（有环绕）。
   - 从第二步开始，每个进程 \( i \) 向进程 \( i+1 \) 转发它从进程 \( i-1 \) 在上一步中接收到的数据。
   - 如果 \( p \) 是进程的数量，整个算法需要 \( p-1 \) 步。如果 \( n \) 是每个进程要收集的总数据量，则在每一步中每个进程发送和接收 \( n/p \) 数据。
   - \( T_{\text{ring}} = (p - 1)\alpha + \frac{(p-1)n\beta}{p} \)。

   - \( T_{\text{tree}} = \log(p)\alpha + \frac{(p-1)n\beta}{p} \)


##### Model states Comm. modeling.
$P_{para}$,$P_{grads}$,$P_{os}$: we use $P_{x}$ define the number of GPUs partitioning the member x
n: number of GPU in world size
M: number of parameters in model
Dt: datatype of parameters/gradient/optimizer states stored
BW: bandwidth, when $P_{x}>8$, BW=IB network,else BW=Nvlink
Gn: number of gradient accumulation steps. It equal to \(Gn=\frac{GlobalBatchSize}{n\times MicroBatchSize}\)

1. \( T_{\text{Comm}}(P_{para},ISP,allgather) = 2 [(P_{para}- 1)\alpha + \frac{(P_{para}-1)MDt}{P_{para} BW}\beta]\)。


2. \( T_{\text{Comm}}(P_{grad},ISP,reducescatter) = Gn [(P_{grad}- 1)\alpha + \frac{(P_{grad}-1)MDt}{P_{grad} BW}\beta]\)。


3. \( T_{\text{Comm}}(P_{grad},ISP,broadcast) = 3 [(P_{os}- 1)\alpha + \frac{(P_{os}-1)MDt}{P_{os} BW}\beta]\)。

##### All2All Comm. modeling.
all2all communication occur before and after the computation of attention related to sequence parallism. b,s,h present MicroBatchSize,SquenceLength and HiddenSize repsectively.

all2all is a point to point communication, inter and intra node communication are independent. Therefore, when \(SP>8\) ,we should model the inter and intra node communication seperately corelated to the specific communication traffic.

IF SP<8:
\( T_{\text{Comm}}(SP,ISP,all2all) = (SP- 1)\alpha + \frac{4(SP-1)bshDt}{SP\times BW}\beta\)。

IF SP>8: TODO


##### Model states Comp. modeling.
1. Compute attention block:
    Calculate Q, K, V: \(3 \times [b,s,h] \times [h,h] = 6bsh^2\)
    QK^T matrix multiplication: \([b,a,s,h] \times [b,a,h] = 2bas^2\)
    Score dot V: \([b,a,s,h] \times [b,a,h] = 2bas^2\)
    Post attention: \([b,s,h] \times [h,h] = 2bsh^2\)

2. Compute mlp block:
    First linear layer: \(\times [b,s,h] \times [h,4h] = 8bsh^2\)
    Second linear layer: \(\times [b,s,4h] \times [4h,h] = 8bsh^2\)

\( T_{\text{Comp}}(SP,b,C_{qkv},Gemm) = Gemm(6bh^2 \times \frac{s}{SP})\)。
\( T_{\text{Comp}}(SP,b,C_{qkT+ScoreV},FlashAttn) = FlashAttn(4bh^2 \times \frac{s}{SP})\)。
\( T_{\text{Comp}}(SP,b,C_{PostAttn},Gemm) = Gemm(2bh^2 \times \frac{s}{SP})\)。
\( T_{\text{Comp}}(SP,b,C_{L1},Gemm) = Gemm(8bh^2 \times \frac{s}{SP})\)。
\( T_{\text{Comp}}(SP,b,C_{L2},Gemm) = Gemm(8bh^2 \times \frac{s}{SP})\)。


##### Overlap Modeling
[T_{\text{Overlap}}=\sum(\max(T_\text{Comm,i}),T_\text{Comp,i})\]


# TODO  12.7

+ Backward Comp.
+ PP
+ activation Checkpoint
