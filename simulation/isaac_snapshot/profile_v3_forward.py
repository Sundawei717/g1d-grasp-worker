import argparse
from pathlib import Path
from time import perf_counter
import gc

import numpy as np
import torch

from chunk_model import SharedResNet18ChunkTransformer
from three_view_detector import write_json


def measure(weights, device, repetitions):
    result = {}
    for architecture in ("legacy_three_view", "v3_final"):
        v3 = architecture=="v3_final"
        for variant, right_dim in (("A",3),("B",4)):
            torch.manual_seed(11419)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline=torch.cuda.memory_allocated(device)
            model=SharedResNet18ChunkTransformer(weights_path=weights,wrist_channels=right_dim,
                                                 v3_final=v3).eval().to(device)
            imgs=[torch.rand(1,c,224,224,device=device) for c in (5,3 if v3 else right_dim,right_dim)]
            inputs=(imgs[0],imgs[1],torch.rand(1,2,17 if v3 else 8,device=device),
                    torch.rand(1,6,device=device),imgs[2])
            times=[]
            with torch.inference_mode():
                for _ in range(10): model(*inputs)
                torch.cuda.synchronize(device)
                for _ in range(repetitions):
                    start=perf_counter()
                    output=model(*inputs)
                    torch.cuda.synchronize(device)
                    times.append(1000*(perf_counter()-start))
                assert output.shape==(1,8,8) and torch.isfinite(output).all().item()
            stats={"wall_forward_ms_p50_p95":np.percentile(times,[50,95]).tolist(),
                   "peak_allocated_mib":(torch.cuda.max_memory_allocated(device)-baseline)/2**20,
                   "peak_reserved_mib":torch.cuda.max_memory_reserved(device)/2**20,
                   "parameters":sum(p.numel() for p in model.parameters()),"visual_tokens":588 if v3 else 147}
            result[f"{architecture}_{variant}"]=stats
            print(architecture,variant,stats,flush=True)
            del imgs, inputs, output, model
            gc.collect()
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("weights",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--repetitions",type=int,default=60)
    args=parser.parse_args()
    device=torch.device("cuda:0")
    torch.set_num_threads(2)
    write_json(args.output,{"passed":True,"seed":11419,"batch":1,"dtype":"fp32",
                            "device":torch.cuda.get_device_name(device),"repetitions":args.repetitions,
                            "results":measure(args.weights,device,args.repetitions)})
