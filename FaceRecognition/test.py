from dataclasses import dataclass
from typing import Any
from pathlib import Path
import argparse
from datetime import datetime

import torch
import torch.utils.data.distributed


from config.config import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX, FaceDatasetFolder
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50
from eval import verification


@dataclass
class Benchmark:
    name : str
    data : list
    best_checkpoint : Any = None
    best_acc        : Any = 0.0
    best_acc_std    : Any = 0.0


def load_benchmark(name,
                   benchmark_path,
                   image_size
):
    
    bench_path = Path(benchmark_path).resolve() / f"{name}.bin"
    data = verification.load_bin(
                              str(bench_path),
                              image_size
                        )
    return Benchmark(name, data) # type: ignore
    
  
def run_tests(
        model_path : str,
        benchmark_path : str,
        output_path : str,
        best_models : str,
        image_size=(112, 112),
        batch_size = 128
):
    
    print("Testing models at", model_path)
    device = torch.device("cuda:0")
    
    # Loading benchmarks
    lfw = load_benchmark("lfw", benchmark_path, image_size)
    agedb_30 = load_benchmark("agedb_30", benchmark_path, image_size)
    calfw = load_benchmark("calfw", benchmark_path, image_size)
    cfp_ff = load_benchmark("cfp_ff", benchmark_path, image_size)
    cfp_fp = load_benchmark("cfp_fp", benchmark_path, image_size)
    cplfw = load_benchmark("cplfw", benchmark_path, image_size)
    
    benchmarks = [lfw, agedb_30, calfw, cfp_ff, cfp_fp, cplfw]
    
    checkpoints = list(Path(model_path).rglob("*backbone.pth"))
    
    for idx, checkpoint in enumerate(checkpoints):
        
        print(f"Running: {idx+1}/{len(checkpoints)} checkpoints. Checkpoint is {checkpoint}")
        
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(device)
        backbone.load_state_dict(torch.load(str(checkpoint.resolve()), map_location=device))
        backbone.eval()
        
        print(f"=================================\n Checkpoint: {checkpoint.name}")
        for benchmark in benchmarks:
        
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                benchmark.data, backbone, batch_size, device, nfolds=10)
            
            print(f"{benchmark.name}: {acc2*100:.2f}%+-{std2*100:.2f}%")
            
            if acc2 > benchmark.best_acc:
                benchmark.best_checkpoint = checkpoint.name
                benchmark.best_acc = acc2
                benchmark.best_acc_std = std2
                
                print(f"[{benchmark.name}] New best Acc: {acc2} Checkpoint: {checkpoint.name}")
                
            del acc1, std1, acc2, std2, xnorm, embeddings_list
            
        del backbone
    
    print("---- Final Resuls ----")

    with open(f"{best_models}", "w") as f:
        f.write(f"Models: {model_path}\n")
        f.write("[Benchmark]\t Acc Mean \t Acc Std \t Checkpoint\n")
        for bm in benchmarks:
            line = f"[{bm.name}]\t {bm.best_acc} \t {bm.best_acc_std} \t {bm.best_checkpoint}\n"
            print(line)
            f.write(line)


    with open(f"{output_path}", "w") as f:
        
        line_header  = [f"{'Model Name':.^40}"]
        line_results = [f"{model_path:.^40}"]
        
        for bm in benchmarks:
            line_header.append(f"{bm.name:^14}")
            result = f"{bm.best_acc*100:.2f} ± {bm.best_acc_std*100:.2f}"
            line_results.append(f"{result}:^14")
            
        line_header = "|".join(line_header) + "\n"
        line_results = "|".join(line_results) + "\n"
        print("=== Writing Results ===")
        print(line_header)
        print(line_results)
        f.write(line_header)
        f.write(line_results)
            
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Test')
    parser.add_argument('--models', type=str, help='Models')
    parser.add_argument('--benchmarks', type=str, help='Benchmarks')
    parser.add_argument('--output', type=str, help='Output file')
    parser.add_argument('--best', type=str, help='Output file for best models')
    args_ = parser.parse_args()
    
    run_tests(
              model_path=args_.models,
              benchmark_path=args_.benchmarks,
              output_path=args_.output,
              best_models=args_.best
             )