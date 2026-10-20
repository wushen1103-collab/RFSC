"""Regenerate only missing validation predictions from frozen GINE checkpoints."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

ROOT=Path(os.environ.get('RFSC_WORKSPACE_ROOT',Path(__file__).resolve().parents[1])).resolve()
OUT=Path(os.environ.get('RFSC_AUDIT_OUTPUT',ROOT/'reproduced/structural_audits')).resolve()
sys.path.insert(0,str(ROOT))


def worker(task):
    target,seed,gpu=task
    if hasattr(os,'sched_getaffinity'):
        cpus=sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0,cpus[:max(1,min(32,len(cpus)-30))])
    os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='2'
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.loader import DataLoader
    from bps_gnn.io import read_structure_table,freeze_prediction_file,sha256_file,read_frozen_prediction
    from bps_gnn.temporal import temporal_split_ids
    from bps_gnn.models import GINERegressor
    from bps_gnn.molecular_graphs import ATOM_DIM,BOND_DIM,load_or_build_graphs
    torch.set_num_threads(2)
    dataroot=ROOT/'data/processed/chembl36_temporal_earliest'
    source=ROOT/f'experiments/chembl36_temporal_earliest/models/gine/{target}/temporal/seed_{seed}/checkpoint.pt'
    output=OUT/f'validation_predictions/{target}/seed_{seed}/predictions.parquet'
    if output.exists():
        read_frozen_prediction(output)
        return dict(target=target,seed=seed,status='existing',gpu=gpu)
    structures=read_structure_table(dataroot,target).reset_index(drop=True)
    molids=structures.mol_id.tolist();lookup={m:i for i,m in enumerate(molids)}
    _,valid,test=temporal_split_ids(dataroot,target)
    assert not set(valid)&set(test)
    ck=torch.load(source,map_location='cpu',weights_only=False)
    config=ck['config'];mc=dict(config['model']);architecture=mc.pop('architecture','gine')
    assert architecture=='gine'
    model=GINERegressor(ATOM_DIM,BOND_DIM,**mc)
    model.load_state_dict(ck['model_state'])
    model.to('cuda').eval()
    cache=ROOT/Path(config['cache_root'])/f'{target}.pt'
    graphs=load_or_build_graphs(structures.canonical_isomeric_smiles.tolist(),cache)
    vgraphs=[copy.copy(graphs[lookup[m]]) for m in valid]
    loader=DataLoader(vgraphs,batch_size=int(config['optimization']['batch_size']),shuffle=False)
    indices=[];pred=[]
    with torch.no_grad():
        for batch in loader:
            batch=batch.to('cuda')
            values=model(batch).float().cpu().numpy()*float(ck['scale'])+float(ck['center'])
            indices.extend(batch.row_id.cpu().numpy().tolist());pred.extend(np.asarray(values).reshape(-1).tolist())
    ids=[molids[i] for i in indices]
    assert len(ids)==len(valid) and set(ids)==set(valid) and np.isfinite(pred).all()
    digest=freeze_prediction_file(pd.DataFrame({'mol_id':ids,'prediction':pred}),output,
        {'target':target,'seed':seed,'partition':'validation','source_checkpoint':str(source.relative_to(ROOT)),
         'source_checkpoint_sha256':sha256_file(source),'validation_labels_materialized':False,
         'test_labels_materialized':False,'model_retrained':False,'gpu':gpu})
    return dict(target=target,seed=seed,status='completed',gpu=gpu,n_validation=len(ids),prediction_sha256=digest)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--gpus',nargs='+',default=None)
    a=ap.parse_args();os.chdir(ROOT)
    available={int(v.split(',')[0]):int(v.split(',')[1].strip()) for v in subprocess.check_output(
        ['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()}
    requested=a.gpus if a.gpus is not None else [str(g) for g in sorted(available)]
    gpus=[g for g in requested if available[int(g)]<500]
    if not gpus:raise RuntimeError('No requested GPU is idle; inference not launched.')
    import torch
    torch.manual_seed(290820)
    device=f'cuda:{gpus[0]}'
    x=torch.randn(8,8,device=device)
    witness=x@x.T
    assert witness.shape==(8,8) and torch.isfinite(witness).all()
    print('SEEDED_GPU_WITNESS',str(witness.shape),torch.cuda.get_device_name(int(gpus[0])),flush=True)
    targets=sorted(p.name for p in (ROOT/'experiments/chembl36_temporal_earliest/predictions').iterdir() if p.is_dir())
    tasks=[(t,s,gpus[(i*3+s)%len(gpus)]) for i,t in enumerate(targets) for s in [0,1,2]]
    if a.smoke:tasks=tasks[:1]
    rows=[]
    # Separate pool per device keeps CUDA assignment fixed across reused workers.
    pools=[ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn')) for g in gpus]
    try:
        futures=[pools[gpus.index(task[2])].submit(worker,task) for task in tasks]
        for future in as_completed(futures):
            result=future.result();rows.append(result);print(json.dumps(result),flush=True)
    finally:
        for pool in pools:pool.shutdown()
    import pandas as pd
    output=OUT/'validation_predictions'
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/('smoke_summary.csv' if a.smoke else 'inference_summary.csv'),index=False)
    assert len(rows)==len(tasks)
    print('VALIDATION_DONE',len(rows),flush=True)


if __name__=='__main__':main()
