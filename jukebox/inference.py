import fire, sys
from jukebox.data.data_processor import DataProcessor
from jukebox.hparams import setup_hparams
from jukebox.train import get_ema
import jukebox.utils.dist_adapter as dist
from make_models import load_checkpoint
from train import evaluate, get_ddp
from logging import init_logging

def run(hps="vqvae", port=29500, **kwargs):
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs
    
    # Setup dataset
    data_processor = DataProcessor(hps)
    
    # Setup model
    vqvae = load_checkpoint(hps.path_to_checkpoint)
    distributed_vqvae = get_ddp(vqvae, hps)
    ema = get_ema(vqvae, hps)
    
    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = vqvae.step
    
    if ema: ema.swap()
    test_metrics = evaluate(distributed_vqvae, vqvae, logger, metrics, data_processor, hps)
    if rank == 0:
        print('Ema',' '.join([f'{key}: {val:0.4f}' for key,val in test_metrics.items()]))
    dist.barrier()
    if ema: ema.swap()
    dist.barrier()

if __name__ == '__main__':
    fire.Fire(run)