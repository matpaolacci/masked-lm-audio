import fire, sys
from jukebox.data.data_processor import DataProcessor
from jukebox.hparams import setup_hparams
from jukebox.train import get_ema
import jukebox.utils.dist_adapter as dist
from make_models import make_vqvae
from train import evaluate, get_ddp
from utils.logger import init_logging
import torch as t
from utils.audio_utils import save_wav

def inference(model, hps, data_processor, logger):
    model.eval()
    with t.no_grad():
        for i, x in logger.get_range(data_processor.test_loader):
            x = x.to('cuda', non_blocking=True)
                
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)
            out_batch, loss, _metrics = model(x, **forw_kwargs)
            
            save_wav(f'{logger.logdir}/batch_{i}', out_batch, hps.sr) # take the batch
                

def run(hps="teeny", port=29500, **kwargs):
    '''Do inference over a dataset using a trained model and save the results in the specified directory.
    '''
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs
    
    # Setup dataset
    data_processor = DataProcessor(hps)
    
    # Setup model
    vqvae = make_vqvae(hps, device, 2)
    
    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = vqvae.step
    
    inference(vqvae, hps, data_processor, logger)


if __name__ == '__main__':
    fire.Fire(run)