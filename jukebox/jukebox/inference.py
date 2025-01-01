import fire, sys, torch as t, os
from jukebox.data.data_processor import DataProcessor
from jukebox.hparams import setup_hparams
from jukebox.train import get_ema
import jukebox.utils.dist_adapter as dist
from make_models import make_vqvae
from train import evaluate, get_ddp
from utils.logger import init_logging
from utils.audio_utils import save_wav_2, audio_preprocess
from jukebox.vqvae.vqvae import VQVAE

def inference(model, hps, data_processor, logger):
    model.eval()
    with t.no_grad():
        for i, x in logger.get_range(data_processor.test_loader):
            x = x.to('cuda', non_blocking=True)
            x_original = audio_preprocess(x, hps)
            
            forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)
            x_recon, loss, _metrics = model(x_original, **forw_kwargs)
            
            save_wav_2(f'{logger.logdir}/batch_{i}', x, hps.sr, is_original=True)
            save_wav_2(f'{logger.logdir}/batch_{i}', x_recon, hps.sr)
                
def inference_on_top_vqvae(model: VQVAE, hps, data_processor, logger):
    model.eval()
    with t.no_grad():
        for i, x in logger.get_range(data_processor.test_loader):
            x = x.to('cuda', non_blocking=True)
            x_original = audio_preprocess(x, hps)
            
            x_l = model.encode(x_original)[2] # [indexes_level_0, indexes_level_1, indexes_level_2]
            x_recon = model.decode(x_l, start_level=2)
            
            assert x_recon.shape == x_original.shape, f"x_recon.shape={x_recon.shape} != x_original.shape={x_original.shape}"
            
            save_wav_2(f'{logger.logdir}/batch_{i}', x, hps.sr, is_original=True)
            save_wav_2(f'{logger.logdir}/batch_{i}', x_recon, hps.sr)

def run(hps="teeny", port=29500, **kwargs):
    '''Do inference over a dataset using a trained model and save the results in the specified directory.
    '''
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    inference_type = kwargs.pop('inference_type') # normal or top
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs
    
    # Setup dataset
    data_processor = DataProcessor(hps)
    
    # Setup model
    vqvae = make_vqvae(hps, device)
    
    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = vqvae.step
    
    if inference_type == 'normal':
        inference(vqvae, hps, data_processor, logger)
    else:
        inference_on_top_vqvae(vqvae, hps, data_processor, logger)


if __name__ == '__main__':
    fire.Fire(run)