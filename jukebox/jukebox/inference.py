import fire, sys, torch as t, os
from jukebox.data.data_processor import DataProcessor
from jukebox.hparams import Hyperparams, setup_hparams
import jukebox.utils.dist_adapter as dist
from make_models import make_vqvae
from utils.logger import init_logging
from utils.audio_utils import save_wav_2, audio_preprocess, save_embeddings, load_batches_of_embeddings
from jukebox.vqvae.vqvae import VQVAE
from jukebox.utils.dist_utils import print_once

# def inference(model, hps, data_processor, logger):
#     model.eval()
#     with t.no_grad():
#         for i, x in logger.get_range(data_processor.test_loader):
#             x = x.to('cuda', non_blocking=True)
#             x_original = audio_preprocess(x, hps)
            
#             forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)
#             x_recon, loss, _metrics = model(x_original, **forw_kwargs)
            
#             save_wav_2(f'{logger.logdir}/batch_{i}', x, hps.sr, is_original=True)
#             save_wav_2(f'{logger.logdir}/batch_{i}', x_recon, hps.sr)

def inference_on_level(model: VQVAE, hps: Hyperparams, data_processor: DataProcessor, logger):
    model.eval()
    with t.no_grad():
        for i, x in logger.get_range(data_processor.test_loader):
            x = x.to('cuda', non_blocking=True)
            x_original = audio_preprocess(x, hps)
            
            # [indexes_level_0, indexes_level_1, indexes_level_2]
            #   indexes_level_i.shape = (bs, encoded_sequence_length)
            x_l = model.encode(x_original, bs_chunks=hps.bs)
            x_recon = model.decode(x_l[hps.use_level:], start_level=hps.use_level, bs_chunks=hps.bs)
            
            assert x_recon.shape == x_original.shape, f"x_recon.shape={x_recon.shape} != x_original.shape={x_original.shape}"
            
            save_wav_2(f'{logger.logdir}/batch_{i}', x, hps.sr, is_original=True)
            save_wav_2(f'{logger.logdir}/batch_{i}', x_recon, hps.sr)


def encode_and_save(model: VQVAE, hps: Hyperparams, data_processor: DataProcessor, logger):
    os.makedirs(f'{logger.logdir}/encoded_data', exist_ok=True)
    model.eval()
    with t.no_grad():
        track_idx = data_processor.dataset.get_song_index(hps.track_name)
        track_data = t.tensor([])
        for i, x in logger.get_range(data_processor.test_loader):
            x = x.to('cuda', non_blocking=True)
            x_original = audio_preprocess(x, hps)
            
            # [indexes_level_0, indexes_level_1, indexes_level_2]
            #   indexes_level_i.shape = (bs, encoded_sequence_length)
            x_l = model.encode(x_original, bs_chunks=hps.bs)
            if track_idx != data_processor.dataset.get_song_index(i):
                save_embeddings(track_data, f'{logger.logdir}/encoded_data/track_{track_idx}')
                track_idx = data_processor.dataset.get_song_index(i)
                track_data = t.tensor([])
            track_data = t.cat((track_data, x_l[hps.use_level]), dim=0)
            if i == len(data_processor.test_loader) - 1:
                save_embeddings(track_data, f'{logger.logdir}/encoded_data/track_{track_idx}')


def decode_and_save(model: VQVAE, hps, logger):
    os.makedirs(f'{logger.logdir}/decoded_data', exist_ok=True)
    
    data: t.Tensor = load_batches_of_embeddings(hps.path_to_encoded_data)
    model.eval()
    with t.no_grad():
        for i, batch in enumerate(data):
            x_recon = model.decode(batch.unsqueeze(0), start_level=hps.use_level, bs_chunks=hps.bs)
            save_wav_2(f'{logger.logdir}/decoded_data/batch_{i}', x_recon, hps.sr)
    
            
def run(hps="teeny", port=29500, **kwargs):
    '''Do inference over a dataset using a trained model and save the results in the specified directory.
    '''
    from jukebox.utils.dist_utils import setup_dist_from_mpi
    rank, local_rank, device = setup_dist_from_mpi(port=port)
    
    hps = setup_hparams(hps, kwargs)
    hps.ngpus = dist.get_world_size()
    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs
    hps.inference = True
    
    # Setup model
    vqvae = make_vqvae(hps, device)
    
    logger, metrics = init_logging(hps, local_rank, rank)
    logger.iters = vqvae.step
    hps.path_to_encoded_data = f'{logger.logdir}/encoded_data'

    if hps.operation_type == "inference":
        # Setup dataset
        data_processor = DataProcessor(hps)
        inference_on_level(vqvae, hps, data_processor, logger)
    elif hps.operation_type == "encode":
        # Setup dataset
        data_processor = DataProcessor(hps)
        encode_and_save(vqvae, hps, data_processor, logger)
    elif hps.operation_type == "decode":
        decode_and_save(vqvae, hps, logger)
    else:
        raise ValueError(f"operation_type={hps.operation_type} not supported")


if __name__ == '__main__':
    fire.Fire(run)