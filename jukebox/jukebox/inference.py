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
        curr_song_data = t.tensor([], dtype=t.long).cpu()
        curr_song_idx = 0
        dataset = data_processor.dataset
        for num_batch, batch in logger.get_range(data_processor.test_loader):
            x, song_idx = batch['data'], batch['song_index']
            x_original = audio_preprocess(x, hps).cuda()
            
            # [indexes_level_0, indexes_level_1, indexes_level_2]
            #   indexes_level_i.shape = (bs, encoded_sequence_length)
            x_l = model.encode(x_original, bs_chunks=hps.bs)
            x_l = x_l[hps.use_level].cpu()
            
            for num_sample, sample_data in enumerate(x_l):
                if curr_song_idx != song_idx[num_sample]:
                    song_name = os.path.basename(dataset.files[curr_song_idx])
                    save_embeddings(curr_song_data, f'{logger.logdir}/encoded_data/{song_name}')
                    curr_song_idx = song_idx[num_sample]
                    curr_song_data = t.tensor([], dtype=t.long).cpu()
                elif num_batch == len(data_processor.test_loader) - 1 and \
                    num_sample == len(hps.bs) - 1:
                    song_name = os.path.basename(dataset.files[curr_song_idx])
                    save_embeddings(curr_song_data, f'{logger.logdir}/encoded_data/{song_name}')
                else:
                    curr_song_data = t.cat((curr_song_data, sample_data), dim=0)

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