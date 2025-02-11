# Masked Language Model for Audio
In this work, the VQ-VAE model is adapted for audio signals and BERT is used to reconstruct masked audio sequences by treating discrete indices as tokens for NLP processing.
Read the [report](Report.pdf) to learn more.

## Some informations about the notebooks
- My implementation of the VQ-VAE is in the notebook [vqvae.ipynb](vqvae.ipynb) that contains the reconstructed audio.
- In [jukebox_vqvae.ipynb](jukebox_vqvae.ipynb) there is generated audio from BERT, with the audio encoding from the Jukebox VQVAE using the 2nd level, since downs_t=(3, 3, 2), the audio is compressed by 64x.
