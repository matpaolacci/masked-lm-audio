import torch.nn as nn

class AudioEmbedding(nn.Embedding):
    """TODO: The weights freezing part is MISSING!!!!
    """

    def __init__(self, vq_vae_embeddings: nn.Embedding, num_special_embeddings = 3):
        super().__init__(vq_vae_embeddings.num_embeddings + num_special_embeddings, vq_vae_embeddings.embedding_dim)
        vq_vae_num_embed = vq_vae_embeddings.num_embeddings

        # We train only the new special tokens and hold down the gradients of the other tokens
        self.weight.data[:vq_vae_num_embed] = vq_vae_embeddings.weight.data
        self.weight.data.requires_grad = False