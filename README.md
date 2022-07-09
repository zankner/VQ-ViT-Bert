# VQ-Vi-Bert
🏞🤖 Masked token prediction for bidirectional vision transformers based on vector quantized latent representations


## Status
Work I did trying to train vision transformers using the codebook from a vq-vae as input. The idea was that we could easily do MLM with transformers if the inputs tokens were discretized. I couldn't get it to work but there was work from msft which did get the method to work (https://www.arxiv-vanity.com/papers/2106.08254/). Wave2Vec also uses discrete tokens in this manner but for different task and different domain.
