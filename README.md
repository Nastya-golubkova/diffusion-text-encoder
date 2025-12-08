# Learning coefficients for embeddings from different layers of a text encoder

The idea of ​​the experiment was inspired by the article [Comprehensive Study of Decoder-only LLMs for Text-to-Image generation](https://arxiv.org/pdf/2506.08210), 
the main idea of which is that it could be helpful not to use the embeddings from the last layer of text encoder, as usual, but to aggregate embeddings from each layer of the text encoder.
The authors trained diffusion model with average of all embeddings and also with average of normalized embeddings. 
Both experiments showed better results than the model which used embedding only from last layer.

This research is about training coefficients to aggregate the embeddings and put it to the frozen unet model.
The dataset was prepaired in the following way:

For each prompt, embeddings were extracted from 24 layers of text encoder and also for each prompt the "teacher" (DeepFloyd/IF-I-XL) model generated images.

All experiments were carried out on a model DeepFloyd/IF-I-M
Since there are 24 embeddings , 24 coefficients were trained in the first experiment.



