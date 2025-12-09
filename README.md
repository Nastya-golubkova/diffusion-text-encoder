# Learning coefficients for embeddings from different layers of a text encoder

The idea of ​​the experiment was inspired by the article [Comprehensive Study of Decoder-only LLMs for Text-to-Image generation](https://arxiv.org/pdf/2506.08210), 
the main idea of which is that it could be helpful not to use the embeddings from the last layer of text encoder, as usual, but to aggregate embeddings from each layer of the text encoder.
The authors trained diffusion model with average of all embeddings and also with average of normalized embeddings. 
Both experiments showed better results than the model which used embedding only from last layer.

This research is about training coefficients to aggregate the embeddings and put it to the frozen unet model.
The dataset was prepaired in the following way:

For each prompt, embeddings were extracted from 24 layers of text encoder and also for each prompt the "teacher" model (DeepFloyd/IF-I-XL) generated images.

All experiments were carried out on a model DeepFloyd/IF-I-M with 25 diffusion steps.
Since there are 24 embeddings , 24 coefficients were trained in the first experiment with lpips loss with the teacher's generated images and then in another setup with lpips+hpsv2.
The best model had 21.4 hpsv2 score on validation dataset (M model with embeddings only from the last layer had 22.5). [train logging](https://app.neptune.ai/o/nastya/org/transformer/runs/details?viewId=a06ed0b2-5308-4ddd-af35-f748aa3e7f12&detailsTab=metadata&shortId=TRAN-408&type=run)

In the second experiment 600 coefficients were trained (24 coefficients for each of 25 diffusion steps). The best model had 20.33 hpsv2 score. [train logging](https://app.neptune.ai/o/nastya/org/transformer/runs/details?viewId=a06ed0b2-5308-4ddd-af35-f748aa3e7f12&detailsTab=metadata&shortId=TRAN-352&type=run&path=.)

The study was conducted under the scientific supervision of [Aliev Mishan](https://github.com/thecrazymage).