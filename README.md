# Few-Shot Image Generation Through Pre-trained Score-Based Generative Models

While Score-Based Models has increased in popularity recently as we are now able to sample GAN-level quality without adversely training, all while having the ability to compute the exact log-likelihood. SBGM do not require retraining the model to do inverse problem solving, but training such model on a certain task will require a large dataset. Few-shot image generation suffers from over-fitting especially when the number of samples is as small as 10 samples, therefore we explore the feasibility of few-shot image generation for score-based generative models. This will allow us to leverage the benefits of SBGMs all while demonstrating diverse and realistic images.

## What does this code do?
Our main contribution is exploring the possibility of domain adaptation in SBGMs as this was possible before with GANs. We conduct experiments to answer those questions and explore the Score-Based models ability to do transfer learning and domain adaptation.

The code supports training pre-trained models with a specified number of extra steps, evaluating the sample quality and likelihoods of existing models and models trained for extra steps. The code SBGM is forked from https://github.com/yang-song/score_sde_pytorch, and additional modified code is explained in the next section.

## What does each notebook do?

*ffhq256.ipynb:* This is the main driver code, it trains a pretrained model for a specified number of steps on the babies small dataset.

*datasets.py:* This has a modified datasetloader for the babies dataset.

*png-to-tfrecords.ipynb:* This converts PNG images to tfrecord format, contains few different methods.

You can change the small set used in to train the model and make likelihood calculation on in the config files under ffhq256 continous.


## References

```bib
@inproceedings{
  song2021scorebased,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Yang Song and Jascha Sohl-Dickstein and Diederik P Kingma and Abhishek Kumar and Stefano Ermon and Ben Poole},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=PxTIG12RRHS}
}
```
