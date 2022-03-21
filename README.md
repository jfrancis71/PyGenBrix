# PyGenBrix
Experimental Generative Models in PyTorch

PyGenBrix is an experimental project for constructing generative probabilistic models with an emphasis on composability of probability distributions. It is aimed at vision applications.


Distribution Objects:
These support log_prob and sample methods.
log_prob actually returns a dictionary which will have a key "log_prob". They may return others keys which can be useful for logging purposes.
On construction they take parameters appropriate to that distribution, eg mean and std for normal distribution.
For composable (learnable) distributions, the distribution object (on construction) takes a Layer type object. This describes the type of the output distribution.


Layer Objects:
These implement the forward method and return associated Distribution objects. This design was based on the layer objects principles used by the TensorFlow Probability package.
They are intented to be used as the final layer of a neural network.


Internal Distribution Objects:
These are for internal use only. They are typically stateful, and have the option of passing in conditional information. Example use case: We want a PixelCNNDistribution and a PixelCNNLayer (this takes in conditional information and returns distribution). As the implementation of these two objects are quite similar (most code is around implementing PixelCNN) we implement them as an internal object and then build corresponding Distribution and Layer objects as appropriate wrappers (to save on duplicated code).


Implemented:
Glow
ParallelCNN
PixelCNN
VDVAE

Glow:
Drives modified version of https://github.com/rosinality/glow-pytorch

ParallelCNN:
My own version of ParallelCNN, based on ideas in https://arxiv.org/abs/1703.03664

PixelCNN:
Drives modified version of https://github.com/pclucas14/pixel-cnn-pp

VDVAE:
Drives modified version of github openai/vdvae
