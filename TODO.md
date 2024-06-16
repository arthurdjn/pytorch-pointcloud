# TODOS

- [ ] Possibility to add multiple biases, activation fns, etc. to MLP and SharedMLP
- [ ] Add generic Conv blocks (CNN -> BatchNorm -> ReLU) that are reused everywhere
- [ ] Refactor the SharedMLP blocks. Could be better that the conv blocks is in fact a sequential block, at list when printed out we will be able to see the structure of the network easily. Also remove the shared_mlp2d functions, replace with classes ? The API will be cleaner. Leave the function methods to instantiate specific pretrained models, just like torchvision or timm does.
- [ ] Think of a factory function or decorator (best) to register the models in a dictionary, so that we can easily instantiate them by name.
- [ ] Think of a config per model / pretrained params to easily load pretrained models. Not urgent but to keep in mind.
- [ ] Refactor model modules naming. We may want to have a more consistent naming convention, like should sub modules always start by the model name (e.g. `PointNetConv` instead of `Conv`) or not ? Some blocks are reused in different models, so we should think about it. Look deeply into the timm library to see how they handle this or ultralytics, they have both clean structures.

## Urgent

- [ ] Need to add tests ASAP.

## Models

- [ ] Finalize VoteNet and add training example!!
