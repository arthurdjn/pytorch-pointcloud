# TODOS

- [ ] Possibility to add multiple biases, activation fns, etc. to MLP and SharedMLP
- [ ] Add generic Conv blocks (CNN -> BatchNorm -> ReLU) that are reused everywhere
- [ ] Refactor the SharedMLP blocks. Could be better that the conv blocks is in fact a sequential block, at list when printed out we will be able to see the structure of the network easily. Also remove the shared_mlp2d functions, replace with classes ? The API will be cleaner. Leave the function methods to instantiate specific pretrained models, just like torchvision or timm does.
- [ ] Think of a factory function or decorator (best) to register the models in a dictionary, so that we can easily instantiate them by name.
- [ ] Think of a config per model / pretrained params to easily load pretrained models. Not urgent but to keep in mind.
- [ ] Refactor model modules naming. We may want to have a more consistent naming convention, like should sub modules always start by the model name (e.g. `PointNetConv` instead of `Conv`) or not ? Some blocks are reused in different models, so we should think about it. Look deeply into the timm library to see how they handle this or ultralytics, they have both clean structures.
- [] Add `pre_logits` params. Follow timm convention: `forward_head(x, pre_logits=False)` fn added to all models to allow separate calls of `forward_features` + `forward_head`

    ```python
    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    ```

- [ ] Rename all the MLP stuff to maybe `LinearMLP`, `Conv1dML` and `Conv2dMLP` to be more explicit ?

## Urgent

- [ ] Need to add tests ASAP.

## Models

- [ ] Finalize VoteNet and add training example!!

Add gradcheks and unit tests for ops, datasets and so on.

Start working on the lightning interface to conduct more advanced training and experiments.

Add object detection training for PVCNN following config https://github.com/mit-han-lab/pvcnn/blob/master/configs/kitti/frustum/pvcnne.py

## Misc

- [ ] Remove the default_tensor function in utils
- [ ] Add better tests
- [ ] Refactor naming conventions for points

- [x] Finish scatter, make sure they are optimized
- [ ] Update the grid cluster to handle large number of channels
- [x] Update the grid cluster CPU id matching to match the GPU version from the torch-scatter test utils
- [ ] Use `std::clamp` in most functions to clamp the length to (0, N)
- [ ] Add packed version for all C++ and CUDA functions. Maybe add a AT_DISPATCH_DATA_TYPE_MODES macro to handle all the types at once (`packed`, `batched`, `auto`). Or maybe use it in python directly ? But we might have to define multiple torch Functions ... ?

- [ ] Add checks for cuda tensors and sizes in csrc
