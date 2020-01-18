# neural_viz
Visualization for neural networks

### Under development

The aim of the library is to provide a simple interface to visualize the feature representation of Convolutional Neural Networks. This includes support for visualizing a single image as well as visualizing the changes in learned features as you train the network. **Works with PyTorch models**.

### GradCam classification visualization for a single image 

To obtain [GradCam](https://arxiv.org/abs/1610.02391) visualization for a single image use **grad_cam_viz** method

```
grad_cam_viz(model, layer, x, y, proc_x=None, unproc_x=None):
    '''
    model (PyTorch Module) - PyTorch model whose features needs to be visualized
    layer (list) - a sequence of layer names representing the heirarchy to reach the layer inside the model 
                    whose visualization needs to be done. 
                    Ex: ['5', '7'] represents that the desired layer can be found using model._modules['5']._modules['7']
    x (PyTorch rank 3 tensor) - the input image. Expected to be of shape (3, h, w) where h is the height and w is the width
    y (PyTorch rank 0 tensor i.e. a scalar) - the true class of "x". The expectation is that model(x)[y] represents the 
                    score for the true class
    proc_x (function): proc_x will be passed "x" as the first and only parameter. Any preprosessing can be done here before 
                    the score for x is evaluated
    unproc_x (function): function should take "x" and return the values back to image space i.e. remove any preprocessing steps 
                    performed either in "proc_x" or before calling "grad_cam_viz" method
    '''
```

To obtain continous visualizations of images as the model is being trained, use "CV2VisualOutput" or "CV2VisualOutputFromFolder"