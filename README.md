# neural_viz
Visualization for neural networks

The aim of the library is to provide a simple interface to visualize the feature representation of Convolutional Neural Networks. **Works with PyTorch models**.

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

#### Example result after applying grad_cam_viz to a Pneumonia classification problem (y is class 'Pneumonia')- 

![Imgur](https://imgur.com/ZCcHI3X.jpg)

### To visualize both the positive and negative contributions towards the specified class, use "grad_cam_no_relu_viz" method

#### Result obtained by visualizing both positive and negative contributions towards class "pneumonia" (same example as shown above) -

![Imgur](https://imgur.com/mqFD99p.jpg)


### Saliency Confusion Matrix

Saliency Confusion Matrix provides a visualization of the positive and negative contributions for both the predicted class as well as the ground truth class. Use the method "grad_cam_saliency_confusion_map" to obtain the Saliency Confusion Matrix.  

#### Saliency Confusion Matrix for the same example explored above - 

![Imgur](https://imgur.com/Ob94q2X.jpg)
