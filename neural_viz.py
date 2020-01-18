from functools import reduce, partial
import matplotlib.pyplot as plt
import pdb
import PIL
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import Normalize

imgnet_normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def find_layer(model, layer_hierarchy):
    return reduce(lambda model, layer: model._modules[layer], layer_hierarchy, model)

BLUE, RED = (255, 0, 0), (0, 0, 255)

def draw_point_on_img(img, point, color=(0, 0, 255), thickness=5):
    return cv2.circle(img, point, 5, color, cv2.FILLED)

def tensor_to_np_img(img): 
    img = img.cpu().numpy().transpose((1, 2, 0))
    return img

def tensor_to_cv2_img(img, scale=255.):
    img = tensor_to_np_img(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img * scale if scale else img

def np_to_tensor(img, scale=255., device=None):
    return torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(scale).to(device=device)

def cv2_to_tensor(img, scale=255., device=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np_to_tensor(img,scale=scale, device=device)

def split_pos_neg(arr):
    pos = F.relu(arr)
    neg = torch.abs(arr * (arr < 0).float())
    return pos, neg

def find_min_max(arr):
    min, max = None, None
    
    for e in arr:
        cur_min, cur_max = torch.min(e), torch.max(e)
        if min is None or cur_min < min:
            min = cur_min
        if max is None or cur_max > max:
            max = cur_max
    return min, max

def scl_min_max(arr, min, max):
    return (arr - min) / max

def upsample(arr, size):
    return F.upsample(arr, size=size, mode='bilinear', align_corners=False)

def gen_heatmap(arr):
    return cv2_to_tensor(cv2.applyColorMap(np.uint8(255 * arr.cpu().squeeze()), cv2.COLORMAP_JET), device=arr.device)

def append_img(x, heatmap):
    conc = (x + heatmap)
    return  conc / conc.max() 

def img_heatmap_viz(x, y, pred, heatmap, reshape_map=True, get_cat_name=None):
    
    heatmap = (heatmap - heatmap.min() ) / heatmap.max()
    heatmap = F.upsample(heatmap, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
    heatmap = cv2_to_tensor(cv2.applyColorMap(np.uint8(255 * heatmap.cpu().squeeze()), cv2.COLORMAP_JET), device=heatmap.device)
    
    img = (x + heatmap)
    img = img / img.max()
    return tensor_to_cv2_img(heatmap), tensor_to_cv2_img(img)

def heatmap_viz_with_points(x, y, pred, heatmap, reshape_map=True, get_cat_name=None):
    heatmap, img = img_heatmap_viz(x, y, pred, heatmap, reshape_map=reshape_map)
    y, pred = y.reshape((-1, 2)), pred.reshape((-1, 2))
    
    for i in range(y.shape[0]):
        img = draw_point_on_img(img, tuple(y[i].cpu().detach().numpy()), color=BLUE)
        img = draw_point_on_img(img, tuple(pred[i].cpu().detach().numpy()), color=RED)

    return heatmap, img

def img_heatmap_no_relu_viz(x, y, pred, heatmap, reshape_map=True, get_cat_name=None):
    pos_map = F.relu(heatmap)
    neg_map = torch.abs(heatmap * (heatmap < 0).float())
    min, max = torch.tensor(0., device=heatmap.device).float(), torch.max(pos_map.max(), neg_map.max())
    pos_map = (pos_map - min ) / max 
    neg_map = (neg_map - min ) / max
    pos_map = F.upsample(pos_map, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
    neg_map = F.upsample(neg_map, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
    pos_map = cv2_to_tensor(cv2.applyColorMap(np.uint8(255 * pos_map.cpu().squeeze()), cv2.COLORMAP_JET), device=heatmap.device)
    neg_map = cv2_to_tensor(cv2.applyColorMap(np.uint8(255 * neg_map.cpu().squeeze()), cv2.COLORMAP_JET), device=heatmap.device)
    pos_img = (x + pos_map)
    pos_img = tensor_to_cv2_img(pos_img / pos_img.max())
    neg_img = (x + neg_map)
    neg_img = tensor_to_cv2_img(neg_img / neg_img.max())
    
    return [tensor_to_cv2_img(pos_map), tensor_to_cv2_img(neg_map)],[pos_img, neg_img]

def img_heatmap_no_relu_viz_conf_mat(x, y, pred, heatmap, heatmap_pred, reshape_map=True, get_cat_name=None):
    y_pos, y_neg = split_pos_neg(heatmap)
    pred_pos, pred_neg = split_pos_neg(heatmap_pred)
    min, max = find_min_max([y_pos, y_neg, pred_pos, pred_neg])
    y_pos, y_neg, pred_pos, pred_neg = scl_min_max(y_pos, min, max), scl_min_max(y_neg, min, max), scl_min_max(pred_pos, min, max), scl_min_max(pred_neg, min, max)
    up_size = (x.shape[-2], x.shape[-1])
    y_pos, y_neg, pred_pos, pred_neg = upsample(y_pos, up_size), upsample(y_neg, up_size), upsample(pred_pos, up_size), upsample(pred_neg, up_size)
    y_pos, y_neg, pred_pos, pred_neg = gen_heatmap(y_pos), gen_heatmap(y_neg), gen_heatmap(pred_pos), gen_heatmap(pred_neg)
    
    
    y_pos_img, y_neg_img, pred_pos_img, pred_neg_img = append_img(x, y_pos), append_img(x, y_neg), append_img(x, pred_pos), append_img(x, pred_neg)
    
    y_pos, y_neg, pred_pos, pred_neg = tensor_to_cv2_img(y_pos), tensor_to_cv2_img(y_neg), tensor_to_cv2_img(pred_pos), tensor_to_cv2_img(pred_neg)
    y_pos_img, y_neg_img, pred_pos_img, pred_neg_img = tensor_to_cv2_img(y_pos_img), tensor_to_cv2_img(y_neg_img), tensor_to_cv2_img(pred_pos_img), tensor_to_cv2_img(pred_neg_img)
    
    return [y_pos, y_neg, pred_pos, pred_neg],[y_pos_img, y_neg_img, pred_pos_img, pred_neg_img]

def _def_font_params():
    return cv2.FONT_HERSHEY_SIMPLEX, (20, 50), (20, 20), 0.45, (255, 255, 255), (0, 255, 0), 1

def layer_based_viz(viz_algo, viz_gen, model, layer, x, y, proc_model_out=None, proc_x=None, unproc_x=None, get_cat_name=None, upsample_saliency=True, y_and_pred=False):
    '''
    viz_algo - function(activations::PyTorch Tensor[], gradients) -> : function visualization 
    y_and_pred :: boolean - if True, get saliency map with input as (x,y) as well as (x, None)
    '''
    font, org, org_y, fontScale, color, color_y, thickness = _def_font_params()
    
    model_state_swap = True if model.training else False
    if model_state_swap:
        model.eval()
    
    layer = find_layer(model, layer)
    activations, gradients = None, None
    
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output
    
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output
    
    fhook_handle = layer.register_forward_hook(forward_hook)
    bhook_handle = layer.register_backward_hook(backward_hook)
    
    x = x if proc_x is None else proc_x(x)

    model.zero_grad()
    out = model(x[None])
    loss = out if proc_model_out is None else proc_model_out(out[0], x, y)
    loss.backward()
    print(f'Loss = {loss}')
    saliency_map = viz_algo(activations, gradients)

    loss_pred, out_pred, saliency_map_pred = None, None, None
    if y_and_pred:
        model.zero_grad()
        out_pred = model(x[None])
        loss_pred = out_pred if proc_model_out is None else proc_model_out(out_pred[0], x, y)
        loss_pred.backward()
        print(f'Loss = {loss_pred}')
        saliency_map_pred = viz_algo(activations, gradients)

        heatmap, img = viz_gen(unproc_x(x) if unproc_x else x, y, out, saliency_map, saliency_map_pred, get_cat_name=get_cat_name)
    else:
        heatmap, img = viz_gen(unproc_x(x) if unproc_x else x, y, out, saliency_map, get_cat_name=get_cat_name)

    if model_state_swap:
        model.train()    
    fhook_handle.remove()
    bhook_handle.remove()
    
    if upsample_saliency:
        saliency_map = F.upsample(saliency_map, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)
        if y_and_pred:
            saliency_map_pred = F.upsample(saliency_map_pred, size=(x.shape[-2], x.shape[-1]), mode='bilinear', align_corners=False)

    # return (out) as well
    return (img, heatmap, (saliency_map[0], saliency_map_pred[0])) if y_and_pred else (img, heatmap, saliency_map[0])

def grad_cam(activations, gradients, higher_is_better=True, relu=True, grad_mean=True):
    '''
    activations :: torch.Tensor shape=[batch_size, num_channels, H, W] - activations at the layer being visualized (i.e. output of the layer being visualized)
    gradients :: Tuple containing torch.Tensor shape=[batch_size, num_channels, H, W] - gradients with respect to the output of the layer being visualized
    higher_is_better :: boolean - if True, considers the number on which the gradient is obtained as moving towards optimality if higher 
        (Ex. output of a neural network for a particular class in a classification network). if False, considers the number to be better if lower (Ex. RMSE on a regression problem)
    relu :: boolean - if True, applies ReLU on the sum obtained after multiplying "alpha * activations".

    returns saliency map :: torch.Tensor shape=[batch_size, 1, H, W] - numbers representing the importance of a particular pixel. Higher numbers indicate higher importance.
    '''
    gradients = gradients[0]
    alpha = gradients.mean(dim=(2,3), keepdim=True) if grad_mean else gradients
    alpha = alpha if higher_is_better else (-1 * alpha)
    
    print(f'activations * alpha = {(alpha * activations).sum()}')

    saliency_map = (alpha * activations).sum(dim=1, keepdim=True)
    if relu:
        saliency_map = F.relu(saliency_map)
    
    return saliency_map

def grad_cam_proc_out(out, x, y):
    return out[y] if y is not None else torch.max(out)

def grad_cam_proc_out_regression(out, x, y):
    return F.mse_loss(out, y)

def pos_neg_plot(neg, pos):
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    for i in range(len(axs)):
            axs[i].tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

    axs[0].imshow(neg )
    axs[0].set_xlabel('Negative', labelpad=20)
    axs[1].imshow(pos )
    axs[1].set_xlabel('Positive', labelpad=20)
    
    return fig

def proc_plt(img):
    img = img.astype('uint8')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def saliency_confusion_plot(y_neg, y_pos, pred_neg, pred_pos, figsize=(10,10)):
    fig, axs = plt.subplots(2,2, figsize=figsize)
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            axs[i][j].tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    #         axs[i][j].tick_params(labelbottom='off',labeltop='on')
            axs[i][j].xaxis.set_label_position('top')

    axs[0][0].imshow(y_neg )
    axs[0][0].set_ylabel('Ground Truth')
    axs[0][0].set_xlabel('Negative', labelpad=20)
    axs[0][1].imshow(y_pos )
    axs[0][1].set_xlabel('Positive', labelpad=20)
    axs[1][0].imshow(pred_neg )
    axs[1][0].set_ylabel('Prediction')
    axs[1][1].imshow(pred_pos )
    return fig


def grad_cam_viz(model, layer, x, y, proc_x=None, unproc_x=None, get_cat_name=None):
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

    returns - image, heatmap, saliency map
    '''
    return layer_based_viz(grad_cam, img_heatmap_viz, model, layer, 
                    x, y, proc_model_out=grad_cam_proc_out, proc_x=proc_x, 
                    unproc_x=unproc_x, get_cat_name=get_cat_name,upsample_saliency=False)

def grad_cam_no_relu_viz(model, layer, x, y, proc_x=None, unproc_x=None, get_cat_name=None, analysis=True):
    return layer_based_viz(partial(grad_cam, relu=False), img_heatmap_no_relu_viz, model, layer, 
                    x, y, grad_cam_proc_out, proc_x, unproc_x, get_cat_name=get_cat_name)


def contribution_cam_viz(model, layer, x, y, proc_x=None, unproc_x=None, get_cat_name=None):
    return layer_based_viz(partial(grad_cam, grad_mean=False), img_heatmap_viz, model, layer, 
                    x, y, proc_model_out=grad_cam_proc_out, proc_x=proc_x, 
                    unproc_x=unproc_x, get_cat_name=get_cat_name,upsample_saliency=False)

def contribution_cam_no_relu_viz(model, layer, x, y, proc_x=None, unproc_x=None, get_cat_name=None, analysis=True):
    return layer_based_viz(partial(grad_cam, relu=False, grad_mean=False), img_heatmap_no_relu_viz, model, layer, 
                    x, y, grad_cam_proc_out, proc_x, unproc_x, get_cat_name=get_cat_name)


def grad_cam_regression_viz(model, layer, x, y, proc_x=None, unproc_x=None):
    return layer_based_viz(partial(grad_cam, higher_is_better=False), heatmap_viz_with_points, model, layer, 
                    x, y, grad_cam_proc_out_regression, proc_x, unproc_x)

def grad_cam_conf_mat_viz(model, layer, x, y, proc_x=None, unproc_x=None, get_cat_name=None, analysis=True):
    return layer_based_viz(partial(grad_cam, relu=False), img_heatmap_no_relu_viz_conf_mat, model, layer, 
                    x, y, grad_cam_proc_out, proc_x, unproc_x, get_cat_name=get_cat_name, y_and_pred=True)

def grad_cam_saliency_confusion_map(model, layer, x, y, proc_x=None, unproc_x=None, get_cat_name=None, analysis=True, figsize=(10,10)):
    img, heatmap, saliency_map =  grad_cam_conf_mat_viz(model, layer, x, y, proc_x, unproc_x, get_cat_name, analysis)
    fig = saliency_confusion_plot(proc_plt(img[1]), proc_plt(img[0]), proc_plt(img[3]),proc_plt(img[2]), figsize=figsize)
    return fig

