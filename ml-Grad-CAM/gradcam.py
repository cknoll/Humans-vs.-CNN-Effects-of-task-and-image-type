import torch
import torch.nn.functional as F
import numpy as np


def get_gradcam(model,target_layer,image_tensor,label_idx, output_shape=None):
    # retrieve the features and gradients from the target layer using hooks.
    hook_dict = {}
    def features_hook(m,i,o):
        hook_dict["features"] = o
    def grad_hook(m,i,o):
        hook_dict["grads"] = o

    handle_f = target_layer.register_forward_hook(features_hook)
    handle_g = target_layer.register_backward_hook(grad_hook) 
    
    out = model.eval()(image_tensor)
    out[:,label_idx].backward(retain_graph=True)
    
    features = hook_dict["features"][0].squeeze().cpu().detach()
    gradients = hook_dict["grads"][0].squeeze().cpu().detach()
    
    handle_f.remove()
    handle_g.remove()
    # we can now pool our gradients for our features
    pooled_grads = torch.mean(gradients, dim = [1,2])
    
    weighted_features = torch.zeros_like(features)
    # apply the pooled gradients as weight to our features.
    for i in range(features.shape[0]):
        weighted_features[i] = features[i]*pooled_grads[i]
    # Since we are interested in the positive features, we use a Relu to prevent negative features from impacting our output too much. 
    raw_gcam  = F.relu(torch.mean(weighted_features,dim=[0]))
    normalized_gcam = (raw_gcam - raw_gcam.min())/(raw_gcam.max()-raw_gcam.min())
    if (output_shape != None):
        image_tensor = F.interpolate(image_tensor.reshape(1,3,image_tensor.shape[-1],image_tensor.shape[-2]),output_shape, mode='bilinear')
    else:
        output_shape = (image_tensor.shape[-1],image_tensor.shape[-2])
    # our results are upsampled to the desired size.
    upsampled_gcam = F.interpolate(normalized_gcam.reshape(1,1,features.shape[1],features.shape[2]),output_shape, mode = 'bilinear')
    output_cam = upsampled_gcam.squeeze().numpy()
    image_out = np.moveaxis(image_tensor.squeeze().numpy(),0,2)
    image_out = (image_out - np.min(image_out))/(np.max(image_out)-np.min(image_out))
    del hook_dict
    # for ease of use we return the normalized image alongside the normalized gradcam results. 
    return output_cam, image_out