"""import cv2
import numpy as np
from matplotlib import cm

# Grad-CAM functions
def generate_gradcam_resnet50(model, image_tensor, target_class=None):
    model.eval()
    target_layer = model.backbone.layer4
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    species_out, sex_out = model(image_tensor)
    if target_class is None:
        target_class = species_out.argmax(dim=1)[0].item()

    model.zero_grad()
    loss = species_out[0, target_class]
    loss.backward()

    grad = gradients['value'][0].detach().cpu().numpy()
    act = activations['value'][0].detach().cpu().numpy()
    weights = np.mean(grad, axis=(1,2))
    cam = np.sum(weights[:, None, None] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam)+1e-8)
    H, W = image_tensor.size(2), image_tensor.size(3)
    cam = cv2.resize(cam, (W,H))
    heatmap = cm.jet(cam)[:,:,:3]*255
    heatmap = heatmap.astype(np.uint8)

    handle_f.remove()
    handle_b.remove()

    return heatmap

def overlay_gradcam(image_path, heatmap, output_path=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)
    if output_path is None:
        output_path = image_path.replace(".jpg","_gradcam.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path

"""
import cv2
import numpy as np
from matplotlib import cm
from PIL import Image
from io import BytesIO

# Grad-CAM functions
def generate_gradcam_resnet50(model, image_tensor, target_class=None):
    model.eval()
    target_layer = model.backbone.layer4
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    species_out, sex_out = model(image_tensor)
    if target_class is None:
        target_class = species_out.argmax(dim=1)[0].item()

    model.zero_grad()
    loss = species_out[0, target_class]
    loss.backward()

    grad = gradients['value'][0].detach().cpu().numpy()
    act = activations['value'][0].detach().cpu().numpy()
    weights = np.mean(grad, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    H, W = image_tensor.size(2), image_tensor.size(3)
    cam = cv2.resize(cam, (W, H))
    heatmap = cm.jet(cam)[:, :, :3] * 255
    heatmap = heatmap.astype(np.uint8)

    handle_f.remove()
    handle_b.remove()

    return heatmap

def overlay_gradcam(image_input, heatmap, in_memory=False):
    """
    Args:
        image_input: path or BytesIO of the original image
        heatmap: numpy array from generate_gradcam_resnet50
        in_memory: if True, return BytesIO instead of saving to disk
    """
    # Load image from path or BytesIO
    if isinstance(image_input, BytesIO):
        img = np.array(Image.open(image_input).convert("RGB"))
    else:
        img = cv2.imread(image_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    overlay = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)

    if in_memory:
        pil_img = Image.fromarray(overlay)
        output_bytes = BytesIO()
        pil_img.save(output_bytes, format="JPEG")
        output_bytes.seek(0)
        return output_bytes
    else:
        if isinstance(image_input, BytesIO):
            raise ValueError("Cannot save to disk without a filepath")
        output_path = image_input.replace(".jpg", "_gradcam.jpg")
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        return output_path
