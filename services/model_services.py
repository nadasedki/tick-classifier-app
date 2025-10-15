import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
from matplotlib import cm
from model import TickSpeciesSexModel

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Classes
species_list = ['dromidarii', 'excavatum', 'impeltatum', 'marginatum', 'scupinse']
sex_list = ['femelle', 'male']

# Transform
transform_infer = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load model
model = TickSpeciesSexModel(num_species=5, num_sex=2, backbone_name='resnet50').to(device)
model.load_state_dict(torch.load('model/best_model3.pth', map_location=device))
model.eval()
print("✅ Model loaded successfully")

# Allowed file check
def allowed_file(filename, allowed_ext={'png','jpg','jpeg'}):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext

# Prediction function
def predict_species_sex(img_path, species_thresh=60.0, sex_thresh=60.0):
    try:
        image = Image.open(img_path).convert('RGB')
    except UnidentifiedImageError:
        raise ValueError("Invalid image file")

    image_tensor = transform_infer(image).unsqueeze(0).to(device)

    with torch.no_grad():
        species_out, sex_out = model(image_tensor)
        _, predicted_species = torch.max(species_out, 1)
        _, predicted_sex = torch.max(sex_out, 1)
        
        species_probs = F.softmax(species_out, dim=1)[0]
        sex_probs = F.softmax(sex_out, dim=1)[0]
        species_probs = [p.item() for p in species_probs]
        sex_probs = [p.item() for p in sex_probs]

        species_idx = int(torch.tensor(species_probs).argmax().item())
        sex_idx = int(torch.tensor(sex_probs).argmax().item())
        top_species_conf = species_probs[species_idx] * 100
        top_sex_conf = sex_probs[sex_idx] * 100
        species_pred = species_list[species_idx] if top_species_conf >= species_thresh else "unknown"
        sex_pred = sex_list[sex_idx] if top_sex_conf >= sex_thresh else "unknown"

    return {
        'species': species_pred,
        'species_confidence': round(species_probs[species_idx] * 100, 2),
        "species_all": {species_list[i]: round(prob*100,2) for i, prob in enumerate(species_probs)},
        'sex': sex_pred,
        'sex_confidence': round(sex_probs[sex_idx] * 100, 2),
        "sex_all": {sex_list[i]: round(prob*100,2) for i, prob in enumerate(sex_probs)}
    }

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
