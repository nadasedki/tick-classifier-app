import os
import uuid
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
from model import TickSpeciesSexModel  # Your actual model class
import torch.nn.functional as F
import cv2
import numpy as np
from matplotlib import cm
# =========================
# CONFIGURATION
# =========================
def generate_gradcam_resnet50_oldcode(model, image_tensor, target_class=None):
    """
    image_tensor: 1x3xHxW
    model: TickSpeciesSexModel with ResNet50 backbone
    target_class: int, index of species class (optional)
    Returns: heatmap as numpy array (H,W,3)
    """
    model.eval()
    gradients = []
    activations = []

    # Hook functions
    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def save_activations(module, input, output):
        activations.append(output)

    # Register hooks on the last conv layer of ResNet50
    layer = model.backbone.layer1
    handle_forward = layer.register_forward_hook(save_activations)
    handle_backward = layer.register_backward_hook(save_gradients)

    # Forward pass
    output_species, output_sex = model(image_tensor)
    if target_class is None:
        target_class = output_species.argmax(dim=1)[0].item()

    # Backward pass
    model.zero_grad()
    loss = output_species[0, target_class]
    loss.backward()

    # Compute Grad-CAM
    gradient = gradients[0].cpu().data.numpy()[0]      # [C,H,W]
    activation = activations[0].cpu().data.numpy()[0]  # [C,H,W]
    weights = np.mean(gradient, axis=(1,2))            # Global average pooling
    cam = np.sum(weights[:, None, None] * activation, axis=0)

    # ReLU & normalize
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    # Resize to original image size
    H, W = image_tensor.size(2), image_tensor.size(3)
    cam = cv2.resize(cam, (W, H))

    # Convert to heatmap RGB
    heatmap = cm.jet(cam)[:, :, :3] * 255
    heatmap = heatmap.astype(np.uint8)

    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()

    return heatmap

def generate_gradcam_resnet50(model, image_tensor, target_class=None):
    model.eval()
    
    # Get the last conv layer
    target_layer = model.backbone.layer4

    # Store activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)  # recommended

    # Forward pass
    species_out, sex_out = model(image_tensor)
    if target_class is None:
        target_class = species_out.argmax(dim=1)[0].item()

    # Backward pass
    model.zero_grad()
    loss = species_out[0, target_class]
    loss.backward()

    # Grad-CAM computation
    grad = gradients['value'][0].detach().cpu().numpy()   # [C,H,W]
    act = activations['value'][0].detach().cpu().numpy()  # [C,H,W]
    weights = np.mean(grad, axis=(1,2))
    cam = np.sum(weights[:, None, None] * act, axis=0)
    cam = np.maximum(cam, 0)
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)

    # Resize to input size
    H, W = image_tensor.size(2), image_tensor.size(3)
    cam = cv2.resize(cam, (W, H))

    # RGB heatmap
    heatmap = cm.jet(cam)[:, :, :3] * 255
    heatmap = heatmap.astype(np.uint8)

    # Remove hooks
    handle_f.remove()
    handle_b.remove()

    return heatmap


def overlay_gradcam(image_path, heatmap, output_path=None):
    """
    image_path: original image path
    heatmap: numpy array HxWx3
    output_path: save overlay image (optional)
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  # width, height

    overlay = cv2.addWeighted(img, 0.6, heatmap_resized, 0.4, 0)
    
    if output_path is None:
        output_path = image_path.replace(".jpg", "_gradcam.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output_path



# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 5 MB limit

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Classes
species_list = ['dromidarii', 'excavatum', 'impeltatum', 'marginatum', 'scupinse']
sex_list = ['femelle', 'male']

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference transform (no augmentation)
transform_infer = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load model
model = TickSpeciesSexModel(num_species=5, num_sex=2, backbone_name='resnet50').to(device)
model.load_state_dict(torch.load('model/best_model3.pth', map_location=device))
model.eval()
logging.info("✅ Model loaded successfully")


# =========================
# HELPER FUNCTIONS
# =========================
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


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
        
        # Convert logits to probabilities
        species_probs = F.softmax(species_out, dim=1)[0]  # remove batch dim
        sex_probs = F.softmax(sex_out, dim=1)[0]

        # Convert to Python floats
        species_probs = [p.item() for p in species_probs]
        sex_probs = [p.item() for p in sex_probs]

        # Get top prediction indices
        species_idx = int(torch.tensor(species_probs).argmax().item())
        sex_idx = int(torch.tensor(sex_probs).argmax().item())
        # Get confidence scores new code 
        top_species_conf = species_probs[species_idx] * 100
        top_sex_conf = sex_probs[sex_idx] * 100
        # Apply thresholds
        species_pred = species_list[species_idx] if top_species_conf >= species_thresh else "unknown"
        sex_pred = sex_list[sex_idx] if top_sex_conf >= sex_thresh else "unknown"


    return {
        #'species': species_list[predicted_species.item()],
        'species': species_pred,
        'species_confidence': round(species_probs[ species_idx] * 100, 2),
        #"species_all": {species_list[i]: round(float(prob*100),2) for i, prob in enumerate(species_probs)},
        "species_all": {species_list[i]: round(prob*100,2) for i, prob in enumerate(species_probs)},
        
        #'sex': sex_list[predicted_sex.item()],
        'sex': sex_pred,
        'sex_confidence': round(sex_probs[ sex_idx] * 100, 2),
        #"sex_all": {sex_list[i]: round(float(prob*100),2) for i, prob in enumerate(sex_probs)}
        "sex_all": {sex_list[i]: round(prob*100,2) for i, prob in enumerate(sex_probs)}
  
    }


# =========================
# ROUTES
# =========================
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPE'}), 400
    #if file and allowed_file(file.filename):
        # Secure and unique filename
    try:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        try:
            with Image.open(filepath) as img:
                img.verify()
        except UnidentifiedImageError:
            return jsonify({'error': 'Invalid or corrupted image file'}), 400
        try:
            prediction = predict_species_sex(filepath)
            prediction['image_url'] = filepath
            logging.info(f"Prediction: {prediction}")
            if prediction['species'] != "unknown":
            # Generate Grad-CAM heatmap
            # Convert image to tensor for Grad-CAM
                image = Image.open(filepath).convert('RGB')
                image_tensor = transform_infer(image).unsqueeze(0).to(device)

            # Generate Grad-CAM heatmap
                heatmap = generate_gradcam_resnet50(model, image_tensor)
                overlay_name = f"gradcam_{os.path.basename(filepath)}"
                overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], overlay_name)
                overlay_path = overlay_gradcam(filepath, heatmap, overlay_path)

            # Make URL for frontend
                prediction['gradcam_url'] = '/' + overlay_path.replace("\\","/")
            else:
                prediction['gradcam_url'] = None # No Grad-CAM for irrelevant images

            #end
            return jsonify(prediction)
        except ValueError as e:
            logging.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 400
    
    
    except Exception as e:
        logging.exception("Unexpected error during prediction")
        return jsonify({'error': 'Internal server error'}), 500

    #return jsonify({'error': 'Invalid file type'}), 400

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'error': 'File too large. Maximum allowed size: 20 MB'}), 413

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# =========================
# MAIN
# =========================
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
