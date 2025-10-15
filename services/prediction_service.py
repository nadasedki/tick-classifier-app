import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
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
