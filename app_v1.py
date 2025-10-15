import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import TickSpeciesSexModel  # Import your actual model class

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Configuration du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transformations (identique à Colab)
transform = transforms.Compose([
        transforms.Resize((128, 128)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomRotation(15),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

# Classes (à adapter selon votre modèle)
species_list = ['dromidarii','excavatum', 'impeltatum', 'marginatum', 'scupinse'] 
sex_list = ['femelle', 'male']
"""
Vérification de l'ordre:
0: dromidarii
1: excavatum
2: impeltatum
3: marginatum
4: scupinse
"""
# Chargement du modèle
model = TickSpeciesSexModel(num_species=5, num_sex=2, backbone_name='resnet50').to(device)
#model = YourModelClass().to(device)  # Remplacez par votre classe de modèle
model.load_state_dict(torch.load('model/best_model3.pth', map_location=device))
model.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'Aucun fichier uploadé'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Aucun fichier sélectionné'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Prédiction
            species, sex = predict_species_sex(filepath, model, transform, device)
            
            return jsonify({
                'species': species,
                'sex': sex,
                'image_url': filepath
            })
    
    return render_template('index.html')

def predict_species_sex(img_path, model, transform, device):
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        species_out, sex_out = model(image_tensor)
        _, predicted_species = torch.max(species_out, 1)
        _, predicted_sex = torch.max(sex_out, 1)
    
    return species_list[predicted_species.item()], sex_list[predicted_sex.item()]
"""
def predict_species_sex(img_path, model, transform, device):
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        species_out, sex_out = model(image_tensor)
        species_probs = torch.softmax(species_out, dim=1)
        sex_probs = torch.softmax(sex_out, dim=1)

        species_idx = torch.argmax(species_probs, dim=1).item()
        sex_idx = torch.argmax(sex_probs, dim=1).item()

    return {
        'species': species_list[species_idx],
        'species_confidence': round(species_probs[0, species_idx].item() * 100, 2),
        'sex': sex_list[sex_idx],
        'sex_confidence': round(sex_probs[0, sex_idx].item() * 100, 2)
    }
"""
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)