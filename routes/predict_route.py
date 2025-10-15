
from werkzeug.utils import secure_filename
import os, uuid, logging
from PIL import Image, UnidentifiedImageError
#from services.model_services import predict_species_sex, generate_gradcam_resnet50, overlay_gradcam, allowed_file, model, transform_infer
import torch
from services.gradcam_service import generate_gradcam_resnet50, overlay_gradcam
from services.prediction_service import predict_species_sex , allowed_file, model, transform_infer
import base64
from io import BytesIO
from services.db_service import save_prediction_with_images,get_all_predictions_from_db, get_image_from_gridfs

from flask import Blueprint, request, jsonify, send_file


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

predict_bp = Blueprint('predict', __name__)
@predict_bp.route('/image/<image_id>')
def serve_image(image_id):
    """
    Serve image from GridFS by calling db_service function.
    """
    image_bytes = get_image_from_gridfs(image_id)
    if image_bytes is None:
        return jsonify({'error': 'Image not found'}), 404

    return send_file(BytesIO(image_bytes), mimetype='image/jpeg')

"""
from gridfs import GridFS
from pymongo import MongoClient
# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['tick_db']
fs = GridFS(db)



@predict_bp.route('/image/<image_id>')
def serve_image(image_id):
    from io import BytesIO
    from flask import send_file
    from bson import ObjectId

    try:
        file = fs.get(ObjectId(image_id))
        return send_file(BytesIO(file.read()), mimetype='image/jpeg')
    except:
        return jsonify({'error': 'Image not found'}), 404
"""

"""
@predict_bp.route('/image/<image_id>')
def serve_image(image_id):
    
    #Serve image stored in GridFS by its ObjectId.
    
    image_bytes = get_image_from_gridfs(image_id)  # returns raw bytes
    if image_bytes is None:
        return jsonify({'error': 'Image not found'}), 404

    return send_file(BytesIO(image_bytes), mimetype='image/jpeg')
"""
@predict_bp.route("/all", methods=["GET"])
def get_all_predictions():
    result = get_all_predictions_from_db()  # returns a JSON array
    
    # If result is empty, return empty array instead of a dict
    if not result.json:  # result.json holds the array from jsonify
        return jsonify([])  # always return an array
    return result  # already a JSON array


@predict_bp.route('/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error':'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error':'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400

    try:
        # Read image into memory
        file_bytes = file.read()
        image = Image.open(BytesIO(file_bytes)).convert('RGB')

        # Verify image
        try:
            image.verify()
        except UnidentifiedImageError:
            return jsonify({'error':'Invalid or corrupted image file'}), 400

        # Predict
        prediction = predict_species_sex(BytesIO(file_bytes))
        logging.info(f"Prediction: {prediction}")

        # Grad-CAM only if species detected
        if prediction['species'] != "unknown":
            image_tensor = transform_infer(image).unsqueeze(0).to(device)
            heatmap = generate_gradcam_resnet50(model, image_tensor)

            # Overlay Grad-CAM on original image in memory
            overlay_image_bytes = overlay_gradcam(BytesIO(file_bytes), heatmap, in_memory=True)

            # Save prediction + images to DB
            inserted_id,gradcam_id = save_prediction_with_images(prediction, file_bytes, overlay_image_bytes)
            prediction['db_id'] = str(inserted_id)
            #prediction['gradcam_url'] = f"/predict/image/{str(gradcam_id)}"
            # Encode Grad-CAM overlay in base64 and send directly
            # If overlay_image_bytes is BytesIO, convert to raw bytes
            if hasattr(overlay_image_bytes, "getvalue"):
                overlay_raw = overlay_image_bytes.getvalue()
            else:
                overlay_raw = overlay_image_bytes
            prediction['gradcam_image'] = base64.b64encode(overlay_raw).decode('utf-8')

        else:
            prediction['db_id'] = None
            prediction['gradcam_url'] = None

        return jsonify(prediction)

    except ValueError as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logging.exception("Unexpected error during prediction")
        return jsonify({'error':'Internal server error'}), 500

# Error handlers remain the same

# Error handlers
@predict_bp.app_errorhandler(413)
def file_too_large(e):
    return jsonify({'error':'File too large. Maximum allowed size: 20 MB'}), 413

@predict_bp.app_errorhandler(404)
def page_not_found(e):
    return jsonify({'error':'Endpoint not found'}), 404

@predict_bp.app_errorhandler(500)
def internal_error(e):
    return jsonify({'error':'Internal server error'}), 500
