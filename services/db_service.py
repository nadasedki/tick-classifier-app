from pymongo import MongoClient
from gridfs import GridFS
from datetime import datetime
from flask import  jsonify
client = MongoClient('mongodb://localhost:27017/')
db = client['tick_db']
fs = GridFS(db)
collection = db['predictions']
"""
def save_prediction_to_db(prediction: dict, original_img_bytes: bytes, gradcam_img_bytes: bytes = None):
    orig_file_id = fs.put(original_img_bytes, filename="original.jpg", contentType='image/jpeg')
    prediction['original_image_id'] = str(orig_file_id)

    if gradcam_img_bytes:
        gradcam_file_id = fs.put(gradcam_img_bytes, filename="gradcam.jpg", contentType='image/jpeg')
        prediction['gradcam_image_id'] = str(gradcam_file_id)
    else:
        prediction['gradcam_image_id'] = None

    prediction['timestamp'] = datetime.utcnow()
    collection.insert_one(prediction)
    return prediction
"""
def save_prediction_with_images(prediction: dict, original_bytes: bytes, gradcam_bytes: bytes):
    """
    Save prediction along with original and Grad-CAM images into MongoDB.

    Args:
        prediction (dict): Dictionary of prediction results.
        original_bytes (bytes): Original image bytes.
        gradcam_bytes (bytes): Grad-CAM overlay image bytes.

    Returns:
        inserted_id: MongoDB ObjectId of the saved prediction.
    """
    # Convert BytesIO -> bytes if needed
    if hasattr(original_bytes, "getvalue"):
        original_bytes = original_bytes.getvalue()
    if hasattr(gradcam_bytes, "getvalue"):
        gradcam_bytes = gradcam_bytes.getvalue()

    # Save images in GridFS
    original_id = fs.put(original_bytes, filename=f"original_{datetime.utcnow().timestamp()}.jpg")
    gradcam_id = fs.put(gradcam_bytes, filename=f"gradcam_{datetime.utcnow().timestamp()}.jpg")

    # Store prediction info with references to image ids
    prediction_record = prediction.copy()
    prediction_record.update({
        "original_image_id": original_id,
        "gradcam_image_id": gradcam_id,
        "timestamp": datetime.utcnow()
    })

    # Insert into collection
    result = collection.insert_one(prediction_record)
    print(f"Prediction saved with ID: {result.inserted_id}")
    return result.inserted_id,gradcam_id
from bson import ObjectId
from gridfs import GridFS
"""
def get_image_from_gridfs(image_id):
    try:
        file = fs.get(ObjectId(image_id))
        return file.read()
    except:
        return None


"""
def get_image_from_gridfs(image_id: str) -> bytes | None:
    """
    Fetch image bytes from GridFS using ObjectId.
    Returns None if not found.
    """
    try:
        file = fs.get(ObjectId(image_id))
        return file.read()
    except:
        return None

def get_all_predictions_from_db():
    predictions = list(db.predictions.find().sort("timestamp", -1))
    result = []
    for p in predictions:
        result.append({
            "species": p.get("species", "unknown"),
            "sex": p.get("sex", "unknown"),
            "species_confidence": p.get("species_confidence", 0),
            "sex_confidence": p.get("sex_confidence", 0),
            "original_image_id": str(p.get("original_image_id")),
            "gradcam_image_id": str(p.get("gradcam_image_id")),
            "timestamp": p.get("timestamp").isoformat() if p.get("timestamp") else None
        })
    return jsonify(result)