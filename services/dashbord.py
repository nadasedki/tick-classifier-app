from flask import Blueprint, jsonify
from services.db_service import get_all_predictions_from_db

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/api/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """
    Returns JSON with counts and metrics for frontend Chart.js
    """
    print("Fetching dashboard data...")
    # Get predictions from DB (already returns jsonify(result))
    predictions_json = get_all_predictions_from_db().json  # list of dicts

    # Compute species distribution
    species_count = {}
    species_conf = {}
    for p in predictions_json:
        sp = p.get("species", "unknown")
        species_count[sp] = species_count.get(sp, 0) + 1

        # Collect species confidence
        species_conf.setdefault(sp, [])
        species_conf[sp].append(p.get("species_confidence", 0))

    # Compute average confidence per species
    avg_conf = {k: sum(v)/len(v) if len(v) > 0 else 0 for k, v in species_conf.items()}
    print(avg_conf)
    return jsonify({
        "species_count": species_count,
        "species_avg_conf": avg_conf
    })
