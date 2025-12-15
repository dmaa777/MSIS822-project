"""
Model persistence utilities.

Note: `save_all_models` function body is copied verbatim from the notebook `phase 3 and 4 updated.ipynb`.
"""

import os
import joblib
from tensorflow.keras.models import Model as KerasModel

def save_all_models(models_dict, save_dir="models"):
    """
    Saves all ML/DL models to disk based on their type.

    Parameters:
    -----------
    models_dict : dict
        Example:
            {
                "logistic_regression": log_reg_model,
                "svm": svm_model,
                "random_forest": rf_model,
                "xgboost": xgb_model,
                "ffnn": ffnn_model
            }

    save_dir : str
        Directory where models will be saved.
    """

    # Create save folder
    os.makedirs(save_dir, exist_ok=True)

    for model_name, model_obj in models_dict.items():

        # Case 1 — Keras deep learning model
        if isinstance(model_obj, KerasModel):
            file_path = os.path.join(save_dir, f"{model_name}.h5")
            model_obj.save(file_path)
            print(f"[Saved] Keras model → {file_path}")

        # Case 2 — All pickle-compatible models (Sklearn, XGBoost)
        else:
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model_obj, file_path)
            print(f"[Saved] Pickle model → {file_path}")

    print("\nAll models saved successfully!")
