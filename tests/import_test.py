import importlib
import sys
import os

packages = ['pandas','numpy','sklearn','tensorflow','joblib','flask']

for p in packages:
    try:
        m = importlib.import_module(p)
        v = getattr(m, '__version__', None)
        print(f"{p}: OK (version={v})")
    except Exception as e:
        print(f"{p}: ERROR - {e}")

# TensorFlow details and optional model load
try:
    import tensorflow as tf
    print(f"tensorflow: OK (version={tf.__version__})")
    model_path = os.path.join('model','delivery_time_model.keras')
    if os.path.exists(model_path):
        try:
            from tensorflow.keras.models import load_model
            model = load_model(model_path)
            print(f"Model loaded: {type(model)}")
        except Exception as e:
            print(f"Model load ERROR: {e}")
    else:
        print(f"Model not found at: {model_path}")
except Exception as e:
    print(f"tensorflow import ERROR: {e}")
