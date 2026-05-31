import numpy as np
import onnxruntime as ort
from PIL import Image

class CatDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path, conf=0.25):
        img = Image.open(image_path).convert("RGB")
        orig_w, orig_h = img.size

        img = img.resize((640, 640))
        x = np.array(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]

        outputs = self.session.run(None, {self.input_name: x})[0][0]

        results = []

        for x1, y1, x2, y2, score, cls in outputs:
            if score < conf:
                continue

            # scale back
            x1 = x1 * orig_w / 640
            y1 = y1 * orig_h / 640
            x2 = x2 * orig_w / 640
            y2 = y2 * orig_h / 640

            results.append({
                "xmin": float(x1),
                "ymin": float(y1),
                "xmax": float(x2),
                "ymax": float(y2),
                "confidence": float(score),
                "class": "cat"
            })

        return results