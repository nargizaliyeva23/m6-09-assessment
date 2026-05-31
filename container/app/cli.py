import sys
import json
import os
from app.detector import CatDetector

def main():
    if len(sys.argv) < 2:
        print("Usage: info | predict")
        return

    command = sys.argv[1]

    # ---------- INFO ----------
    if command == "info":
        with open("/app/STUDENT.json", "r") as f:
            data = json.load(f)
        print(json.dumps(data))
        return

    # ---------- PREDICT ----------
    if command == "predict":
        input_dir = "/data/input"
        output_file = "/data/output/predictions.csv"

        model = CatDetector("/app/models/best.onnx")

        os.makedirs("/data/output", exist_ok=True)

        with open(output_file, "w") as f:
            f.write("image_path,xmin,ymin,xmax,ymax,confidence,class\n")

            for root, _, files in os.walk(input_dir):
                for file in files:
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img_path = os.path.join(root, file)
                        rel_path = os.path.relpath(img_path, input_dir).replace("\\", "/")

                        preds = model.predict(img_path)

                        if len(preds) == 0:
                            f.write(f"{rel_path},,,,,,\n")
                        else:
                            for p in preds:
                                f.write(
                                    f"{rel_path},"
                                    f"{p['xmin']},{p['ymin']},"
                                    f"{p['xmax']},{p['ymax']},"
                                    f"{p['confidence']},"
                                    f"{p['class']}\n"
                                )

        print("Prediction saved to", output_file)
        return

if __name__ == "__main__":
    main()