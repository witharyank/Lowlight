from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")

    model.train(
        data="coco128.yaml",
        epochs=30,
        imgsz=640,
        hsv_h=0.1,
        hsv_s=0.7,
        hsv_v=0.7
    )

if __name__ == "__main__":
    train_model()
