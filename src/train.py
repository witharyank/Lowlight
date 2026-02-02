from ultralytics import YOLO

def train_model():
    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="coco128.yaml",   # change this during hackathon
        epochs=10,
        imgsz=640
    )

if __name__ == "__main__":
    train_model()
