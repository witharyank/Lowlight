from ultralytics import YOLO

def main():
    model = YOLO("yolov8l.pt")

    model.train(
        data="C:/Users/krary/OneDrive/Desktop/datasets/coco128_enhanced/coco128_enhanced.yaml",
        epochs=40,
        imgsz=640,
        batch=4,
        cos_lr=True,
        patience=10,
        name="yolov8l_enhanced"
    )

if __name__ == "__main__":
    main()
