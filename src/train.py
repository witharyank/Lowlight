from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="C:/Users/krary/OneDrive/Desktop/datasets/coco128_enhanced/coco128_enhanced.yaml",
        epochs=40,
        imgsz=640
    )


if __name__ == "__main__":
    main()
