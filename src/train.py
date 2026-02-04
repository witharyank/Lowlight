from ultralytics import YOLO

def main():
    model = YOLO("yolov8m.pt")  # upgrade model

    model.train(
        data="C:/Users/krary/OneDrive/Desktop/datasets/coco128_enhanced/coco128_enhanced.yaml",
        epochs=40,
        imgsz=640,
        batch=8  
    )

if __name__ == "__main__":
    main()
