import cv2
from ultralytics import YOLO
import numpy as np
import time
import argparse
import os

def real_time_demo(video_path, model, window_name="Comparison", model_name=None, save=None):
    cap = cv2.VideoCapture(video_path)
    
    running_mean_processing_time = 0
    
    # Getting video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if save:
        output_path = f"{save}/{model_name}_{os.path.basename(video_path)}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        process_start = time.time()
        results = model(frame, device='cpu', verbose=False, imgsz=640)
        annotated_frame = results[0].plot()
        process_end = time.time()
        
        process_time = process_end - process_start
        
        running_mean_processing_time = (running_mean_processing_time * (frame_count - 1) + process_time) / frame_count
        
        # Adding info to frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        color = (128, 0, 128) 
        thickness = 2
        line_spacing = 30

        cv2.putText(annotated_frame, f"Model: {model_name}", (10, 30), font, font_scale, color, thickness)
        cv2.putText(annotated_frame, f"Running Mean Processing Time: {running_mean_processing_time*1000:.2f} ms", (10, 30 + line_spacing), font, font_scale, color, thickness)
        cv2.putText(annotated_frame, f"Video FPS: {video_fps:.2f}", (10, 30 + 2*line_spacing), font, font_scale, color, thickness)
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", (10, 30 + 3*line_spacing), font, font_scale, color, thickness)
        
        cv2.imshow(window_name, annotated_frame)
        
        if save:
            out.write(annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()
    
    return running_mean_processing_time*1000

def main():
    parser = argparse.ArgumentParser(description="Run real-time object detection on a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--model", type=str, default="yolov8n", choices=["yolov8n", "yolov8n_finetuned", "yolov8n_finetuned_onnx"], help="Model to use for detection")
    parser.add_argument("--weights_path", type=str, default="../../models/", help="Path to the directory containing model weights")
    parser.add_argument("--save", type=str, help="Path to save the output video (optional)")

    args = parser.parse_args()

    if args.model == "yolov8n":
        model = YOLO(os.path.join(args.weights_path, "yolov8n.pt"))
        model_name = "Regular YOLOv8n"
    elif args.model == "yolov8n_finetuned":
        model = YOLO(os.path.join(args.weights_path, "yolov8n_finetuned_visdrone_19_epochs.pt"))
        model_name = "Finetuned YOLOv8n"
    elif args.model == "yolov8n_finetuned_onnx":
        model = YOLO(os.path.join(args.weights_path, "yolov8n_finetuned_visdrone_19_epochs.onnx"))
        model_name = "Finetuned YOLOv8n ONNX"

    mean_processing_time = real_time_demo(args.video_path, model, window_name=model_name, model_name=model_name, save=args.save)

    print(f"Average processing time: {mean_processing_time:.2f} ms")

if __name__ == "__main__":
    main()