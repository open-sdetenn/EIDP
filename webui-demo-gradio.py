import gradio as gr
import cv2
import numpy as np
import tempfile
import os
from tqdm import tqdm

from inference import Inference

inference_model = Inference()

def process_video(uploaded_video, batch_size):
    video_capture = cv2.VideoCapture(uploaded_video)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    output_path = "processed_video.mp4"
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'vp90'), fps, (frame_width, frame_height))
    frame_count = 0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc='Processing frames')
    
    frame_buffer = []
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        frame_path = os.path.join('temp', f"frame_{frame_count}.png")
        cv2.imwrite(frame_path, frame)
        frame_buffer.append(frame_path)
        frame_count += 1
        progress_bar.update(1)
        
        if len(frame_buffer) == batch_size or frame_count == total_frames:
            batch_results = [inference_model.inference(frame_path) for frame_path in frame_buffer]  # Using ImageInference instance for inference
            for i, result in enumerate(batch_results):
                frame = cv2.imread(frame_buffer[i])
                overlay = f"Rotation: {float(result['rot']):.4f} | L2: {float(result['l2']):.4f} | R2: {float(result['r2']):.4f}"
                font_scale = 1.5
                font_color = (0, 255, 0)
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 2
                cv2.putText(frame, overlay, (50, 50), font, font_scale, font_color, thickness)
                video_writer.write(frame)
            frame_buffer = []
    
    progress_bar.close()
    video_capture.release()
    video_writer.release()
    return output_path

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## IDM V1 DEMO")
        uploaded_video = gr.Video(label="Upload a video file")
        batch_size = gr.Slider(label="Select Batch Size", minimum=1, maximum=10, value=5, step=1)
        process_button = gr.Button("Process Video")
        output = gr.Video(label="Processed Video")

        process_button.click(process_video, inputs=[uploaded_video, batch_size], outputs=output)

    demo.launch(share=False)

if __name__ == "__main__":
    main()