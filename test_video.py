import cv2
from ultralytics import YOLO

# Load the trained model
model = YOLO("/Users/joshfeather/Documents/cbod/final.pt")  # Update with the path to your trained model weights

# Open video file
video_path = "/Users/joshfeather/Documents/cbod/cricket-ball-videos/T20 cricket in slow-motion! _ Incredible moments from the Vitality Blast at The Kia Oval.mp4"  # Path to your input video
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform prediction on the frame
    results = model.predict(frame, device='mps')  # Use 'mps' for Apple Silicon or 'cuda' for NVIDIA GPUs

    # Process each result in the list
    for result in results:
        # Annotate the frame using the plot method
        annotated_frame = result.plot()  # Annotates the frame with bounding boxes and labels

    # Write the frame with annotations
    out.write(annotated_frame)

    # Display the resulting frame
    cv2.imshow('Frame', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()
