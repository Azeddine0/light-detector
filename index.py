import cv2
import numpy as np

def detect_light_sources(frame, threshold=200):
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to isolate bright spots
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the bright spots
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the centers of the contours
    light_sources = []
    for contour in contours:
        if cv2.contourArea(contour) > 5:  # Filter small areas
            (x, y), radius = cv2.minEnclosingCircle(contour)
            light_sources.append((int(x), int(y)))

    return light_sources

def main():
    # Open a connection to the camera
    camera = cv2.VideoCapture(0)  

    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to quit.")

    while True:
        # Capture a frame
        ret, frame = camera.read()

        if not ret:
            print("Failed to grab a frame.")
            break

        # Detect light sources in the frame
        light_sources = detect_light_sources(frame)

        # Draw circles and lines between detected light sources
        for (x, y) in light_sources:
            # Draw a circle around the light source
            cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
            cv2.putText(frame, f"({x}, {y})", (x + 15, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw lines between all pairs of light sources
        for i in range(len(light_sources)):
            for j in range(i + 1, len(light_sources)):
                cv2.line(frame, light_sources[i], light_sources[j], (255, 0, 0), 2)

        # Show the frame with detected light sources and lines
        cv2.imshow("Light Points Detector with Lines", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

