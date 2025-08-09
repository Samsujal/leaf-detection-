from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
import sys

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

def initialize_camera():
    """Initialize camera with multiple backend support and error handling."""
    
    # Try different camera backends and indices
    camera_configs = [
        (0, cv2.CAP_DSHOW),    # DirectShow backend for Windows
        (0, cv2.CAP_MSMF),     # MSMF backend (default)
        (1, cv2.CAP_DSHOW),    # Camera index 1 with DirectShow
        (1, cv2.CAP_MSMF),     # Camera index 1 with MSMF
        (-1, cv2.CAP_DSHOW),   # Any available camera with DirectShow
    ]
    
    for camera_index, backend in camera_configs:
        print(f"Trying camera index {camera_index} with backend {backend}...")
        camera = cv2.VideoCapture(camera_index, backend)
        
        if camera.isOpened():
            # Test if we can actually read a frame
            ret, frame = camera.read()
            if ret and frame is not None:
                print(f"Successfully initialized camera {camera_index} with backend {backend}")
                return camera
            else:
                print(f"Camera {camera_index} opened but cannot read frames")
                camera.release()
    
    return None

# Initialize camera
camera = initialize_camera()

if camera is None:
    print("\nError: Could not initialize any camera!")
    print("\nTroubleshooting steps:")
    print("1. Check Windows Camera permissions:")
    print("   - Go to Settings > Privacy > Camera")
    print("   - Ensure 'Allow apps to access your camera' is ON")
    print("   - Ensure Python has camera access")
    print("2. Close any other applications using the camera")
    print("3. Check if camera is working in Windows Camera app")
    print("4. Update camera drivers")
    print("5. Try restarting your computer")
    
    # Ask user if they want to use image mode instead
    choice = input("\nWould you like to use image file mode instead? (y/n): ").lower()
    if choice == 'y':
        image_path = input("Enter path to leaf image: ")
        try:
            image = cv2.imread(image_path)
            if image is None:
                print("Error: Could not load image")
                sys.exit(1)
            print("Image loaded successfully. Processing...")
            
            # Process the image directly
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            
            prediction = model.predict(image)
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = prediction[0][index]
            
            print("Class:", class_name[2:], end="")
            print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
            sys.exit(0)
        except Exception as e:
            print(f"Error processing image: {e}")
            sys.exit(1)
    else:
        sys.exit(1)

print("Camera opened successfully. Press ESC to exit.")

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()
    
    if not ret:
        print("Error: Could not read frame from camera")
        break

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the models input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
