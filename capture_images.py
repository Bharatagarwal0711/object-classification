import cv2
import os

# Dataset ka main folder
DATASET_PATH = "Dataset"

# 10 objects ke naam
objects = [
    "watch",
    "fork",
    "glasses",
    "pen",
    "stapler",
    "notebook",
    "remote",
    "bottle",
    "mouse",    
    "cup",
    "headphones"
]

# Har object ki kitni images leni hain
IMAGES_PER_OBJECT = 50

# Webcam open karna
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam open nahi ho raha!")
    exit()

# Har object ke liye images capture karna
for object_name in objects:

    # Object ka folder automatically create karna
    object_folder = os.path.join(DATASET_PATH, object_name)
    os.makedirs(object_folder, exist_ok=True)

    count = 0

    print("\nCurrent object:", object_name)
    print("SPACE = Capture image | Q = Exit")

    while count < IMAGES_PER_OBJECT:

        success, frame = camera.read()

        if not success:
            print("Error: Image capture nahi hui.")
            break

        # Screen par instructions dikhana
        display_frame = frame.copy()

        cv2.putText(
            display_frame,
            f"Object: {object_name}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            f"Images: {count}/{IMAGES_PER_OBJECT}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display_frame,
            "SPACE: Capture | Q: Quit",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow("Object Dataset Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # SPACE press karne par image save karna
        if key == ord(' '):

            image_name = f"image_{count + 1:02d}.jpg"

            image_path = os.path.join(
                object_folder,
                image_name
            )

            cv2.imwrite(image_path, frame)

            count += 1

            print(
                f"{object_name}: "
                f"{count}/{IMAGES_PER_OBJECT} images captured"
            )

        # Q press karne par program band karna
        elif key == ord('q'):
            print("Image collection stop kar di gayi.")
            camera.release()
            cv2.destroyAllWindows()
            exit()

    print(f"{object_name} ki image collection complete!")

# Camera band karna
camera.release()
cv2.destroyAllWindows()

print("\nDataset collection complete!")
print("Total expected images: 500")
