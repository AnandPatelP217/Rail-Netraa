import time

from ultralytics import YOLO
import cv2
import cvzone
import math


cap = cv2.VideoCapture("peoples1.mp4")  # Input video file
model = YOLO("peoples.pt")

ClassNames = ['people']
myColor = (0, 255, 0)

alert_sent = False
alert_threshold = 10  # The number of people to trigger an alert
people_count = 0

# the image capture interval time (6 seconds)
image_capture_interval = 30

# last saved time gap
last_save_time = time.time()

# where you want to save detected images
save_directory = "detected_images"

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = ClassNames[cls]

            if currentClass == 'people':
                myColor = (0, 255, 0)
                people_count += 1  # Increment the people count for each detection
            else:
                myColor = (255, 0, 0)
                # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)

                # Display class name
            cv2.putText(img, "", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, myColor, 1)

    current_time = time.time()
    time_difference = current_time - last_save_time

    if time_difference >= image_capture_interval:
        # alert_sent set to  be  false again after a interval
        alert_sent = False

    cv2.putText(img, f"Current People Count: {people_count}", (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6, myColor, 1)

    # Check if the people count exceeds the alert threshold
    if people_count > alert_threshold and not alert_sent:
        # Send an alert here, e.g., through a notification or email
        print(f"Alert: More than {alert_threshold} people detected!")
        # Save the frame(image) with "Employee not working" class
        frame_name = f"{currentClass}_{current_time}.jpg"
        cv2.imwrite(f"{save_directory}/{frame_name}", img)

        # Print a message
        print(f"Saved image: {frame_name}")
        # Update the last saved timestamp
        last_save_time = current_time

        alert_sent = True  # Set the alert_sent flag to avoid continuous alerts



    people_count = 0  # reset people count after sending alert

    # exit the code a
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

    cv2.imshow("Image", img)
    cv2.waitKey(1)


