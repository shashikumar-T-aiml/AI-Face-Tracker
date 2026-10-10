import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

print("Starting camera... Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    success, frame = cap.read()
    if not success:
        break

    # Convert to grayscale (AI models process grayscale images faster)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a blue rectangle around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Display the video feed in a popup window
    cv2.imshow('Live AI Face Tracker', frame)

    # Quit safely when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release hardware resources
cap.release()
cv2.destroyAllWindows()