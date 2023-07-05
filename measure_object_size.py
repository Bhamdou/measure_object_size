import cv2

# Known width of reference object
WIDTH_OF_REFERENCE_OBJECT = 5.0 # in cm or inches

# Open video stream
cap = cv2.VideoCapture(0)

while True:
    # Read frame from camera
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edged = cv2.Canny(blurred, 50, 100)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Filter out smaller contours based on area
        if cv2.contourArea(contour) < 500:  # Adjust this value as needed
            continue

        # Find the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw the contour and bounding box on the original color image
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display width in pixels on the image
        cv2.putText(frame, f"Width: {w} pixels", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Image", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object and destroy all windows
cap.release()
cv2.destroyAllWindows()
