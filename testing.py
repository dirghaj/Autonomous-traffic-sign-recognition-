import cv2

# Load trained model
model = tf.keras.models.load_model("traffic_sign_classifier.h5")

# Load labels (class names) for traffic signs
sign_names = pd.read_csv("signnames.csv")  # The sign names file maps the class indexes to sign labels

# Open webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess the frame
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    img = np.expand_dims(img, axis=0) / 255.0

    # Predict traffic sign
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    class_label = sign_names.iloc[class_idx]['SignName']
    
    # Display predicted class on the frame
    cv2.putText(frame, f"Prediction: {class_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame
    cv2.imshow('Traffic Sign Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
