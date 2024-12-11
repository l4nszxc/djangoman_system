# views.py
import cv2
import numpy as np
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse
from .models import RoadSignImage
from .forms import RoadSignForm
import tensorflow as tf
import threading
import os
from django.core.files.base import ContentFile
from django.utils import timezone

# Load the updated model
try:
    model = tf.keras.models.load_model("detector/models/real_road_sign.h5")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
except Exception as e:
    print(f"Error loading model: {e}")

# Updated road sign labels according to your new dataset
road_sign_labels = {
    0: ("10mph", "Speed limit of 10 mph."),
    1: ("20mph", "Speed limit of 20 mph."),
    2: ("30mph", "Speed limit of 30 mph."),
    3: ("40mph", "Speed limit of 40 mph."),
    4: ("crossroads", "Warning of a crossroads ahead."),
    5: ("cycle_route_ahead", "Indicates a cycle route ahead."),
    6: ("frail", "Warning for frail pedestrians."),
    7: ("give way", "Give way to other traffic."),
    8: ("give_way_to_oncoming", "Give way to oncoming traffic."),
    9: ("keep_left", "Keep left."),
    10: ("keep_right", "Keep right."),
    11: ("mini_roundabout", "Indicates a mini roundabout ahead."),
    12: ("no_entry", "No entry for vehicular traffic."),
    13: ("no_left_turn", "No left turn allowed."),
    14: ("no_motor_vehicles", "No motor vehicles allowed."),
    15: ("no_right_turn", "No right turn allowed."),
    16: ("no_through_road", "No through road."),
    17: ("one_way_traffic", "One-way traffic."),
    18: ("road_humps", "Warning of road humps."),
    19: ("road_narrows_on_both_sides", "Road narrows on both sides."),
    20: ("road_narrows_on_left", "Road narrows on the left."),
    21: ("road_narrows_on_right", "Road narrows on the right."),
    22: ("road_works", "Warning of road works ahead."),
    23: ("school_crossing", "Warning of school crossing ahead."),
    24: ("traffic_has_priority", "You have priority over oncoming vehicles."),
    25: ("traffic signals", "Traffic signals ahead."),
    26: ("turn_left", "Turn left."),
    27: ("turn_left_ahead", "Turn left ahead."),
    28: ("turn_right", "Turn right."),
    29: ("two_way_traffic", "Two-way traffic ahead."),
    30: ("zebra_crossing", "Zebra crossing ahead.")
}

def homepage(request):
    return render(request, 'detector/homepage.html')

def preprocess_image(image):
    # Resize and normalize the image
    image_size = (64, 64)
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return np.expand_dims(image, axis=0)

def predict_sign(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    label, description = road_sign_labels.get(class_index, ("Unknown", "No description available."))
    return label, description, confidence

def speak(description):
    import pyttsx3
    engine = pyttsx3.init()
    engine.say(description)
    engine.runAndWait()

def gen_frames():
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise IOError("Cannot open webcam")

    previous_label = None
    square_size = 300
    square_color = (0, 255, 0)
    square_thickness = 2

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            top_left = (width // 2 - square_size // 2, height // 2 - square_size // 2)
            bottom_right = (width // 2 + square_size // 2, height // 2 + square_size // 2)
            cv2.rectangle(frame, top_left, bottom_right, square_color, square_thickness)

            # Crop the region of interest
            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            label, description, confidence = predict_sign(roi)
            if label in [desc[0] for desc in road_sign_labels.values()] and confidence > 0.8:
                cv2.putText(frame, f"{label}: {confidence*100:.2f}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, description, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                if label != previous_label:
                    threading.Thread(target=speak, args=(description,)).start()
                    previous_label = label

                    # Save the detection result
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_content = ContentFile(buffer.tobytes())
                    image_instance = RoadSignImage(
                        source='camera',
                        label=label,
                        description=description,
                        confidence=confidence,
                        detected_at=timezone.now()
                    )
                    image_instance.image.save(f'{label}_{timezone.now().strftime("%Y%m%d_%H%M%S")}.jpg', image_content)
                    image_instance.save()
            else:
                previous_label = None
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed(request):
    return StreamingHttpResponse(gen_frames(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def camera_feed(request):
    return render(request, 'detector/camera_feed.html')

def detect_sign(request):
    if request.method == 'POST':
        form = RoadSignForm(request.POST, request.FILES)
        if form.is_valid():
            image_instance = form.save()
            image_url = image_instance.image.url
            # Read and process the uploaded image
            image_path = image_instance.image.path
            image = cv2.imread(image_path)
            label, description, confidence = predict_sign(image)
            # Text-to-speech output
            threading.Thread(target=speak, args=(description,)).start()
            context = {
                'label': label,
                'description': description,
                'confidence': f"{confidence*100:.2f}%",
                'image_url': image_url,
            }
            return render(request, 'detector/result.html', context)
    else:
        form = RoadSignForm()
    return render(request, 'detector/upload.html', {'form': form})

def upload_image(request):
    return redirect('detect_sign')

def history(request):
    signs = RoadSignImage.objects.all().order_by('-uploaded_at')
    return render(request, 'detector/history.html', {'signs': signs})