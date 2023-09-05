import cv2
import mediapipe as mp

def init_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(min_detection_confidence = 0.2)

def load_image(file_path, dimensions=(50, 50)):
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"No image found from: {file_path}")
    return cv2.resize(image, dimensions)

def get_eye_avg_position(face_landmarks, width, height):
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    return int((left_eye.x + right_eye.x) / 2 * width), int((left_eye.y + right_eye.y) / 2 * height)

def update_position(avg_eye_x, avg_eye_y, box_x, box_y, width, height, overlay_width, overlay_height):
    if avg_eye_x < width * 0.45 and box_x - overlay_width // 2 > 0:
        box_x -= 5
    elif avg_eye_x > width * 0.55 and box_x + overlay_width // 2 < width:
        box_x += 5

    if avg_eye_y > height * 0.55 and box_y + overlay_height // 2 < height:
        box_y += 5
    elif avg_eye_y < height * 0.45 and box_y - overlay_height // 2 > 0:
        box_y -= 5

    return box_x, box_y

def main():
    face_mesh = init_face_mesh()
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    box_x, box_y = width // 2, height // 2
    overlay_img = load_image('Ghost.png')

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                avg_eye_x, avg_eye_y = get_eye_avg_position(face_landmarks, width, height)
                box_x, box_y = update_position(avg_eye_x, avg_eye_y, box_x, box_y, width, height, overlay_img.shape[1], overlay_img.shape[0])

        start_x = max(0, box_x - overlay_img.shape[1] // 2)
        start_y = max(0, box_y - overlay_img.shape[0] // 2)
        end_x = min(image.shape[1], start_x + overlay_img.shape[1])
        end_y = min(image.shape[0], start_y + overlay_img.shape[0])

        if overlay_img.shape[2] == 4:
            alpha_s = overlay_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                image[start_y:end_y, start_x:end_x, c] = (alpha_s * overlay_img[:, :, c] +
                                                          alpha_l * image[start_y:end_y, start_x:end_x, c])
        else:
            image[start_y:end_y, start_x:end_x] = overlay_img

        cv2.imshow('Game', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
