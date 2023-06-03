import cv2
import mediapipe as mp 

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_drawing_styles = mp.solutions.drawing_styles

with mp_face_mesh.FaceMesh(
    static_image_mode = False,
    max_num_faces = 1,
    min_detection_confidence=0.5,
    refine_landmarks = True
    ) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        image.flags.writeable = False
        image = cv2.flip(image,1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable= True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        if results.multi_face_landmarks is not None:
            for face_lambdarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    face_lambdarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(255,0,0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(150,150,150), thickness=1)
                )
        cv2.imshow("Frame", image)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()