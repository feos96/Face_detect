import cv2
import mediapipe as mp

video = cv2.VideoCapture(0)

solucao_reconhecedor_rosto = mp.solutions.face_detection

reconhecedor_rostos = solucao_reconhecedor_rosto.FaceDetection()

desenho = mp.solutions.drawing_utils

while True:
  verificador, frame = video.read()
  if not verificador:
    break

  lista_rostos = reconhecedor_rostos.process(frame)

  if lista_rostos.detections:
    for rosto in lista_rostos.detections:
      desenho.draw_detection(frame, rosto)

  cv2.imshow("Rostos da WebCam", frame)
  if cv2.waitKey(5) == 27:
    break

video.release()
cv2.destroyAllWindows()