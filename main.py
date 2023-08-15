import cv2

from contour import get_contours, get_biggest_contour, compare_contours
from frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours


def on_trackbar(val):
    pass

def main():

    # definción devariables
    max_value = 255
    threshold_trackbar_name = "threshold"
    denoise_trackbar_name = "denoise"
    window_name = 'Window'
    color_green = (0, 255, 0)

    # creación de la ventana
    cv2.namedWindow(window_name)

    # creación de la trackbar para el threshold y el denoise
    cv2.createTrackbar(threshold_trackbar_name, window_name, 0, max_value, on_trackbar)
    cv2.createTrackbar(denoise_trackbar_name, window_name, 0, max_value, on_trackbar)

    # captura el video de la WebCam
    webcam = cv2.VideoCapture(0)

    # como VideoCapture devuelve un boolean me puedo asegurar de que este capturando correctamente
    while True:
        _, frame = webcam.read() # repetir para refrescar la imagen

        # paso a formato monocromatico
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # consigo el valor des las trackbar
        threshold_value = cv2.getTrackbarPos(threshold_trackbar_name , window_name)
        denoise_value = cv2.getTrackbarPos(denoise_trackbar_name , window_name)

        # paso a formato binario
        _, tresh = cv2.threshold(gray, threshold_value, max_value, cv2.THRESH_BINARY )

        # saco el ruido
        noiseless = cv2.fastNlMeansDenoising(tresh, None, denoise_value, 7, 21)
        
        contours = get_contours(noiseless, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 2000:
                draw_contours(frame, contour, color_green, 3)
                approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
            
                x = approx.ravel() [0]
                y = approx.ravel() [1]

                if len(approx) == 3:
                    cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspectRatio = float(w)/h
                    if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                        cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                    else:
                        cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif len(approx) == 5:
                    cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif len(approx) == 10:
                    cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                else:
                    cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)


        # muestro la imagen
        cv2.imshow('color', frame)
        cv2.imshow(window_name, noiseless)




        tecla = cv2.waitKey(30)  # espera 30 ms para que se presione una tecla . El mínimo es 1 ms.  tecla == 0 si no se pulsó ninguna.
        if tecla == 27:	# tecla ESC para salir
            break



main()