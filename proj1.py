import cv2
from contour import get_contours, get_biggest_contour, compare_contours
from frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours

def on_trackbar(val):
    pass

def main():
    window_name= "Filtered"
    color_name = "Color"

    max_val = 254
    trackbar_name = "Threshold"
    denoised_trackbar = "Denoised"
    distance_trackbar = "Distance"
    color_green = (0,255,0)
    color_red = (0, 0, 255)
    cv2.namedWindow(window_name)
    cv2.namedWindow(color_name)
    cap = cv2.VideoCapture(0)

    cv2.createTrackbar(trackbar_name, window_name, 0, max_val, on_trackbar)
    cv2.createTrackbar(denoised_trackbar, window_name, 0, max_val, on_trackbar)
    cv2.createTrackbar(distance_trackbar, window_name, 0, max_val, on_trackbar)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        threshold_val= cv2.getTrackbarPos(trackbar_name, window_name)
        denoised_val = cv2.getTrackbarPos(denoised_trackbar, window_name) +1
        distance_val = cv2.getTrackbarPos(distance_trackbar, window_name)

        ret1, thresh1 = cv2.threshold(gray_frame,threshold_val,max_val,cv2.THRESH_BINARY)

        denoised = denoise(thresh1, cv2.MORPH_ELLIPSE, denoised_val)

        ret2, thresh2 = cv2.threshold(gray_frame, 0, max_val, cv2.THRESH_BINARY)
        frame1 = denoise(thresh2, cv2.MORPH_ELLIPSE, denoised_val)

        #cv2.imshow(color_name, frame)
        #cv2.imshow(window_name, denoised)


        contours = get_contours(frame=denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        contours_ref = get_contours(frame1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #para matchShapes() uso de referencia el contorno mas externo de la imagen
        # por que pinto la verdad, no se que hacer sino
        draw_contours(frame1, contours_ref, color_green, 3)
        draw_contours(frame, contours_ref, color_green, 3)

        #sabemos que el resultado de este metodo es una lista, entonces la tenemos que recorrer

        for c in contours:
            distance = cv2.matchShapes(contours_ref[0], c,cv2.CONTOURS_MATCH_I2, 0.0)
            if distance < distance_val:
                area = cv2.contourArea(c)
                #calcula el valor del area del contorno encontrado, es el numero de pixeles
                # que es distinto de 0.
                if area > 2000:
                    draw_contours(frame, c, color_green, 3)
                    #dibuja el contorno c que le paso en la imagen de frame con el color_red con un ancho 3
                    approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
                    # cv2.arcLength(c, True) es el perimetro del contorno c, como dice True, considera que es cerrado
                    #cv2.approxPolyDP le paso el contorno, el perimetro y le digo que esta cerrado.
                    #el perimetro*0.01 sirve para medir la precision de la aproximacion.
                    #te devuelve un array de contornos aproximados

                    x = approx.ravel()[0]
                    y = approx.ravel()[1]
                    # cantidad de contornos que saco en funcion de los vertices y la precision dada

                    if len(approx) == 3:
                        cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.5, color_red)

                    elif len(approx) == 4:
                        x, y, w, h = cv2.boundingRect(approx)
                        #es para dibujar un rectangulo aproximado en la imagen
                        aspectRatio = float(w) / h # para ver si es rectangulo o cuadrado
                        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
                            cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                        else:
                                cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                    elif len(approx) == 5:
                        cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                    elif len(approx) == 10:
                        cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                    elif 11< len(approx)< 50:
                        cv2.putText(frame, "Unkown", (x, y),cv2.FONT_HERSHEY_COMPLEX, 0.5, color_red)
                    else:
                        cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)

        cv2.imshow(color_name, frame)
        cv2.imshow(window_name, denoised)
        if cv2.waitKey(1) & 0xFF == ord('m'):
            break



main()