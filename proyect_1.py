import cv2
from contour import get_contours, get_biggest_contour, compare_contours
from frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours


def on_trackbar(val):
    pass


def main():
    window_name = "Filtered"
    color_name = "Color"
    #nombres de las ventanas

    max_val = 254
    trackbar_name = "Threshold"
    denoised_trackbar = "Denoised"
    distance_trackbar = "Distance"
    #solo hago un trackbar, en lugar de hacer un trackbar para cada formar
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    cv2.namedWindow(window_name)
    cv2.namedWindow(color_name)
    cap = cv2.VideoCapture(0)

    #lista de las imagenes para comparar los contornos, son formas geometricas rellenas
    images = ["../static/images/circle.jpg", "../static/images/triangle.jpg", "../static/images/square.jpg", "../static/images/star.jpg", "../static/images/rectangle.jpg", "../static/images/pentagon.jpg"]
            # circle, triangle, square, star, rectangle, pentagon

    #lista de imagenes binarizadas
    thresh_objects = []
    for i in images:
        img_gray = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
        a, thresh_im = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
        thresh_objects.append(thresh_im)


    #lista de contornos obtenidos de las imagenes binarizadas de las formas
    cont_object = []

    for t in thresh_objects:
        cont_object.append(cv2.findContours(t, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))
    #instancio los trackbars
    cv2.createTrackbar(trackbar_name, window_name, 0, max_val, on_trackbar)
    cv2.createTrackbar(denoised_trackbar, window_name, 0, 50, on_trackbar)
    cv2.createTrackbar(distance_trackbar, window_name, 0, 100, on_trackbar)

    #print(cont_object)
    #ya esta chequeado que son todos distintos
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #obtengo el valor que voy tomando mientras muevo la barrita
        threshold_val = cv2.getTrackbarPos(trackbar_name, window_name)
        denoised_val = cv2.getTrackbarPos(denoised_trackbar, window_name) +1
        distance_val = cv2.getTrackbarPos(distance_trackbar, window_name)

        #obtengo la imagen de la camara en binaria
        ret1, thresh1 = cv2.threshold(gray_frame, threshold_val, max_val, cv2.THRESH_BINARY)

        #le saco el ruido con las funciones hechas por fabri
        denoised = denoise(thresh1, cv2.MORPH_ELLIPSE, denoised_val)

        #obtengo los contornos el get_contours esconde el cv2.findContours()
        contours =  get_contours(frame=denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # sabemos que el resultado de este metodo es una lista, entonces la tenemos que recorrer

        for c in contours:
            #realizo esta función para obtener las coordenadas de mi contorno c
            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            #lista con los valores del 0 al 1 de que tan cerca esta mi contorno con el contorno de las formas geométricas
            #el matchShapes() me da el valor del 0 al 1 que tan parecida es mi contorno con el contorno comparativo
            distance_circle = cv2.matchShapes(cont_object[0][0][1], c, cv2.CONTOURS_MATCH_I1,0)
            distance_triangle = cv2.matchShapes(c, cont_object[1][0][1], cv2.CONTOURS_MATCH_I1,0)
            distance_square = cv2.matchShapes(c, cont_object[2][0][1], cv2.CONTOURS_MATCH_I1,0)
            distance_star = cv2.matchShapes(cont_object[3][0][1], c, cv2.CONTOURS_MATCH_I1,0)
            distance_rectangle = cv2.matchShapes(cont_object[4][0][1], c, cv2.CONTOURS_MATCH_I1,0)
            distance_pentagon = cv2.matchShapes(cont_object[5][0][1], c, cv2.CONTOURS_MATCH_I1,0)

            #cuanto mas chiquito el numero mas parecidos son

            distances = [distance_circle, distance_triangle, distance_square, distance_star, distance_rectangle, distance_pentagon]
            # circle, triangle, square, star, rectangle, pentagon
            #     0,         1,      2,    3,         4,        5
            if max(distances) <= 0.10:
                cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_red)
            elif min(distances)*100 <= distance_val:
                if distances.index(min(distances)) == 1:
                    cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif distances.index(min(distances)) == 0:
                    cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif distances.index(min(distances)) == 1:
                    cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif distances.index(min(distances)) == 2:
                    cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif distances.index(min(distances)) == 3:
                    cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif distances.index(min(distances)) == 4:
                    cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                elif distances.index(min(distances)) == 5:
                    cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
                else:
                    cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_red)


            draw_contours(frame, c, color_green, 3)


        cv2.imshow(color_name, frame)
        cv2.imshow(window_name, denoised)
        if cv2.waitKey(1) & 0xFF == ord('m'):
            break


main()