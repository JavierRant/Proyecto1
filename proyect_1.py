import cv2
from contour import get_contours, get_biggest_contour, compare_contours
from frame_editor import apply_color_convertion, adaptive_threshold, denoise, draw_contours


def on_trackbar(val):
    pass


def main():
    window_name = "Filtered"
    color_name = "Color"

    max_val = 254
    trackbar_name = "Threshold"
    denoised_trackbar = "Denoised"
    distance_trackbar = "Distance"
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    cv2.namedWindow(window_name)
    cv2.namedWindow(color_name)
    cap = cv2.VideoCapture(0)

    images = ["../static/images/circle.jpg", "../static/images/triangle.jpg", "../static/images/square.jpg", "../static/images/star.jpg", "../static/images/rectangle.jpg", "../static/images/pentagon.jpg"]
            # circle, triangle, square, star, rectangle, pentagon
    thresh_objects = []

    img_gray = cv2.cvtColor(cv2.imread(images[0]), cv2.COLOR_BGR2GRAY)
    a, thresh_im = cv2.threshold((img_gray), 0, 255, cv2.THRESH_BINARY)
    thresh_objects.append(thresh_im)

    i = 1
    while i < len(images):
        img_gray = cv2.cvtColor(cv2.imread(images[i]), cv2.COLOR_BGR2GRAY)
        a, thresh_im = cv2.threshold((img_gray), 0, 255, cv2.THRESH_BINARY)
        thresh_objects.append(thresh_im)
        i += 1

    cont_object = []
    cont_object.append(cv2.findContours(thresh_objects[0], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))
    j = 1
    while j < len(cont_object):
        cont_object.append(cv2.findContours(thresh_objects[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))

    cv2.createTrackbar(trackbar_name, window_name, 0, max_val, on_trackbar)
    cv2.createTrackbar(denoised_trackbar, window_name, 0, max_val, on_trackbar)
    cv2.createTrackbar(distance_trackbar, window_name, 0, 100, on_trackbar)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        threshold_val = cv2.getTrackbarPos(trackbar_name, window_name)
        denoised_val = cv2.getTrackbarPos(denoised_trackbar, window_name) + 1
        distance_val = cv2.getTrackbarPos(distance_trackbar, window_name)

        ret1, thresh1 = cv2.threshold(gray_frame, threshold_val, max_val, cv2.THRESH_BINARY)

        denoised = denoise(thresh1, cv2.MORPH_ELLIPSE, denoised_val)

        contours =  get_contours(frame=denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        # sabemos que el resultado de este metodo es una lista, entonces la tenemos que recorrer

        for c in contours:

            approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            distances = []
            distances.append(cv2.matchShapes(cont_object[0][0][0], c, cv2.CONTOURS_MATCH_I1, 0.0))
            k = 1
            while k < len(cont_object):
                distances.append(cv2.matchShapes(cont_object[k][0][0], c, cv2.CONTOURS_MATCH_I1, 0.0))
                k += 1

            # circle, triangle, square, star, rectangle, pentagon

            if max(distances) <= 0.10:
                cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_red)
            elif distances.index(max(distances)) == 0 and max(distances)*100  <= distance_val:
                cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
            elif distances.index(max(distances)) == 1 and max(distances)*100 <= distance_val:
                cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
            elif distances.index(max(distances)) == 2 and max(distances)*100  <= distance_val:
                cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
            elif distances.index(max(distances)) == 3 and max(distances)*100  <= distance_val:
                cv2.putText(frame, "Star", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
            elif distances.index(max(distances)) == 4 and max(distances)*100  <= distance_val:
                cv2.putText(frame, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
            elif distances.index(max(distances)) == 5 and max(distances)*100 <= distance_val:
                cv2.putText(frame, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_green)
            else:
                cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, color_red)


            draw_contours(frame, c, color_green, 3)


        cv2.imshow(color_name, frame)
        cv2.imshow(window_name, denoised)
        if cv2.waitKey(1) & 0xFF == ord('m'):
            break


main()