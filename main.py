# importar bibliotecas
import cv2 as cv
import numpy as np
import matplotlib
from scipy import ndimage
from skimage import io, color, measure

# definicion de variables
maxValue = 100
minValue = 0
windowName1 = 'levadura'
windowName2 = 'mapa de color'
trackbarName = 'umbral'

img1 = cv.imread('../static/images/levadura.png')
img = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# creación de la función recursiva que llama la funcion createTrackbar()
def segementation(trackbarValue):
    _, thresh = cv.threshold(img, trackbarValue, maxValue, cv.THRESH_BINARY)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=7)

    sure_bg = cv.dilate(opening, kernel, iterations=2) # Esto es background (sure background)

    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
    _, sure_fg = cv.threshold(dist_transform, 0.01*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg) # Estas son celulas (sure foreground)

    unknown = cv.subtract(sure_bg, sure_fg) # pixeles q no sabemos q son exactamente
    
    # individualizacion de las celulas
    num, markers = cv.connectedComponents(sure_fg)
    # aprovechamos el el hecho de que la funcion connectedComponentes() nos
    # devuelva la cantidad de etiquetas para coontar las celulas
    strNUM = str(num-1) # para que no cuente el fondo
    cantCelulas = 'cant celulas:'+ strNUM

    markers = markers+1

    markers[unknown == 255] = 0

    markers = cv.watershed(img1, markers)

    img1[markers == -1] = [0, 255, 255]

    labelIMG = np.uint8(255 * markers / num) # convertir los labels en mapa de semillas

    colorMap = cv.applyColorMap(labelIMG, cv.COLORMAP_JET) # crea el mapa de marcadores

    cv.putText(thresh, cantCelulas, (30, 700), cv.FONT_HERSHEY_SIMPLEX , 1, 255)  # muestra la imagen actualizada con los valores de threshold

    clusters = measure.regionprops(label_image=labelIMG, intensity_image=img1) # para medir el area

    cv.imshow(windowName1, thresh)
    cv.imshow(windowName2, colorMap)
    # cv.imshow("Opening", opening)
    # cv.imshow("Sure background", sure_bg)
    # cv.imshow("Sure foreground", sure_fg)
    # cv.imshow("Unkown", unknown)

# crea el trackbar y lo asocia a la imagen
cv.namedWindow(windowName1)

cv.createTrackbar(trackbarName, windowName1, minValue, maxValue, segementation)

segementation(0)

cv.waitKey(0)