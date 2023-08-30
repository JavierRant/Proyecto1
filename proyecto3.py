"""Segmentación instanciada con Watershed"""

# 1. Segmentar los núcleos con threshold
#   a. ajustar el umbral con trackbar
#   b. las células con dos núcleos son en realidad dos células en proceso de mitosis
#   c. anotar al pie de la imagen la cantidad de células encontradas

# 2. Individualizarlos con componentes conectados
#   a. produce un mapa de semillas
#   b. visualizar con colorMap

# 3. Agregar al menos una semilla para el fondo
#   a. se puede generar un mapa de semilla de fondo con otro threshold, que requiere otro trackbar para el umbral

# importar bibliotecas
import cv2 as cv
import numpy as np

# definicion de variables
maxValue = 100
minValue = 0
windowName1 = 'levadura'
windowName2 = 'mapa de color'
trackbarName = 'umbral'

# carga la imagen en formato monocromático
img = cv.imread('proyectos\levadura.png', cv.IMREAD_GRAYSCALE)


# creación de la función recursiva que llama la funcion createTrackbar()
def segementation(trackbarValue):
    
    # utiliza threshold() para segmentar la imagen en formato binario. Argumentos:
    # 1ro: imagen en escala de grises
    # 2do: valor umbral que se actualiza a partir de la trackbar 
    # 3ro: valor máximo
    # 4to: tipo de método de binarizacón
    _, thresh = cv.threshold(img, trackbarValue, maxValue, cv.THRESH_BINARY) 

    # individualizacion de las celulas
    num, markers = cv.connectedComponents(thresh)

    # aprovechamos el el hecho de que la funcion connectedComponentes() nos devuelva la cantidad de etiquetas para coontar las celulas
    strNUM = str(num-1) #para que no cuente el fondo
    cantCelulas = 'cant celulas:'+ strNUM


    # convertir los labels en mapa de semillas
    labelIMG = np.uint8(255 * markers / num)

    # crea el mapa de marcadores
    colorMap = cv.applyColorMap(labelIMG, cv.COLORMAP_JET)
    
    # muestra la imagen actualizada con los valores de threshold
    cv.putText(thresh, cantCelulas, (50, 50), cv.FONT_HERSHEY_SIMPLEX , 1, 255)
    cv.imshow(windowName1, thresh)
    cv.imshow(windowName2, colorMap)
    

    

# crea el trackbar y lo asocia a la imagen 
cv.namedWindow(windowName1)
# Argumentos:
# 1ro: nombre de la trackbar
# 2do: nombre de la ventana
# 3ro: valor minimo
# 4to: valor maximo
# 5to: funcion recursiva que se llama cada vez que se actualiza el valor de la trackbar
cv.createTrackbar(trackbarName, windowName1, minValue, maxValue, segementation) 

segementation(0)
cv.waitKey(0)