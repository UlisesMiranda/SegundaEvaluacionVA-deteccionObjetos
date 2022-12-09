import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kmeans as kf
import libreriaFiltros as lf

def extraerObjetoPorColor(imgAgrupadaColores, color):
    
    filas, columnas, c = imgAgrupadaColores.shape
    
    nuevaImagen = np.zeros((filas, columnas, c), dtype=np.uint8)
    
    for i in range(filas):
        for j in range(columnas):
            if np.array_equal(imgAgrupadaColores[i][j], color):
                nuevaImagen[i][j] = imgAgrupadaColores[i][j]
                
    return nuevaImagen

def separarJitomates(contornosJitomates, imagen):
    objetosMatricesList = [] 
    
    for i in range(len(contornosJitomates)):
        matriz = np.zeros(imagen.shape)
           
        for listaCoords in contornosJitomates[i]:
            coords = listaCoords[0]
            x = coords[0]
            y = coords[1]
            
            matriz[x][y] = 255
        
                
        objetosMatricesList.append(matriz)
    
    return objetosMatricesList   

def obtenerDistancia(m1, m2):
    
    distancia = math.sqrt((m1[0] - m2[0])**2 + (m1[1] - m2[1])**2)

    return distancia

def obtenerPuntoMedio(l1, l2):
    pm =[ (l1[0] + l2[0]) / 2 , (l1[1] + l2[1]) / 2 ]
    
    return pm

random.seed(0)

porcentaje = 0.15

imagen = cv2.imread(f'cv2-resize-image-{porcentaje}.png')
cv2.imshow("Imagen Original", imagen)

imagenAgrupadaColores, centroides = lf.agrupamientoPorColoresKmeans(imagen, 4)
cv2.imshow("Imagen agrupada colores kmeans", imagenAgrupadaColores)
cv2.imwrite("ImagenAgrupadaColores.png", imagenAgrupadaColores)

jitomates = extraerObjetoPorColor(imagenAgrupadaColores, np.array(centroides[0], dtype=np.uint8))
cv2.imshow("Jitomates", jitomates)
cv2.imwrite("JitomatesSeparados.png", jitomates)

gauss = cv2.GaussianBlur(jitomates, (5,5), 0)
cv2.imshow("suavizado", gauss)
cv2.imwrite("SuavizadoGaussiano.png", gauss)

canny = cv2.Canny(gauss, 50, 150)
cv2.imshow("canny", canny)
cv2.imwrite("Canny.png", canny)

(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
""" 
para 5%
jito2 = c[2]
jito4 = c[0]

Para 15%
jito2 = [2]
jito4 = [0]

Para 16%
jito2 = c[3]
jito4 = c[1]
"""

if porcentaje == 0.15:
    jitomate2_contornos = contornos[2]
    jitomate4_contornos = contornos[0]
elif porcentaje == 0.05:
    jitomate2_contornos = contornos[2]
    jitomate4_contornos = contornos[0]

# Para jitomate 2
jitomate2_img = imagen.copy()
area = cv2.contourArea(jitomate2_contornos)
(x, y, w, h) = cv2.boundingRect(jitomate2_contornos)
#cv2.rectangle(jitomate2_img, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
puntosJito2 = [[int(x), int(y + h/2)], [int(x + w), int(y + h/2)]]
cv2.line(jitomate2_img,(puntosJito2[0][0], puntosJito2[0][1]), (puntosJito2[1][0], puntosJito2[1][1]),(255,0,0), 1)
distanciaJito2 = obtenerDistancia(puntosJito2[0], puntosJito2[1])

print("\nCoordenadas distancias jitomate 2: ", puntosJito2)
print("Distancia = ", distanciaJito2)
cv2.imshow("Jitomate 2", jitomate2_img)
cv2.imwrite("jitomate2.png", jitomate2_img)
    
#Para jitomate 4
jitomate4_img = imagen.copy()
rect = cv2.minAreaRect(jitomate4_contornos)
box = cv2.boxPoints(rect)
box = np.int0(box)
#cv2.drawContours(jitomate4_img,[box],0,(0,255,0),1)
esquina1, esquina2, esquina3, esquina4 = box
p1 = obtenerPuntoMedio(esquina1, esquina4)
p2 = obtenerPuntoMedio(esquina2, esquina3)
puntosJito4 = [p1, p2]
distanciaJito4 = obtenerDistancia(puntosJito4[0], puntosJito4[1])
cv2.line(jitomate4_img, (int(puntosJito4[0][0]), int(puntosJito4[0][1])), (int(puntosJito4[1][0]), int(puntosJito4[1][1])), (255,0,0), 1)

print("\nCoordenadas distancias jitomate 4: ", puntosJito4)
print("Distancia = ", distanciaJito4)
cv2.imshow("Jitomate 4",jitomate4_img)
cv2.imwrite("jitomate4.png", jitomate4_img)

cv2.waitKey(0)
cv2.destroyAllWindows()