import math
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import kmeans as kf
import libreriaFiltros as lf

def extraerObjetoPorColor(imgAgrupadaColores, color):
    
    """ 
    Iteramos sobre nuestra imagen y buscamos los pixeles donde cuya
    intensidad coincida con el color que buscamos y retornamos una
    imagen con solo esos objetos
    """
    
    filas, columnas, c = imgAgrupadaColores.shape
    nuevaImagen = np.zeros((filas, columnas, c), dtype=np.uint8)
    
    for i in range(filas):
        for j in range(columnas):
            if np.array_equal(imgAgrupadaColores[i][j], color):
                nuevaImagen[i][j] = imgAgrupadaColores[i][j]
                
    return nuevaImagen   

def obtenerDistancia(m1, m2):
    
    #Distancia entre dos puntos
    distancia = math.sqrt((m1[0] - m2[0])**2 + (m1[1] - m2[1])**2)

    return distancia

def obtenerPuntoMedio(p1, p2):
    
    #Obtenemos el punto medio entre dos puntos
    pm =[ (p1[0] + p2[0]) / 2 , (p1[1] + p2[1]) / 2 ]
    
    return pm

random.seed(0)

""" 
Para poder realizar el preprocesamiento de la imagen, opte primero por redimensionar la 
imagen original al 15%, con el fin de que las operaciones que necesito no sean tan complejas 
y tardadas. Entonces los resultados finales son escalados.
"""
porcentaje = 0.15

imagen = cv2.imread(f'cv2-resize-image-{porcentaje}.png')
cv2.imshow("Imagen Original", imagen)

""" 
1. El primer paso que necesitamos es extraer los jitomates de la imagen. Para esto
obtenemos la imagen agrupada en 4 colores para poder diferenciar los jitomates de las piedras
y el fondo con el metodo de K-means
"""
imagenAgrupadaColores, centroides = lf.agrupamientoPorColoresKmeans(imagen, 4)
cv2.imshow("Imagen agrupada colores kmeans", imagenAgrupadaColores)
cv2.imwrite("ImagenAgrupadaColores.png", imagenAgrupadaColores)

""" 
Una vez que obtenga la imagen agrupada por colores, verificamos el centroide que le pertenecen los
jitomates, es decir, el centroide mas acercado al color rojo y extraemos los objetos que sean de color 
rojo 
"""
jitomates = extraerObjetoPorColor(imagenAgrupadaColores, np.array(centroides[0], dtype=np.uint8))
cv2.imshow("Jitomates", jitomates)
cv2.imwrite("JitomatesSeparados.png", jitomates)

""" 
Ya obtenidos nuestros jitomates, aplicamos un suavizado Gaussiano para poder eliminar un poco del
ruido que traiga consigo, para posteriormente obtener los contornos de los jitomates a traves de
canny 
"""
gauss = cv2.GaussianBlur(jitomates, (5,5), 0)
cv2.imshow("suavizado", gauss)
cv2.imwrite("SuavizadoGaussiano.png", gauss)

canny = cv2.Canny(gauss, 50, 150)
cv2.imshow("canny", canny)
cv2.imwrite("Canny.png", canny)

""" 
Con el metodo findContours obtenemos una lista de las coordenadas de los contornos por objeto 
encontrados en canny.
Como parametros de la funcion tenemos de primera instancia una imagen binarizada,
luego tenemos el modo de contorno, el cual indica el tipo de contorno que necesitamos; en este caso 
utilizamos RETR_EXTERNAL que obtiene el contorno externo de un objeto lo cual nos sirve por
si tenemos algun otro contorno dentro de los jitomates, esto nos ayuda a que solo tomemos los 
contornos de los jitomates; el siguiente parametro es el tipo de aproximacion de nuestros 
contornos, en este caso CHAIN_APPROX_SIMPLE el cual toma todos los puntos mas relevantes de 
los contornos y elimina los menos necesarios
"""

(contornos,_) = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
""" 
Lo siguiente es obtener los contornos que necesitaremos, en este caso, unicamente requerimos de 
los contornos del jitomate 2 y del jitomate 4

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
    
""" 
Para el procesamiento de ambos jitomates, mi idea en general es ubicar un pequeño rectangulo que 
abrace los contornos del jitomate, desde la raiz hasta la punta del jitomate.
Con ello espero obtener los puntos medios de ambos extremos del rectangulo y realizar el calculo de 
las distancias entre puntos y obtener una linea que cruce estos

Para el caso del jitomate 2, necesito de un rectangulo recto sin tomar en consideracion el angulo de
inclinación. Sin embargo, para el caso del jitomate 4, como la figura del jitomate se encuentra rotada 
es por eso que necesito de otro rectangulo con una inclinacion y medidas que se ajustan automaticamente a la menor 
area que abarque los contornos del jitomate; una vez obtenido el rectangulo, verifico las esquinas de este
y obtengo los puntos medios que me permitan obtener una linea que cruce estos puntos y calcular la distancia
"""

#Para jitomate 2
jitomate2_img = imagen.copy()
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