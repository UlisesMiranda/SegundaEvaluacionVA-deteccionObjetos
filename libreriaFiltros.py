import math
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt
import kmeans as km
from collections import defaultdict
from sklearn.cluster import KMeans

def convertirEscalaGrisesNTSC(imagen):
    largo, ancho, canales = imagen.shape

    imgEscalaGrises = np.zeros(largo * ancho, dtype=np.float32).reshape(largo, ancho)
    
    for i in range(largo):
        for j in range(ancho):
            pixel = imagen[i][j]
            azul = pixel[0]
            verde = pixel[1]
            rojo = pixel[2]
            
            imgEscalaGrises[i][j] = 0.299 * azul + 0.587 * verde + 0.11 * rojo
            
    return imgEscalaGrises

def crearMatrizRelleno(imagen, mascSize):
    largo, ancho = imagen.shape
    difBordes = mascSize - 1
    bordesSize = int(difBordes / 2)
    
    largoRelleno = largo + difBordes
    anchoRelleno = ancho + difBordes
    
    #matrizRelleno = np.zeros((largoRelleno, anchoRelleno, 1), np.float64)
    matrizRelleno = np.zeros(largoRelleno * anchoRelleno, dtype=np.uint8).reshape(largoRelleno, anchoRelleno)
    
    
    for i in range(bordesSize, largoRelleno - bordesSize):
        for j in range(bordesSize, anchoRelleno - bordesSize ):
            matrizRelleno[i][j] = imagen[i - bordesSize][j - bordesSize]
            
    return matrizRelleno
    

def aplicarFiltro(imagen, matrizRelleno, mascara, mascSize):
    largo, ancho = imagen.shape
    
    imgFiltroAplicado = np.zeros(largo * ancho, dtype=np.float64).reshape(largo, ancho)
    
    for i in range(largo):
        for j in range(ancho):
            # val = abs(convolucionPixel(matrizRelleno, mascara, mascSize, i, j))
            val = convolucionPixel(matrizRelleno, mascara, mascSize, i, j)
            imgFiltroAplicado[i][j] = val
            
    return imgFiltroAplicado
            
def convolucionPixel(matrizRelleno, mascara, mascSize, x, y):
    
    limites = int((mascSize - 1) / 2)
    sumatoriaFiltro = 0.0
    
    for i in range(-limites, limites + 1):
        for j in range(-limites, limites + 1):
            valMascara = mascara[i + limites][j + limites]
            coordY = y + j + limites
            coordX = x + i + limites

            valImagen = matrizRelleno[coordX][coordY]
            
            sumatoriaFiltro += valMascara * valImagen
    
    return sumatoriaFiltro

def mascaraGaussiana(mascSize, sigma):
    limite = int((mascSize - 1) / 2)
    gaussResultado= 0.0
    mascara = np.zeros((mascSize, mascSize), dtype=np.float64)
    sum = 0.0
    
    s = 2.0 * sigma * sigma;
    
    for x in range(-limite, limite + 1):
        for y in range(-limite, limite + 1):
            
            r = math.sqrt(x * x + y * y);
            z = (math.exp(-(r * r) / s)) / (math.pi * s);
            gaussResultado = (math.exp(-(r * r) / s)) / (math.pi * s);
            mascara[x + limite][y + limite] = gaussResultado;
            
            sum += gaussResultado
            
    for i in range(mascSize):
        for j in range(mascSize):
            mascara[i][j] /= sum
              
    return mascara

def mascaraGy():

    #Mascara de Gy
    mascara = np.zeros((3,3), dtype=np.float64)

    mascara[0][0] = -1;
    mascara[0][1] = -2;
    mascara[0][2] = -1;

    mascara[1][0] = 0;
    mascara[1][1] = 0;
    mascara[1][2] = 0;

    mascara[2][0] = 1;
    mascara[2][1] = 2;
    mascara[2][2] = 1;

    return mascara


def mascaraGx() :

    #Mascara de Gx
    mascara = np.zeros((3,3), dtype=np.float64)

    mascara[0][0] = -1;
    mascara[0][1] = 0;
    mascara[0][2] = 1;

    mascara[1][0] = -2;
    mascara[1][1] = 0;
    mascara[1][2] = 2;

    mascara[2][0] = -1;
    mascara[2][1] = 0;
    mascara[2][2] = 1;

    return mascara;

def imagenFiltroSobel(imagenGx, imagenGy) :

    filas, columnas = imagenGx.shape
    umbral = 100;
    intensidad = None;
    valGx = None
    valGy = None;

    sobel = np.zeros((filas, columnas), dtype=np.uint8)

    for i in range(filas):
        for j in range(columnas) :

            valGx = imagenGx[i][j];
            valGy = imagenGy[i][j];

            #Realizamos la operacion de la magnitud de G por pixel
            intensidad = math.sqrt(math.pow(valGx, 2) + math.pow(valGy, 2));

            sobel[i][j] = int(intensidad);
        
    

    return sobel;

def calcularDirecciones(imagenGx, imagenGy) :

    filas, columnas = imagenGx.shape
    valGx , valGy = None, None;

    direcciones = np.zeros((filas, columnas), dtype=np.float32)

    for i in range(filas):
        for j in range(columnas) :
            valGx = imagenGx[i][j];
            valGy = imagenGy[i][j];

            #Obtenemos el angulo del pixel
            if valGx == 0:
                direcciones[i][j] = 0
            else:
                direcciones[i][j] = (math.atan(valGy / valGx) * 180.0) / math.pi; 

            if (direcciones[i][j] < 0) :
                direcciones[i][j] += 180;
            

    return direcciones;
    
def nonMaxSupression(imagenSobel, direcciones):

    filas, columnas = imagenSobel.shape

    imgNonMaxSupr = np.zeros((filas, columnas), dtype=np.uint8)

    for i in range(1, filas - 1):
        for j in range(1, columnas-1) :
            primerLado = 255;
            segundoLado = 255;

            #si el angulo es 0° o bien 180°, obtiene las intensidades de izquierda y derecha
            if ((0 <= direcciones[i][j] < 22.5) or (157.5 <= direcciones[i][j] <= 180)) :
                primerLado = imagenSobel[i][j + 1];
                segundoLado = imagenSobel[i][j - 1]
            

            elif (22.5 <= direcciones[i][j] < 67.5) :
                primerLado = imagenSobel[i + 1][j - 1];
                segundoLado = imagenSobel[i - 1][j + 1];
            

            elif (67.5 <= direcciones[i][j] < 112.5) :
                primerLado = imagenSobel[i + 1][j];
                segundoLado = imagenSobel[i - 1][j];

            elif (112.5 <= direcciones[i][j] < 157.5) :
                primerLado = imagenSobel[i - 1][j - 1];
                segundoLado = imagenSobel[i + 1][j + 1]
            

            if (imagenSobel[i][j] >= primerLado) and (imagenSobel[i][j] >= segundoLado):
                imgNonMaxSupr[i][j] = imagenSobel[i][j];
            
            else:
                imgNonMaxSupr[i][j] = int(0);
            
        
    return imgNonMaxSupr;

def umbralHysteresis(imgNonMaxSupr, upperThresholdPorcentaje, lowThresholdPorcentaje):
    filas, columnas = imgNonMaxSupr.shape

    imgHysteresis = np.zeros((filas, columnas), dtype=np.uint8)

    upperThreshold, lowThreshold = None, None

    upperThreshold = np.amax(imgNonMaxSupr) * upperThresholdPorcentaje; 
    lowThreshold = upperThreshold * lowThresholdPorcentaje; 

    weak = lowThreshold;
    strong = 255;
    irrelevant = 0;

    for i in range(filas):
        for j in range(columnas) :

            if (float(imgNonMaxSupr[i][j]) >= upperThreshold) :
                imgHysteresis[i][j] = strong;
            
            elif ((lowThreshold < float(imgNonMaxSupr[i][j])) and (float(imgNonMaxSupr[i][j]) < upperThreshold)) :
                imgHysteresis[i][j] = weak;
            
            else :
                imgHysteresis[i][j] = irrelevant;
            
    return imgHysteresis;


def hysteresis( imgHysteresis, upperThresholdPorcentaje, lowThresholdPorcentaje, imgNonMaxSupr ):
    filas, columnas = imgHysteresis.shape

    upperThreshold, lowThreshold = None, None;

    upperThreshold = np.amax(imgNonMaxSupr) * upperThresholdPorcentaje;
    lowThreshold = upperThreshold * lowThresholdPorcentaje;

    imgHysteresisFinal = np.zeros((filas, columnas), dtype=np.float32)

    for i in range(1, filas - 1):
        for j in range(1, columnas-1) :
        
            if (float(imgHysteresis[i][j] == round(lowThreshold))):
            
                if ((imgHysteresis[i+1][ j-1] == 255) or (imgHysteresis[i+1][ j] == 255) or (imgHysteresis[i+1][ j+1] == 255)
                    or (imgHysteresis[i][ j-1] == 255) or (imgHysteresis[i][ j+1] == 255)
                    or (imgHysteresis[i-1][ j-1] == 255) or (imgHysteresis[i-1][ j] == 255) or (imgHysteresis[i-1][ j+1] == 255)):

                    imgHysteresisFinal[i][j] = 255
                
                else :
                    imgHysteresisFinal[i][j] = 0;
               
            else :
                imgHysteresisFinal[i -1][ j -1] = float(imgHysteresis[i][j]);
                
    return imgHysteresisFinal;


def obtenerHistograma(imagen):
    largo, ancho = imagen.shape
    
    histograma = np.zeros((256))
    nivelIntensidad = 0
    
    for i in range(largo):
        for j in range(ancho):
            nivelIntensidad = imagen[i][j]
            histograma[nivelIntensidad] += 1
            
    return histograma

def calcularPeso(histograma, inicio, final):
    total = 0
    totalAux = 0
    peso = 0

    for i in range(inicio, final + 1):

        totalAux += histograma[i]
    

    for i in range(0, len(histograma)):
    
        total += histograma[i]
    
    if (total != 0):
        peso = totalAux / total
    
    return peso

def calcularPromedio(histograma, inicio, final):
    
    sumAux = 0
    sumF = 0
    promedio = 0

    for i in range(inicio, final + 1):
    
        sumAux += histograma[i] * i
        sumF += histograma[i]
    
    if (sumF != 0) :
        promedio = sumAux / sumF
    

    return promedio

def calcularVarianza(histograma, inicio, final):
    totalAux = 0
    sumatoria = 0
    promedio = calcularPromedio(histograma, inicio, final)
    var = 0

    for i in range(inicio, final + 1):
    
        sumatoria += pow((i - promedio), 2) * histograma[i]
        totalAux += histograma[i]
    

    if (totalAux != 0) :
        var = sumatoria / totalAux
    

    return var

def umbralAlgoritmoOTSU(histograma):
    vMinima = 10000000
    umbral = 0

    for t in range(256):
    
        wb = calcularPeso(histograma, 0, t)
        vb = calcularVarianza(histograma, 0, t)

        wf = calcularPeso(histograma, t + 1, 255)
        vf = calcularVarianza(histograma, t + 1, 255)

        vw = (wb * vb) + (wf * vf)

        if (vw < vMinima) :
            vMinima = vw
            umbral = t

    return umbral

def umbralizarImagen(imagen, umbral):
    
    largo, ancho = imagen.shape
    
    imagenUmbralizada = np.zeros(largo * ancho, dtype=np.uint8).reshape(largo, ancho)

    for i in range(largo):
    
        for j in range(ancho):
        
            if (imagen[i][j] > umbral) :
                imagenUmbralizada[i][j] = 0
            
            else :
                imagenUmbralizada[i][j] = 255
            

    return imagenUmbralizada


def mascaraLaplacianoGaussiano(mascSize, sigma):
    limite = int((mascSize - 1) / 2)
    logResultado= 0.0
    mascara = np.zeros((mascSize, mascSize), dtype=np.float64)
    sum = 0.0
    
    for x in range(-limite, limite + 1):
        for y in range(-limite, limite + 1):
            a = 1 / (2 * (3.1416) * sigma**4)
            b = 2 - ((x**2 + y**2) / sigma**2)
            c = - ((x**2 + y**2) / (2 * sigma**2))
            d = math.exp(c)
            
            logResultado = a * b * d
            
            mascara[x + limite][y + limite] = logResultado
            sum += logResultado
              
    return mascara

def umbraladoLog(imagen, delta):
    
    largo, ancho = imagen.shape
    
    matrizCambioSigno = np.zeros(largo * ancho, dtype=np.float64).reshape(largo, ancho)
    matrizDifDelta = np.zeros(largo * ancho, dtype=np.float64).reshape(largo, ancho)
    imagenUmbralada = np.zeros(largo * ancho, dtype=np.float64).reshape(largo, ancho)
    
    for i in range(1, largo - 1):
        for j in range(1, ancho - 1):
            vecinos = [(imagen[i-1][j-1], (i-1, j-1)), (imagen[i][j-1], (i, j-1)), (imagen[i+1][j-1], (i+1, j-1)), 
                       (imagen[i-1][j], (i-1, j)), (imagen[i+1][j], (i+1, j)),
                       (imagen[i-1][j+1], (i-1, j+1)), (imagen[i][j+1], (i, j+1)), (imagen[i+1][j+1], (i+1, j+1))] 
            
            val_actual = imagen[i][j]  
            signo_actual = signoNumero(val_actual)       
            
            for vecino in vecinos:
                val_vecino = vecino[0]
                signo_vecino = signoNumero(val_vecino)
                
                coords = vecino[1]
                coordX = coords[0]
                coordY = coords[1]
                
                matrizCambioSigno[coordX][coordY] = cambioSigno(signo_actual, signo_vecino)
                
                difDelta = val_vecino - val_actual
                matrizDifDelta[coordX][coordY] = cumplioDelta(delta, difDelta)
                
    for i in range(largo):
        for j in range(ancho):
            
            if (matrizCambioSigno[i][j] == 255 and matrizDifDelta[i][j] == 255):
                imagenUmbralada[i][j] = 255
            else:
                imagenUmbralada[i][j] = 0
             
                
    return imagenUmbralada, matrizCambioSigno, matrizDifDelta

def signoNumero(numero):
    signo = ""
    if numero < 0:
        signo = "-"
    else:
        signo = "+"  
    
    return signo   

def cambioSigno(signo_actual, signo_vecino )  :
    if signo_vecino == signo_actual:
        return 0
    else:
        return 255
    
def cumplioDelta(delta, dif):
    if abs(dif) > delta:
        return 255
    else:
        return 0
    
def algoritmoWaterShed(image):
    
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Overlapping objects')
    ax[1].imshow(-distance, cmap=plt.cm.gray)
    ax[1].set_title('Distances')
    ax[2].imshow(labels, cmap=plt.cm.nipy_spectral)
    ax[2].set_title('Separated objects')

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    plt.show()
    
    return labels, -distance

def agrupamientoPorColoresKmeans(imagen, kClusters):

    imagenNp = np.array(imagen, dtype=np.uint8)

    R = imagenNp[:,:,0]
    G = imagenNp[:,:,1]
    B = imagenNp[:,:,2]

    XR = R.reshape((-1, 1))  
    XG = G.reshape((-1, 1)) 
    XB = B.reshape((-1, 1)) 

    X = np.concatenate((XR,XG,XB), axis=1)

    modelo = KMeans(kClusters, random_state=0)
    modelo.fit(X)
    centroides = modelo.cluster_centers_
    asignaciones = modelo.labels_

    m = XR.shape
    for i in range(m[0]):
        XR[i] = int(centroides[asignaciones[i]][0]) 
        XG[i] = int(centroides[asignaciones[i]][1]) 
        XB[i] = int(centroides[asignaciones[i]][2]) 
        
    XR.shape = R.shape 
    XG.shape = G.shape
    XB.shape = B.shape 

    XR = XR[:, :, np.newaxis]  
    XG = XG[:, :, np.newaxis]
    XB = XB[:, :, np.newaxis]

    imagenAgrupadaColores = np.concatenate((XR,XG,XB), axis=2)
    
    return imagenAgrupadaColores, centroides

def obtenerCoordenadasPorNivelGris(imagen, nivelGris):
    largo, ancho = imagen.shape
    coordenadas = []
    for i in range(largo):
        for j in range(ancho):
            if (imagen[i][j] == nivelGris):
                coordenadas.append([i, j])
                
    return coordenadas

def extraerCoordenadasPorObjeto (imgOriginal, objetos):
    imgAgrupadaColores, centroides = agrupamientoPorColoresKmeans(imgOriginal, objetos)
    
    largo, ancho = imgAgrupadaColores.shape
            
    matrizObjetosColor = []

    for color in centroides:
        matriz = np.zeros(imgAgrupadaColores.shape)
        
        for i in range(largo):
            for j in range(ancho):
                intensidad = imgAgrupadaColores[i][j]
                color = np.array(color, dtype=np.float32)
                color = np.flip()
                if (intensidad == color).all():
                    matriz[i][j] = intensidad
                
        matrizObjetosColor.append(matriz)
        
    objetosSeparadosBordes = []
    for matriz in matrizObjetosColor: 
        sigma = 1
        mascSize = 5
        delta = 10
        
        matrizEscGrises = convertirEscalaGrisesNTSC(matriz)
        mascaraLog = mascaraLaplacianoGaussiano(mascSize, sigma)
        matrizRelleno = crearMatrizRelleno(matrizEscGrises, mascSize) 
        imgLog = aplicarFiltro(matrizEscGrises, matrizRelleno, mascaraLog, mascSize)
        imgUmbralada, matrizCambioSigno, matrizDifDelta = umbraladoLog(imgLog, delta)
        
        objetosSeparadosBordes.append(imgUmbralada)
    
    coordenadasPorObjeto = []
    for img in objetosSeparadosBordes:
        coordenadasPorObjeto.append(obtenerCoordenadasPorNivelGris(img, 255))
        
    return matrizObjetosColor, objetosSeparadosBordes, coordenadasPorObjeto    
    

def agrupamientoPorCoordenadasKmeans(imagenUmbralizada, noObjetos):
    
    largo, ancho = imagenUmbralizada.shape
    
    coordenadasBlanco = []    
    
    for i in range(largo):
        for j in range(ancho):
            if (imagenUmbralizada[i][j] == 255):
                coordenadasBlanco.append([i, j])
            
    coordenadasBlanco = np.array(coordenadasBlanco, dtype=np.uint8)
    
    kClusters = noObjetos
    
    centroides, asignaciones= km.k_means(coordenadasBlanco, kClusters)
    
    asignaciones = np.array(asignaciones, dtype=np.uint8)
    
    asig_Coords = list(zip(asignaciones, coordenadasBlanco))
    
    objetosMatricesList = []
    
    for ob in range(noObjetos):
        matriz = np.zeros(imagenUmbralizada.shape)
           
        for tupla in asig_Coords:
            if (tupla[0] == ob):
                coords = tupla[1]
                x = coords[0]
                y = coords[1]
                
                matriz[x][y] = 255
        
                
        objetosMatricesList.append(matriz)
        
    return objetosMatricesList
    
        
    
    
    
    
    
    
    