from collections import defaultdict
from random import uniform
from math import sqrt
import random

import numpy as np

def genCentroidesRandom(datos, k):

    centroides = []
    dimensiones = len(datos[0])
    mejorMinimo = []
    mejorMaximo = []
    
    for i in range(dimensiones):
        pointsInDimension = np.array(datos)[:, i]
        minMax = min(pointsInDimension)
        maxMax = max(pointsInDimension)
        mejorMinimo.append(minMax)
        mejorMaximo.append(maxMax)

    for cluster in range(k):
        puntoRandom = []
        for i in range(dimensiones):
            minVal = mejorMinimo[i]
            maxVal = mejorMaximo[i]
            
            puntoRandom.append(random.randint(minVal, maxVal))

        centroides.append(puntoRandom)
        
    return centroides

def nuevosCentros(datos, asignaciones):
    
    grupos = defaultdict(list)
    nuevosCentroides = []
    
    for centroideAsignado, punto in zip(asignaciones, datos):
        grupos[centroideAsignado].append(punto)
        
    for centroide, puntos in grupos.items():
        nuevosCentroides.append(obtenerCentroidePromediado(puntos))

    return nuevosCentroides

def obtenerCentroidePromediado(puntos):
    
    dimensiones = len(puntos[0])
    nuevoCentroide = []

    for dimension in range(dimensiones):
        sumaAux = 0 
        for p in puntos:
            sumaAux += p[dimension]

        promedio = sumaAux / float(len(puntos))
        nuevoCentroide.append(promedio)

    return nuevoCentroide

def distancia(punto, centroide):
    
    dimensiones = len(punto)
    
    sumaAux = 0
    for dimension in range(dimensiones):
        difDistancia = (centroide[dimension] - punto[dimension]) ** 2
        sumaAux += difDistancia
        
    return sqrt(sumaAux)

def asignarPuntos(listaPuntos, centroides):

    asignacionesPuntosCentroides = []
    
    for punto in listaPuntos:
        distMasCorta = float('inf')  
        centroideDistMasCorta_index = 0
        
        for i in range(len(centroides)):
            distanciaObtenida = distancia(punto, centroides[i])
            
            if distanciaObtenida < distMasCorta:
                distMasCorta = distanciaObtenida
                centroideDistMasCorta_index = i
                
        asignacionesPuntosCentroides.append(centroideDistMasCorta_index)
    return asignacionesPuntosCentroides

def k_means(dataset, k):
    
    centroidesRandom = genCentroidesRandom(dataset, k)
    asignaciones = asignarPuntos(dataset, centroidesRandom)
    asignacionesAnteriores = None
    
    while asignaciones != asignacionesAnteriores:
        nuevosCentroides = nuevosCentros(dataset, asignaciones)
        asignacionesAnteriores = asignaciones
        asignaciones = asignarPuntos(dataset, nuevosCentroides)
        
    return nuevosCentroides, asignaciones
