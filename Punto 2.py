import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class ProcesadorImagen(ABC):
    @abstractmethod
    def cargar_imagen(self, ruta):
        pass

    @abstractmethod
    def aplicar_filtro(self):
        pass

    @abstractmethod
    def mostrar_imagen(self):
        pass

class FiltroGrises(ProcesadorImagen):
    def __init__(self):
        self.imagen = None
        self.imagen_filtrada = None
        self.ruta = None

    def cargar_imagen(self, ruta):
        self.ruta = ruta
        self.imagen = cv2.imread(ruta)
        self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)

    def aplicar_filtro(self):
        self.imagen_filtrada = cv2.cvtColor(self.imagen, cv2.COLOR_RGB2GRAY)

    def mostrar_imagen(self):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(self.imagen)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(self.imagen_filtrada, cmap="gray")
        plt.title("Escala de grises")
        plt.axis("off")

        plt.suptitle(os.path.basename(self.ruta))
        plt.tight_layout()
        plt.show()

class FiltroDesenfoque(ProcesadorImagen):
    def __init__(self):
        self.imagen = None
        self.imagen_filtrada = None
        self.ruta = None

    def cargar_imagen(self, ruta):
        self.ruta = ruta
        self.imagen = cv2.imread(ruta)
        self.imagen = cv2.cvtColor(self.imagen, cv2.COLOR_BGR2RGB)

    def aplicar_filtro(self):
        self.imagen_filtrada = cv2.GaussianBlur(self.imagen, (9, 9), 0)

    def mostrar_imagen(self):
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(self.imagen)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(self.imagen_filtrada)
        plt.title("Desenfocada")
        plt.axis("off")

        plt.suptitle(os.path.basename(self.ruta))
        plt.tight_layout()
        plt.show()

class AnalizadorCarpeta:
    def __init__(self, tipo_filtro):
        self.tipo_filtro = tipo_filtro

    def procesar_carpeta(self, carpeta):
        for elemento in os.listdir(carpeta):
            ruta = os.path.join(carpeta, elemento)
            if os.path.isdir(ruta):
                self.procesar_carpeta(ruta)
            elif ruta.lower().endswith((".jpg", ".png", ".jpeg")):
                self.procesar_imagen(ruta)

    def procesar_imagen(self, ruta):
        if self.tipo_filtro == "grises":
            procesador = FiltroGrises()
        else:
            procesador = FiltroDesenfoque()

        procesador.cargar_imagen(ruta)
        procesador.aplicar_filtro()
        procesador.mostrar_imagen()

if __name__ == "__main__":
    carpeta = input("Ingrese la ruta de la carpeta con las imágenes: ").strip()
    if not os.path.isdir(carpeta):
        print("Ruta no encontrada.")
    else:
        print("\nElija el filtro a aplicar:")
        print("1. Escala de grises")
        print("2. Desenfoque")
        opcion = input("Opción (1 o 2): ").strip()

        if opcion == "1":
            tipo = "grises"
        else:
            tipo = "desenfoque"

        analizador = AnalizadorCarpeta(tipo)
        analizador.procesar_carpeta(carpeta)
