import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class AudioBase(ABC):
    @abstractmethod
    def cargar_audio(self, ruta):
        pass

    @abstractmethod
    def obtener_duracion(self):
        pass

    @abstractmethod
    def graficar_onda(self):
        pass

class AnalizadorAudio(AudioBase):
    def __init__(self):
        self.audio = None
        self.frecuencia = None
        self.ruta = None

    def cargar_audio(self, ruta):
        self.ruta = ruta
        self.audio, self.frecuencia = librosa.load(ruta, sr=None)

    def obtener_duracion(self):
        return librosa.get_duration(y=self.audio, sr=self.frecuencia)

    def energia_promedio(self):
        return np.mean(self.audio ** 2)

    def graficar_onda(self):
        plt.figure(figsize=(8, 3))
        plt.plot(self.audio, color='steelblue')
        plt.title(f"Onda de audio - {os.path.basename(self.ruta)}")
        plt.xlabel("Muestras")
        plt.ylabel("Amplitud")
        plt.tight_layout()
        plt.show()

class AnalizadorCarpeta:
    def __init__(self):
        self.resultados = []

    def buscar_audios(self, carpeta):
        for elemento in os.listdir(carpeta):
            ruta = os.path.join(carpeta, elemento)
            if os.path.isdir(ruta):
                self.buscar_audios(ruta)
            elif ruta.lower().endswith(".wav"):
                self.analizar_audio(ruta)

    def analizar_audio(self, ruta):
        analizador = AnalizadorAudio()
        analizador.cargar_audio(ruta)
        duracion = analizador.obtener_duracion()
        energia = analizador.energia_promedio()
        self.resultados.append({
            "nombre": os.path.basename(ruta),
            "duracion": duracion,
            "energia": energia
        })
        analizador.graficar_onda()

    def graficar_duraciones(self):
        nombres = [r["nombre"] for r in self.resultados]
        duraciones = [r["duracion"] for r in self.resultados]
        plt.figure(figsize=(8, 4))
        plt.bar(nombres, duraciones, color='skyblue')
        plt.title("Duración de los audios")
        plt.ylabel("Segundos")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    carpeta = input("Ingrese la ruta de los audios (.wav): ").strip()

    if not os.path.isdir(carpeta):
        print("Ruta no encontrada.")
    else:
        analizador = AnalizadorCarpeta()
        analizador.buscar_audios(carpeta)

        if not analizador.resultados:
            print("No se encontraron archivos .wav.")
        else:
            print("\nResultados del análisis:")
            for r in analizador.resultados:
                print(f" - {r['nombre']}: {r['duracion']:.2f} s | Energía: {r['energia']:.6f}")
            analizador.graficar_duraciones()
