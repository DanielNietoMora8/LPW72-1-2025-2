# LPW72-1-2025-2
# Pacial 4 lenguajes de programación 

**Profesor**: Daniel Alexis Nieto Mora

**Estudiante**: Julian David Arias Zapata
_____________________________________
_____________________________________

# Punto 1: Análisis de Audio con Librosa y Programación Orientada a Objetos

## Descripción del problema:
Se requiere desarrollar un programa que analice archivos de audio **.wav** ubicados en una carpeta, y de acuerdo con las indicaciones del trabajo se debe usar programación orientada a objetos, así que el sistema debe realizar lo siguiente:

1. Cargar y analizar múltiples archivos de audio.
2. Calcular duración y energía promedio.
3. Mostrar la forma de onda.
4. Recorrer carpetas de forma recursiva.
5. Graficar las duraciones de todos los archivos procesados.

## Solución implementada
Se implementó un sistema basado en *herencia, abstracción y polimorfismo*:

- **AudioBase**: Clase abstracta que define la interfaz común (load_audio, duration, plot_waveform).
- **AudioAnalyzer**: Clase hija que implementa el análisis básico (carga con Librosa, cálculo de duración y energía, graficación).
- **BatchAudioAnalyzer**: Clase que utiliza una *función recursiva* para explorar subcarpetas y procesar todos los **.wav**.
- **Programa principal**: Solicita una carpeta, procesa los audios, guarda resultados en una lista de diccionarios y genera un gráfico de barras con las duraciones.

## A continución se muestra un ejemplo de la ejecución:

**1. FORMAS DE ONDAS PROCESADAS**

*Las gráficas representan la forma de la onda para los archivos ".wav"; se muestra como cambia la amplitud del sonido a lo largo del tiempo*

<img width="573" height="435" alt="image" src="https://github.com/user-attachments/assets/fd92e8c7-cc3e-491a-819c-378b529fad2e" />
<img width="548" height="432" alt="image" src="https://github.com/user-attachments/assets/b6649e91-e56a-4661-903c-dfd6ba8fdd99" />

_________________________________________________


**2. DIAGRAMA DE BARRAS**

*La gráfica representa el tiempo de duración en segundos para los archivos ".wav" de la carpeta*

<img width="976" height="585" alt="image" src="https://github.com/user-attachments/assets/c100c379-2127-4f76-8a52-1497c9a49433" />

_________________________________________________
_________________________________________________


# Punto 2: Procesamiento de Imágenes y Herencia con OpenCV 

## Descripción del Problema:
Se desea implementar un sistema que procese imágenes aplicando filtros básicos, usando herencia y polimorfismo, así que el sistema debe realizar lo siguiente:

1. Recorrer carpetas recursivamente.
2. Aplicar filtros (escala de grises y desenfoque).
3. Mostrar la imagen original y procesada.
4. Guardar los resultados en una carpeta "resultados".

## Solución Implementada
Se diseñó un sistema modular con:

- **ImageProcessor**: Clase abstracta con métodos load_image, apply_filter, show_image.
- **GrayScaleFilter**: Convierte a escala de grises usando cv2.cvtColor().
- **BlurFilter**: Aplica desenfoque gaussiano con cv2.GaussianBlur().
- **Recorrido recursivo**: Función que explora subcarpetas y procesa .jpg,.jpeg y .png.
- **Visualización**: Usa matplotlib para mostrar pares de imágenes (original vs filtrada).
- **Salida**: Guarda imágenes procesadas en "./resultados/".

## A continución se muestra un ejemplo de la ejecución:

**1. ESCALA DE GRISES**

<img width="980" height="291" alt="image" src="https://github.com/user-attachments/assets/61034666-9728-4aa3-9649-fb385a7d6eea" />


**2. DESENFOQUE GAUSSIANO**

<img width="958" height="501" alt="image" src="https://github.com/user-attachments/assets/3fd5d548-13c7-4601-a2ba-779f85e440c5" />

**3. ESCALA DE GRISES Y DESENFOQUE GAUSSIANO**

<img width="735" height="571" alt="image" src="https://github.com/user-attachments/assets/b74de422-5406-4ac5-8e1c-04db71bcbb17" />


## Librerias utilizadas:

- **librosa**: Carga y análisis de audio.
- **numpy**: Operaciones numéricas.
- **matplotlib.pyplot**: Graficación de formas de onda y duraciones.
- **cv2 (OpenCV)**: Procesamiento de imágenes.
- **os**: Manejo de rutas y sistema de archivos.
- **abc**: Clases abstractas en Python.

## Enlace del video

https://youtu.be/BwFthPRSQZk




