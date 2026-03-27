Punto 1

**Descripción del problema**

El objetivo principal de este proyecto es automatizar el análisis de múltiples archivos de audio (.wav) almacenados en diferentes carpetas.
El problema surge cuando se necesita calcular información clave de muchos audios —como la duración, la energía promedio o visualizar su forma de onda— y hacerlo manualmente sería lento y propenso a errores.

El programa resuelve esto recorriendo de forma recursiva todas las carpetas y subcarpetas, procesando cada archivo de audio automáticamente.
Además, implementa Programación Orientada a Objetos (POO) con herencia y clases abstractas, para mantener una estructura clara, reutilizable y escalable.

**Solución implementada**

Se diseñaron tres clases principales:

AudioBase (clase abstracta):
Define los métodos esenciales que cualquier clase hija debe implementar, como load_audio(), get_duration() y plot_waveform().
Esto asegura una estructura uniforme en el análisis de audio.

AudioAnalyzer (clase hija):
Implementa los métodos definidos por la clase base, permitiendo cargar archivos de audio con Librosa, calcular la duración, la energía promedio y graficar su forma de onda mediante Matplotlib.

BatchAudioAnalyzer:
Aplica recursividad para explorar carpetas y analizar automáticamente todos los archivos .wav.
Usa polimorfismo al crear objetos de AudioAnalyzer dentro de su método analizar_audio(), lo que permite procesar cada audio de forma independiente.
Finalmente, genera una gráfica comparativa de la duración de todos los audios analizados.

En conjunto, el programa realiza un análisis automatizado, visual y organizado de los audios, mostrando resultados en consola y gráficos que facilitan la interpretación.

**Librerías utilizadas**

os → Exploración de carpetas, verificación de rutas y acceso a subdirectorios.

librosa → Lectura, carga y análisis técnico de archivos de audio (.wav).

numpy → Cálculos matemáticos, como energía promedio e intensidades.

matplotlib.pyplot → Generación de gráficos de forma de onda y barras comparativas.

abc → Implementación de clases abstractas y herencia.

https://www.youtube.com/watch?v=fl2diKexfyA