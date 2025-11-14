Punto 2

**Descripción del problema**

El objetivo de este proyecto es procesar automáticamente imágenes aplicando transformaciones básicas como escala de grises y desenfoque.
El programa recorre una carpeta que contiene imágenes, las carga y aplica los filtros seleccionados, mostrando los resultados visualmente.

Este tipo de procesamiento es útil en tareas de preprocesamiento de datos para inteligencia artificial, análisis visual o edición masiva de imágenes, donde hacerlo manualmente sería lento y poco eficiente.

Para resolver este problema, se desarrolló un sistema que implementa clases, herencia, recursividad y abstracción, aplicando los principios de la Programación Orientada a Objetos (POO).

**Solución implementada**

El programa está estructurado en varias clases:

ProcesadorImagen (clase abstracta)
Define los métodos esenciales (cargar_imagen, aplicar_filtro, mostrar_imagen) que deben implementar las clases derivadas.
Esto garantiza que todos los filtros sigan la misma estructura base.

FiltroGrises (clase hija)
Convierte la imagen original a escala de grises utilizando cv2.cvtColor().
Permite observar la intensidad luminosa de las imágenes y reducir la información de color para análisis posteriores.

FiltroDesenfoque (clase hija)
Aplica un filtro gaussiano con cv2.GaussianBlur() para suavizar la imagen, reduciendo ruido y mejorando la estética.

AnalizadorCarpeta
Utiliza recursividad para recorrer automáticamente todas las carpetas y subcarpetas, procesando cada imagen encontrada.
Dependiendo de la elección del usuario (escala de grises o desenfoque), aplica el filtro correspondiente a todas las imágenes detectadas.

En el programa principal, el usuario ingresa la ruta de la carpeta y selecciona el tipo de filtro que desea aplicar.
El sistema muestra las imágenes originales junto con las filtradas, permitiendo comparar visualmente los resultados.

**librerías utilizadas**

cv2 (OpenCV) → Lectura, escritura y manipulación de imágenes.

numpy → Operaciones matemáticas sobre los píxeles.

os → Manejo de rutas, carpetas y archivos.

matplotlib.pyplot → Visualización gráfica de las imágenes originales y procesadas.

abc (Abstract Base Class) → Implementación de clases abstractas y métodos obligatorios.


https://www.youtube.com/watch?v=_i8kIFnjz80