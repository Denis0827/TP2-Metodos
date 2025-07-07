import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def vectorizar_imagen(ruta, size=(256, 256), normalizar=False):
    """
    Vectoriza una imagen convirtiéndola a escala de grises y redimensionándola.
    
    Args:
        ruta: Ruta de la imagen
        size: Tamaño de la imagen (ancho, alto)
        normalizar: Si True, normaliza los valores entre -0.5 y 0.5
    
    Returns:
        Vector numpy de la imagen
    """
    img = Image.open(ruta) 
    img = img.resize(size)  # redimensiona la imagen
    gray = img.convert('L')  # transforma a escala de grises
    vector = np.array(gray).flatten().reshape(-1, 1)
    
    if normalizar:  # normalizar el vector
        vector = vector / 255.0
        vector = vector - 0.5

    return vector

def prediccion(i, w, b):
    """
    Calcula la predicción del modelo usando la función sigmoid.
    
    Args:
        i: Vector de entrada de la imagen
        w: Vector de pesos
        b: Término de sesgo
    
    Returns:
        Predicción entre 0 y 1
    """
    z = np.dot(w.T, i) + b
    return 1 / (1 + np.exp(-z))

def error_cuadratico(carpeta, w, b, size=(256, 256), normalizar=False):
    """
    Calcula el error cuadrático medio para todas las imágenes en una carpeta.
    
    Args:
        carpeta: Ruta de la carpeta con las imágenes
        w: Vector de pesos
        b: Término de sesgo
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
    
    Returns:
        Error cuadrático medio
    """
    error_total = 0.0
    contador = 0
    
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            # si tiene parkinson, el diagnóstico es positivo (1), sino es negativo (0)
            if archivo.startswith("Healthy"): 
                d = 0
            else:
                d = 1
                
            ruta = os.path.join(carpeta, archivo)
            i = vectorizar_imagen(ruta, size=size, normalizar=normalizar)
            pred = prediccion(i, w, b)  # calculo la predicción
            error_total += (pred - d) ** 2  # sumo el error parcial al total
            contador += 1

    return error_total / contador if contador > 0 else 0.0

def derivada_parcial_w(carpeta, w, b, size=(256, 256), normalizar=False):
    """
    Calcula la derivada parcial de la función de pérdida con respecto a w.
    
    Args:
        carpeta: Ruta de la carpeta con las imágenes
        w: Vector de pesos
        b: Término de sesgo
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
    
    Returns:
        Vector de derivadas parciales con respecto a w
    """
    derivada_w_total = np.zeros_like(w)
    
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            if archivo.startswith("Healthy"):
                d = 0
            else:
                d = 1

            ruta = os.path.join(carpeta, archivo)
            i = vectorizar_imagen(ruta, size=size, normalizar=normalizar)
            pred = prediccion(i, w, b)
            # Para sigmoid, la derivada es pred * (1 - pred) * (pred - d) * i
            derivada_w_total += pred * (1 - pred) * (pred - d) * i
    
    return derivada_w_total

def derivada_parcial_b(carpeta, w, b, size=(256, 256), normalizar=False):
    """
    Calcula la derivada parcial de la función de pérdida con respecto a b.
    
    Args:
        carpeta: Ruta de la carpeta con las imágenes
        w: Vector de pesos
        b: Término de sesgo
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
    
    Returns:
        Derivada parcial con respecto a b
    """
    derivada_b_total = 0.0

    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            if archivo.startswith("Healthy"):
                d = 0
            else:
                d = 1
                
            ruta = os.path.join(carpeta, archivo)
            i = vectorizar_imagen(ruta, size=size, normalizar=normalizar)
            pred = prediccion(i, w, b)
            # Para sigmoid, la derivada es pred * (1 - pred) * (pred - d)
            derivada_b_total += pred * (1 - pred) * (pred - d)
    
    return derivada_b_total

def actualizacion(carpeta, w, b, alpha, size=(256, 256), normalizar=False):
    """
    Aplica la regla de actualización para w y b usando descenso de gradiente.
    
    Args:
        carpeta: Ruta de la carpeta con las imágenes
        w: Vector de pesos actual
        b: Término de sesgo actual
        alpha: Tasa de aprendizaje
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
    
    Returns:
        Nuevos valores de w y b
    """
    derivada_w = derivada_parcial_w(carpeta, w, b, size=size, normalizar=normalizar)
    derivada_b = derivada_parcial_b(carpeta, w, b, size=size, normalizar=normalizar)
    w_nuevo = w - alpha * derivada_w
    b_nuevo = b - alpha * derivada_b
    return w_nuevo, b_nuevo

def entrenar_descenso_gradiente(carpeta, w_inicial, b_inicial, alpha, size=(256, 256), 
                               normalizar=False, repeticiones=10, epsilon=0.0000001):
    """
    Entrena el modelo usando descenso de gradiente.
    
    Args:
        carpeta: Ruta de la carpeta de entrenamiento
        w_inicial: Vector de pesos inicial
        b_inicial: Término de sesgo inicial
        alpha: Tasa de aprendizaje
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
        repeticiones: Número máximo de iteraciones
        epsilon: Tolerancia para convergencia
    
    Returns:
        w_final, b_final, lista de errores durante el entrenamiento
    """
    w = w_inicial.copy()
    b = b_inicial
    
    # Listas para almacenar el historial de errores
    errores_entrenamiento = []
    
    error_anterior = error_cuadratico(carpeta, w, b, size=size, normalizar=normalizar)
    errores_entrenamiento.append(error_anterior)
    
    print(f"Error inicial: {error_anterior}")
    
    for i in range(repeticiones):
        # Actualizar parámetros
        w, b = actualizacion(carpeta, w, b, alpha, size=size, normalizar=normalizar)
        
        # Calcular nuevo error
        error_actual = error_cuadratico(carpeta, w, b, size=size, normalizar=normalizar)
        errores_entrenamiento.append(error_actual)
        
        # Mostrar progreso cada 10 iteraciones
        if i % 10 == 0:
            print(f"Iteración {i}, Error: {error_actual}")
        
        # Verificar convergencia
        if abs(error_actual - error_anterior) < epsilon:
            print(f"Convergencia alcanzada en la iteración {i}")
            break
        
        error_anterior = error_actual
    
    return w, b, errores_entrenamiento

def calcular_accuracy(carpeta, w, b, size=(256, 256), normalizar=False, umbral=0.5):
    """
    Calcula la precisión (accuracy) del modelo.
    
    Args:
        carpeta: Ruta de la carpeta con las imágenes
        w: Vector de pesos
        b: Término de sesgo
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
        umbral: Umbral para clasificación
    
    Returns:
        Accuracy del modelo
    """
    correctos = 0
    total = 0
    
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            if archivo.startswith("Healthy"): 
                d = 0
            else:
                d = 1
                
            ruta = os.path.join(carpeta, archivo)
            i = vectorizar_imagen(ruta, size=size, normalizar=normalizar)
            pred = prediccion(i, w, b)
            pred_clase = 1 if pred > umbral else 0
                
            if pred_clase == d:
                correctos += 1
            total += 1
    
    if total == 0:
        return 0.0
    
    return correctos / total

def matriz_confusion(carpeta, w, b, size=(256, 256), normalizar=False, umbral=0.5):
    """
    Calcula la matriz de confusión.
    
    Args:
        carpeta: Ruta de la carpeta con las imágenes
        w: Vector de pesos
        b: Término de sesgo
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
        umbral: Umbral para clasificación
    
    Returns:
        VP, FP, VN, FN (Verdaderos Positivos, Falsos Positivos, Verdaderos Negativos, Falsos Negativos)
    """
    vp = 0
    fp = 0
    vn = 0
    fn = 0
    
    for archivo in os.listdir(carpeta):
        if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            if archivo.startswith("Healthy"): 
                d = 0
            else:
                d = 1

            ruta = os.path.join(carpeta, archivo)
            i = vectorizar_imagen(ruta, size=size, normalizar=normalizar)
            pred = prediccion(i, w, b)
            pred_clase = 1 if pred > umbral else 0

            if d == 1 and pred_clase == 1:
                vp += 1
            elif d == 0 and pred_clase == 1:
                fp += 1
            elif d == 0 and pred_clase == 0:
                vn += 1
            elif d == 1 and pred_clase == 0:
                fn += 1

    return vp, fp, vn, fn

def visualizar_entrenamiento(errores):
    """
    Visualiza la evolución del error durante el entrenamiento.
    
    Args:
        errores: Lista de errores durante el entrenamiento
    """
    plt.figure(figsize=(10, 6))
    plt.plot(errores)
    plt.title('Evolución del Error Durante el Entrenamiento')
    plt.xlabel('Iteración')
    plt.ylabel('Error Cuadrático')
    plt.grid(True)
    plt.show()

def ejecutar_entrenamiento(carpeta_entrenamiento, carpeta_testing=None, size=(256, 256), 
                          normalizar=False, alpha=0.01, repeticiones=5):
    """
    Función principal para ejecutar el entrenamiento completo.
    
    Args:
        carpeta_entrenamiento: Ruta de la carpeta de entrenamiento
        carpeta_testing: Ruta de la carpeta de testing (opcional)
        size: Tamaño de las imágenes
        normalizar: Si normalizar las imágenes
        alpha: Tasa de aprendizaje
        repeticiones: Número de iteraciones
    
    Returns:
        w_final, b_final, errores
    """
    # Verificar que la carpeta existe
    if not os.path.exists(carpeta_entrenamiento):
        print(f"Error: La carpeta {carpeta_entrenamiento} no existe")
        print(f"Directorio actual: {os.getcwd()}")
        print("Archivos y carpetas disponibles:")
        for item in os.listdir('.'):
            print(f"  - {item}")
        return None, None, None

    # Inicializar parámetros
    n = size[0] * size[1]
    w = np.random.randn(n, 1) * 0.01  # Inicialización más pequeña
    b = 0.0
    
    print(f"Iniciando entrenamiento con:")
    print(f"  - Tamaño de imagen: {size}")
    print(f"  - Normalización: {normalizar}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Repeticiones: {repeticiones}")
    print(f"  - Dimensión de w: {w.shape}")

    # Entrenar el modelo
    w_final, b_final, errores = entrenar_descenso_gradiente(
        carpeta_entrenamiento, w, b, alpha, size=size, 
        normalizar=normalizar, repeticiones=repeticiones
    )

    # Calcular accuracy en entrenamiento
    acc_entrenamiento = calcular_accuracy(carpeta_entrenamiento, w_final, b_final, 
                                         size=size, normalizar=normalizar)
    print(f"Accuracy en entrenamiento: {acc_entrenamiento:.4f}")

    # Calcular accuracy en testing si se proporciona
    if carpeta_testing and os.path.exists(carpeta_testing):
        acc_testing = calcular_accuracy(carpeta_testing, w_final, b_final, 
                                       size=size, normalizar=normalizar)
        print(f"Accuracy en testing: {acc_testing:.4f}")

    # Visualizar entrenamiento
    visualizar_entrenamiento(errores)
    
    return w_final, b_final, errores

if __name__ == "__main__":
    # Configurar el directorio de trabajo
    os.chdir(r"C:\Users\Denis Wu\Desktop\TP2-Metodos")
    print("Directorio actual:", os.getcwd())
    
    # Carpeta de entrenamiento
    carpeta_entrenamiento = "Entrenamiento"
    
    # Verificar que existen imágenes
    archivos = [f for f in os.listdir(carpeta_entrenamiento) 
                if f.lower().endswith(('.png','.jpg','.jpeg'))]
    if not archivos:
        print("No se encontraron imágenes en la carpeta de entrenamiento")
        exit()
    
    print(f"Se encontraron {len(archivos)} imágenes en la carpeta de entrenamiento")
    
    # Contar imágenes por clase
    healthy_count = sum(1 for f in archivos if f.startswith("Healthy"))
    parkinson_count = sum(1 for f in archivos if f.startswith("Parkinson"))
    print(f"Imágenes Healthy: {healthy_count}")
    print(f"Imágenes Parkinson: {parkinson_count}")
    
    # Probar vectorización
    ruta_ejemplo = os.path.join(carpeta_entrenamiento, archivos[0])
    vector = vectorizar_imagen(ruta_ejemplo, normalizar=True)
    print("Vector shape:", vector.shape)
    print("Valores min y max:", vector.min(), vector.max())

    # Inicializar parámetros de prueba
    n = 256**2
    w_test = np.random.randn(n, 1) * 0.01  # Inicialización más pequeña
    b_test = 0.0

    # Probar predicción
    pred = prediccion(vector, w_test, b_test)
    print("Predicción ejemplo:", pred)
    
    # Probar error cuadrático
    error = error_cuadratico(carpeta_entrenamiento, w_test, b_test, normalizar=True)
    print("Error cuadrático ejemplo:", error)
    
    # Probar derivadas
    dw = derivada_parcial_w(carpeta_entrenamiento, w_test, b_test, normalizar=True)
    db = derivada_parcial_b(carpeta_entrenamiento, w_test, b_test, normalizar=True)
    print("Derivada parcial w - norma:", np.linalg.norm(dw))
    print("Derivada parcial b:", db)

    # Probar actualización
    alpha = 0.001  # Tasa de aprendizaje más pequeña
    w_n, b_n = actualizacion(carpeta_entrenamiento, w_test, b_test, alpha=alpha, normalizar=True)
    print("Cambio relativo en w:", np.linalg.norm(w_n - w_test) / np.linalg.norm(w_test))
    print("Nuevo b:", b_n)

    # Probar accuracy
    print("\nProbando accuracy...")
    acc = calcular_accuracy(carpeta_entrenamiento, w_n, b_n, normalizar=True)
    print("Accuracy ejemplo:", acc)
    
    # Probar matriz de confusión
    print("\nProbando matriz de confusión...")
    vp, fp, vn, fn = matriz_confusion(carpeta_entrenamiento, w_n, b_n, normalizar=True)
    print(f"VP: {vp}, FP: {fp}, VN: {vn}, FN: {fn}")

    # Entrenar el modelo con parámetros mejorados
    print("\nIniciando entrenamiento completo...")
    w_nuevo, b_nuevo, errores = entrenar_descenso_gradiente(
        carpeta_entrenamiento, w_test, b_test, alpha=0.001, repeticiones=200
    )

    # Evaluar modelo entrenado
    print("\nEvaluando modelo entrenado...")
    acc = calcular_accuracy(carpeta_entrenamiento, w_nuevo, b_nuevo, normalizar=True)
    print("Accuracy final:", acc)
    
    vp, fp, vn, fn = matriz_confusion(carpeta_entrenamiento, w_nuevo, b_nuevo, normalizar=True)
    print(f"Matriz de confusión final - VP: {vp}, FP: {fp}, VN: {vn}, FN: {fn}")
    
    # Mostrar algunas predicciones de ejemplo
    print("\nPredicciones de ejemplo:")
    for i, archivo in enumerate(archivos[:5]):
        ruta = os.path.join(carpeta_entrenamiento, archivo)
        vector = vectorizar_imagen(ruta, normalizar=True)
        pred = prediccion(vector, w_nuevo, b_nuevo)
        clase_real = "Healthy" if archivo.startswith("Healthy") else "Parkinson"
        print(f"{archivo}: {clase_real} -> Predicción: {pred:.4f}") 