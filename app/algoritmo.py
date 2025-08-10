import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import filedialog, messagebox
import os
from datetime import datetime
from flask import current_app
import os

def get_models_folder():
    return os.path.join(current_app.root_path, 'data', 'models')

class ClasificadorConsumidores:
    def __init__(self):
        self.modelo = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        self.encoder = None
        self.accuracy = None
        self.reporte = None
    
    def entrenar_evaluar(self, X_train, X_test, y_train, y_test):
        self.modelo.fit(X_train, y_train)
        
        # Evaluación
        y_pred = self.modelo.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)
        self.reporte = classification_report(y_test, y_pred, target_names=self.encoder.classes_)
        
        return self.accuracy, self.reporte
    
    def guardar_modelo(self, ruta):
        joblib.dump({
            'modelo': self.modelo,
            'encoder': self.encoder,
            'accuracy': self.accuracy,
            'reporte': self.reporte
        }, ruta)
    
    @staticmethod
    def cargar_modelo(ruta):
        datos = joblib.load(ruta)
        clf = ClasificadorConsumidores()
        clf.modelo = datos['modelo']
        clf.encoder = datos['encoder']
        clf.accuracy = datos['accuracy']
        clf.reporte = datos['reporte']
        return clf

def entrenar_modelo_flask(df):
    """Versión adaptada para Flask del entrenamiento"""
    preguntas = [f'P{i}' for i in range(1, 21)]
    X = df[preguntas]
    y = df['Tipo_Consumidor']
    
    # Codificar etiquetas
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Entrenar modelo
    modelo = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )
    modelo.fit(X, y_encoded)
    
    # Guardar modelo
    modelo_dir = os.path.join(current_app.root_path, 'data', 'models')
    os.makedirs(modelo_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"modelo_{timestamp}.pkl"
    model_path = os.path.join(modelo_dir, model_filename)
    
    joblib.dump({
        'model': modelo,
        'encoder': le,
        'features': preguntas,
        'training_date': timestamp
    }, model_path)
    
    return model_path

def cargar_datos(ruta_archivo):
    try:
        df = pd.read_excel(ruta_archivo)
        
        # Verificar estructura básica
        preguntas_requeridas = [f'P{i}' for i in range(1, 21)]
        if not all(p in df.columns for p in preguntas_requeridas):
            raise ValueError("El archivo no tiene las 20 preguntas requeridas (P1-P20)")
        
        if 'Tipo_Consumidor' not in df.columns:
            raise ValueError("El archivo no contiene la columna 'Tipo_Consumidor'")
        
        # Codificar la variable objetivo
        le = LabelEncoder()
        df['Tipo_Consumidor_encoded'] = le.fit_transform(df['Tipo_Consumidor'])
        
        # Separar características (X) y etiqueta (y)
        X = df[preguntas_requeridas]
        y = df['Tipo_Consumidor_encoded']
        
        return X, y, le, None
    
    except Exception as e:
        return None, None, None, str(e)

def interpretar_resultados(respuestas):
    """Interpretación basada en puntajes por dimensión"""
    dimensiones = {
        'Ahorro/Conservador': respuestas[0:5],
        'Planificador/Analítico': respuestas[5:10],
        'Impulsivo/Explorador': respuestas[10:15],
        'Leal a la marca/Status': respuestas[15:20]
    }
    
    puntajes = {dim: sum(preguntas) for dim, preguntas in dimensiones.items()}
    max_puntaje = max(puntajes.values())
    perfil_secundario = None
    
    sorted_puntajes = sorted(puntajes.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_puntajes) > 1 and sorted_puntajes[1][1] >= 8:
        perfil_secundario = sorted_puntajes[1][0]
    
    interpretacion = {}
    for dim, puntaje in puntajes.items():
        if puntaje <= 5:
            interpretacion[dim] = 'Baja afinidad'
        elif puntaje <= 10:
            interpretacion[dim] = 'Afinidad media'
        else:
            interpretacion[dim] = 'Alta afinidad'
    
    return {
        'puntajes_por_dimension': puntajes,
        'interpretacion_dimensiones': interpretacion,
        'perfil_secundario': perfil_secundario
    }

def seleccionar_archivo(titulo="Seleccionar archivo"):
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    root.wm_attributes('-topmost', 1)  # Mantener la ventana encima
    
    archivo = filedialog.askopenfilename(
        title=titulo,
        filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
    )
    return archivo if archivo else None

def mostrar_mensaje(titulo, mensaje):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(titulo, mensaje)
    root.destroy()

def entrenar_modelo_interactivo():
    # 1. Seleccionar archivo de entrenamiento
    archivo_entrenamiento = seleccionar_archivo("Seleccione el archivo Excel para entrenamiento")
    if not archivo_entrenamiento:
        mostrar_mensaje("Error", "No se seleccionó ningún archivo de entrenamiento")
        return None
    
    # 2. Cargar datos
    X, y, encoder, error = cargar_datos(archivo_entrenamiento)
    if error:
        mostrar_mensaje("Error", f"Error al cargar datos:\n{error}")
        return None
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 4. Entrenar modelo
    clf = ClasificadorConsumidores()
    clf.encoder = encoder
    accuracy, reporte = clf.entrenar_evaluar(X_train, X_test, y_train, y_test)
    
    # 5. Guardar modelo con timestamp en la carpeta data/models
    modelo_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'data', 'models')
    
    # Crear directorio si no existe
    os.makedirs(modelo_dir, exist_ok=True)
    
    # Generar nombre con fecha y hora
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_modelo = f"modelo_{timestamp}.pkl"
    ruta_modelo = os.path.join(modelo_dir, nombre_modelo)
    
    clf.guardar_modelo(ruta_modelo)
    
    # Mostrar resultados
    mensaje = (f"Modelo entrenado exitosamente!\n\n"
               f"Exactitud: {accuracy:.2f}\n\n"
               f"Ubicación del modelo:\n{ruta_modelo}\n\n"
               f"Reporte:\n{reporte}")
    
    mostrar_mensaje("Resultados del Entrenamiento", mensaje)
    
    return ruta_modelo



if __name__ == '__main__':
    # Ejecutar el entrenamiento interactivo
    ruta_modelo = entrenar_modelo_interactivo()
    if ruta_modelo:
        print(f"Modelo guardado en: {ruta_modelo}")