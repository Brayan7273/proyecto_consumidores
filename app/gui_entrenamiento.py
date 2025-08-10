import os
import tkinter as tk
from tkinter import ttk, messagebox
from algoritmo import entrenar_modelo_interactivo

class AppEntrenamiento:
    def __init__(self, root):
        self.root = root
        self.root.title("Entrenamiento de Modelo de Consumidores")
        
        # Configuración de estilo
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TButton', font=('Arial', 10), padding=5)
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        
        # Marco principal
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Título
        ttk.Label(
            self.main_frame, 
            text="Sistema de Entrenamiento de Modelo de Consumidores",
            font=('Arial', 12, 'bold')
        ).pack(pady=(0, 20))
        
        # Descripción
        ttk.Label(
            self.main_frame,
            text="Esta herramienta permite entrenar un modelo de clasificación para identificar tipos de consumidores\n"
                 "basado en respuestas a un cuestionario psicográfico.",
            justify=tk.CENTER
        ).pack(pady=(0, 30))
        
        # Botón de entrenamiento
        btn_entrenar = ttk.Button(
            self.main_frame,
            text="Seleccionar Archivo y Entrenar Modelo",
            command=self.iniciar_entrenamiento
        )
        btn_entrenar.pack(pady=10, ipadx=10, ipady=5)
        
        # Instrucciones
        ttk.Label(
            self.main_frame,
            text="Instrucciones:\n"
                 "1. Haga clic en el botón superior\n"
                 "2. Seleccione el archivo Excel con los datos de entrenamiento\n"
                 "3. El modelo se guardará automáticamente en la misma carpeta",
            justify=tk.LEFT
        ).pack(pady=(30, 0))
        
    def iniciar_entrenamiento(self):
        try:
            resultado = entrenar_modelo_interactivo()
            if resultado:
                # Extraer solo el nombre del archivo para mostrarlo más limpio
                nombre_modelo = os.path.basename(resultado)
                messagebox.showinfo(
                    "Entrenamiento Completado",
                    f"Proceso de entrenamiento completado con éxito.\n\n"
                    f"Modelo guardado como:\n{nombre_modelo}\n"
                    f"en la carpeta data/models/"
                )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Ocurrió un error durante el entrenamiento:\n{str(e)}"
            )

if __name__ == '__main__':
    root = tk.Tk()
    app = AppEntrenamiento(root)
    root.mainloop()