from flask import jsonify, render_template, request, redirect, send_file, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import glob
import io
from matplotlib import pyplot as plt
from fpdf import FPDF
import tempfile
from app import app
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'data', 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'app', 'data', 'models')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'app', 'data', 'outputs')

# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Diccionario con preguntas por tipo (ajústalo a tus preguntas)
preguntas_por_tipo = {
    "Ahorro": [
        "1. Prefiero esperar a que un producto esté en oferta antes de comprarlo.",
        "2. Me esfuerzo por gastar menos de lo que gano cada mes.",
        "3. Antes de comprar, comparo precios en varias tiendas o sitios web.",
        "4. Considero que ahorrar para emergencias es más importante que gastar en lujos.",
        "5. Evito compras que impliquen endeudarme."
    ],
    "Planificador": [
        "6. Me gusta planificar mis compras con antelación y hacer listas detalladas.",
        "7. Antes de tomar una decisión de compra, investigo a fondo las opciones.",
        "8. Prefiero retrasar una compra hasta estar completamente seguro de mi elección.",
        "9. Me gusta establecer metas financieras claras y cumplirlas.",
        "10. Evalúo los beneficios y desventajas antes de comprar cualquier producto importante."
    ],
    "Impulsivo": [
        "11. Disfruto comprar cosas nuevas aunque no las necesite realmente.",
        "12. A veces hago compras sin pensarlo mucho, solo porque algo me llamó la atención.",
        "13. Me atraen los productos novedosos o ediciones limitadas.",
        "14. Me emociona la idea de probar marcas o artículos que nunca he usado antes.",
        "15. Es probable que compre algo solo por el placer de tenerlo en ese momento."
    ],
    "Leal": [
        "16. Prefiero comprar siempre las mismas marcas en las que confío.",
        "17. Estoy dispuesto a pagar más por una marca que considero prestigiosa.",
        "18. Me gusta que mis compras reflejen mi estilo y estatus social.",
        "19. Recomiendo mis marcas favoritas a familiares y amigos.",
        "20. Me siento más seguro comprando productos de marcas reconocidas, aunque sean más caros."
    ]
}

# Asignar a Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

df_global = None

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_model_files():
    return glob.glob(os.path.join(MODEL_FOLDER, 'modelo_*.pkl'))

def validate_dataframe(df):
    """Valida la estructura del DataFrame"""
    required_cols = [f'P{i}' for i in range(1, 21)] + ['Tipo_Consumidor']
    if df.empty:
        return False, "El archivo está vacío"
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        return False, f"Faltan columnas requeridas: {', '.join(missing)}"
    return True, ""

@app.route('/')
def index():
    return render_template('index.html', now=datetime.now())

@app.route('/entrenar', methods=['GET', 'POST'])
def entrenar():
    datos_df = None  # Mantenemos el DataFrame original
    datos_template = None  # Datos preparados para el template
    modelo_info = None
    
    if request.method == 'POST':
        if 'archivo' not in request.files:
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(url_for('entrenar'))
        
        file = request.files['archivo']
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'danger')
            return redirect(url_for('entrenar'))
        
        try:
            # Guardar archivo
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Leer y validar datos
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Validación de estructura
            required_cols = [f'P{i}' for i in range(1, 21)] + ['Tipo_Consumidor']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                flash(f'Faltan columnas requeridas: {", ".join(missing_cols)}', 'danger')
                return redirect(url_for('entrenar'))
            
            if df.empty:
                flash('El archivo está vacío', 'danger')
                return redirect(url_for('entrenar'))
            
            # Procesamiento
            X = df[[f'P{i}' for i in range(1, 21)]]
            y = df['Tipo_Consumidor']
            
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Entrenamiento
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42
            )
            
            modelo = RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42
            )
            modelo.fit(X_train, y_train)
            
            # Evaluación
            y_pred = modelo.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Guardar modelo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"modelo_{timestamp}.pkl"
            model_path = os.path.join(MODEL_FOLDER, model_filename)
            
            joblib.dump({
                'model': modelo,
                'encoder': le,
                'training_date': timestamp,
                'features': [f'P{i}' for i in range(1, 21)],
                'accuracy': accuracy,
                'report': report
            }, model_path)
            
            # Preparar datos para template
            datos_df = df  # Guardamos el DataFrame completo
            datos_template = {
                'columnas': df.columns.tolist(),
                'filas': df.head(10).to_dict('records'),
                'total_registros': len(df)
            }
            
            modelo_info = {
                'algoritmo': 'Random Forest',
                'accuracy': accuracy,
                'fecha': timestamp,
                'tamanio': os.path.getsize(model_path)/(1024*1024),  # MB
                'metricas': report,
                'distribucion': df['Tipo_Consumidor'].value_counts().to_dict()
            }
            
            flash('Modelo entrenado exitosamente!', 'success')
            
        except Exception as e:
            flash(f'Error al procesar el archivo: {str(e)}', 'danger')
    
    return render_template(
        'entrenar.html',
        datos=datos_template,
        modelo_info=modelo_info,
        datos_df=datos_df  # Pasamos el DataFrame por si acaso
    )


@app.route('/descargar_datos')
def descargar_datos():
    try:
        files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
        if not files:
            flash('No hay archivos para descargar', 'warning')
            return redirect(url_for('entrenar'))
            
        latest_file = max(files, key=os.path.getctime)
        df = pd.read_excel(latest_file)
        
        output = io.BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='datos_consumidores.csv'
        )
    except Exception as e:
        flash(f'Error al descargar datos: {str(e)}', 'danger')
        return redirect(url_for('entrenar'))

@app.route('/descargar_modelo')
def descargar_modelo():
    try:
        models = get_model_files()
        if not models:
            flash('No hay modelos entrenados', 'warning')
            return redirect(url_for('entrenar'))
            
        latest_model = max(models, key=os.path.getctime)
        return send_from_directory(
            MODEL_FOLDER,
            os.path.basename(latest_model),
            as_attachment=True
        )
    except Exception as e:
        flash(f'Error al descargar modelo: {str(e)}', 'danger')
        return redirect(url_for('entrenar'))

@app.route('/generar_reporte/<formato>')
def generar_reporte(formato):
    try:
        # Obtener modelo
        models = get_model_files()
        if not models:
            flash('No hay modelos entrenados', 'warning')
            return redirect(url_for('entrenar'))
            
        latest_model = max(models, key=os.path.getctime)
        model_data = joblib.load(latest_model)
        
        # Obtener datos
        files = glob.glob(os.path.join(UPLOAD_FOLDER, '*'))
        if not files:
            flash('No hay archivos de datos', 'warning')
            return redirect(url_for('entrenar'))
            
        latest_file = max(files, key=os.path.getctime)
        df = pd.read_excel(latest_file)
        
        if formato == 'pdf':
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            
            # Cabecera
            pdf.cell(200, 10, txt="Reporte de Entrenamiento", ln=1, align='C')
            pdf.ln(10)
            
            # Información del modelo
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Información del Modelo", ln=1)
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt=f"Algoritmo: Random Forest", ln=1)
            pdf.cell(200, 10, txt=f"Precisión: {float(model_data['accuracy']):.2%}", ln=1)
            pdf.cell(200, 10, txt=f"Fecha: {model_data['training_date']}", ln=1)
            pdf.ln(10)
            
            # Gráfico
            distribucion = df['Tipo_Consumidor'].value_counts()
            plt.figure(figsize=(6,6))
            plt.pie(distribucion, labels=distribucion.index, autopct='%1.1f%%')
            plt.title('Distribución de Tipos')
            
            temp_img = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_img.name)
            plt.close()
            
            pdf.image(temp_img.name, x=50, w=110)
            os.unlink(temp_img.name)
            pdf.ln(80)
            
            # Métricas
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Métricas por Tipo", ln=1)
            
            for tipo, metricas in model_data['report'].items():
                if isinstance(metricas, dict):
                    pdf.set_font("Arial", 'B', 10)
                    pdf.cell(200, 10, txt=f"Tipo: {tipo}", ln=1)
                    pdf.set_font("Arial", size=10)
                    for k, v in metricas.items():
                        if isinstance(v, (int, float)):
                            pdf.cell(200, 10, txt=f"{k}: {v:.2f}", ln=1)
                    pdf.ln(5)
            
            # Generar PDF
            pdf_output = io.BytesIO()
            pdf.output(pdf_output)
            pdf_output.seek(0)
            
            return send_file(
                pdf_output,
                mimetype='application/pdf',
                as_attachment=True,
                download_name='reporte_entrenamiento.pdf'
            )
            
        elif formato == 'excel':
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Datos
                df.to_excel(writer, sheet_name='Datos', index=False)
                
                # Métricas
                metric_data = []
                for tipo, vals in model_data['report'].items():
                    if isinstance(vals, dict):
                        row = {'Tipo': tipo}
                        row.update({k: v for k, v in vals.items() if isinstance(v, (int, float))})
                        metric_data.append(row)
                
                pd.DataFrame(metric_data).to_excel(writer, sheet_name='Métricas', index=False)
                
                # Resumen
                summary = pd.DataFrame({
                    'Métrica': ['Algoritmo', 'Precisión', 'Fecha'],
                    'Valor': ['Random Forest', model_data['accuracy'], model_data['training_date']]
                })
                summary.to_excel(writer, sheet_name='Resumen', index=False)
            
            output.seek(0)
            return send_file(
                output,
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                as_attachment=True,
                download_name='reporte_entrenamiento.xlsx'
            )
            
    except Exception as e:
        flash(f'Error al generar reporte: {str(e)}', 'danger')
        return redirect(url_for('entrenar'))

# ... (resto de las rutas se mantienen igual) ...
def entrenar_modelo(df):
    # Preparar datos
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"modelo_{timestamp}.pkl"
    model_path = os.path.join(MODEL_FOLDER, model_filename)
    
    joblib.dump({
        'model': modelo,
        'encoder': le,
        'training_date': timestamp,
        'features': preguntas
    }, model_path)
    
    return model_path


@app.route('/modelos')
def modelos():
    modelos_info = []
    for model_path in get_model_files():
        model_data = joblib.load(model_path)
        modelos_info.append({
            'nombre': os.path.basename(model_path),
            'tipo_consumidor': 'General',  # O actualízalo si tu modelo es específico
            'precision': model_data.get('accuracy', 'N/A'),
            'fecha': model_data['training_date'],
            'metricas': model_data.get('report', {}),
            'ruta': model_path
        })
    
    # Ordenar por fecha (más reciente primero)
    modelos_info.sort(key=lambda x: x['fecha'], reverse=True)
    
    return render_template('modelos.html', modelos=modelos_info)

import json

@app.route('/predecir', methods=['GET', 'POST'])
def predecir():
    global df_global
    modelos_disponibles = get_model_files()
    resultados = None
    chart_data = None
    tipos_disponibles = []
    estadisticas = None
    excel_filename = None

    if request.method == 'POST':
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            tipos_seleccionados = request.form.getlist('tipos_seleccionados')

            if df_global is None:
                return jsonify({'error': 'No hay datos cargados'}), 400

            if not tipos_seleccionados:
                df_filtrado = df_global.copy()
                preguntas_a_mostrar = [col for col in df_global.columns if col.startswith('P')]
            else:
                df_filtrado = df_global[df_global['Prediccion'].isin(tipos_seleccionados)]
                preguntas_a_mostrar = []
                for tipo in tipos_seleccionados:
                    if tipo in preguntas_por_tipo:
                        preguntas_a_mostrar.extend(preguntas_por_tipo[tipo])
                preguntas_a_mostrar = list(set(preguntas_a_mostrar))

            columnas_fijas = ['Nombre', 'Sexo', 'Prediccion']
            columnas = [col for col in (columnas_fijas + preguntas_a_mostrar) if col in df_filtrado.columns]
            df_filtrado = df_filtrado[columnas]

            resultados = [
                {col: row.get(col, None) for col in columnas}
                for row in df_filtrado.head(30).to_dict('records')
            ]

            estadisticas = {
                'conteo': len(df_filtrado),
                'promedios': df_filtrado[preguntas_a_mostrar].mean().round(2).replace({pd.NA: None}).to_dict() if preguntas_a_mostrar else {}
            }

            conteo_tipos = df_filtrado['Prediccion'].value_counts().to_dict()
            promedios_por_tipo = df_filtrado.groupby('Prediccion')[preguntas_a_mostrar].mean().round(2).to_dict() if preguntas_a_mostrar else {}

            chart_data = json.dumps({
                'conteo_tipos': conteo_tipos,
                'promedios_por_tipo': promedios_por_tipo
            })

            if len(df_filtrado) > 0:
                excel_filename = 'filtrado.xlsx'
                path_excel = os.path.join(app.config['OUTPUT_FOLDER'], excel_filename)
                df_filtrado.to_excel(path_excel, index=False)

            return jsonify({
                'resultados': resultados,
                'estadisticas': estadisticas,
                'chart_data': chart_data,
                'columnas': columnas,
                'excel_filename': excel_filename
            })

        else:
            if 'modelo' not in request.form or not request.form['modelo']:
                flash('Debe seleccionar un modelo', 'danger')
                return redirect(request.url)

            if 'archivo' not in request.files or request.files['archivo'].filename == '':
                flash('No se seleccionó ningún archivo', 'danger')
                return redirect(request.url)

            file = request.files['archivo']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)

                model_path = request.form['modelo']
                model_data = joblib.load(model_path)
                modelo = model_data['model']
                encoder = model_data['encoder']
                required_columns = model_data['features']

                if not all(col in df.columns for col in required_columns):
                    flash('El archivo no tiene la estructura requerida por el modelo', 'danger')
                    return redirect(request.url)

                X = df[required_columns]
                predicciones = modelo.predict(X)
                df['Prediccion'] = encoder.inverse_transform(predicciones)

                df_global = df.copy()
                tipos_disponibles = sorted(df['Prediccion'].unique().tolist())

                preguntas_a_mostrar = [col for col in required_columns if col.startswith('P')]
                columnas_fijas = ['Nombre', 'Sexo', 'Prediccion']
                columnas = [col for col in (columnas_fijas + preguntas_a_mostrar) if col in df.columns]

                resultados = [
                    {col: row.get(col, None) for col in columnas}
                    for row in df[columnas].head(30).to_dict('records')
                ]

                conteo_tipos = df['Prediccion'].value_counts().to_dict()
                promedios_por_tipo = df.groupby('Prediccion')[preguntas_a_mostrar].mean().round(2).to_dict()

                chart_data = json.dumps({
                    'conteo_tipos': conteo_tipos,
                    'promedios_por_tipo': promedios_por_tipo
                })

                excel_filename = 'general.xlsx'
                path_excel = os.path.join(app.config['OUTPUT_FOLDER'], excel_filename)
                df[columnas].to_excel(path_excel, index=False)

                flash('Predicciones generadas exitosamente', 'success')

            else:
                flash('Archivo no permitido', 'danger')
                return redirect(request.url)

    return render_template('predecir.html',
                           modelos=modelos_disponibles,
                           resultados=resultados,
                           chart_data=chart_data,
                           tipos_disponibles=tipos_disponibles,
                           estadisticas=estadisticas,
                           excel_filename=excel_filename)





@app.route('/descargar/<filename>')
def descargar_filtrados(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)


import io
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tempfile


import numpy as np
import tempfile
import os
from fpdf import FPDF
import matplotlib.pyplot as plt

def generar_pdf_con_resultados(df, conteo_tipos, promedios_por_tipo, distribucion_puntaje, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Reporte de Clasificación", ln=True, align='C')

    # Gráfica 1: Pie chart conteo tipos
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1:
        plt.figure(figsize=(4,4))
        plt.pie(conteo_tipos.values(), labels=conteo_tipos.keys(), autopct='%1.1f%%')
        plt.title("Conteo por tipo")
        plt.savefig(tmp1.name)
        plt.close()
        pdf.image(tmp1.name, x=10, y=30, w=80)
    os.unlink(tmp1.name)

    # Gráfica 2: Line plot promedios por tipo
    preguntas = list(promedios_por_tipo[list(promedios_por_tipo.keys())[0]].keys())
    valores = {k: list(v.values()) for k,v in promedios_por_tipo.items()}
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
        plt.figure(figsize=(6,4))
        for tipo, vals in valores.items():
            plt.plot(preguntas, vals, marker='o', label=tipo)
        plt.title("Promedios por tipo")
        plt.legend()
        plt.savefig(tmp2.name)
        plt.close()
        pdf.image(tmp2.name, x=110, y=30, w=80)
    os.unlink(tmp2.name)


    # Página de tabla (primeras 30 filas)
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Resultados (primeras 30 filas)", ln=True)
    pdf.set_font("Arial", size=10)

    col_names = df.columns.tolist()
    col_width = pdf.w / (len(col_names) + 1)

    for col in col_names:
        pdf.cell(col_width, 8, col, border=1)
    pdf.ln()

    for i, row in df.head(30).iterrows():
        for val in row:
            texto = str(val)
            if len(texto) > 15:
                texto = texto[:12] + "..."
            pdf.cell(col_width, 8, texto, border=1)
        pdf.ln()

    pdf.output(output_path)




@app.route('/descargar/<filename>')
def descargar(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

@app.route('/eliminar_modelo/<modelname>')
def eliminar_modelo(modelname):
    try:
        model_path = os.path.join(MODEL_FOLDER, modelname)
        if os.path.exists(model_path):
            os.remove(model_path)
            flash(f'Modelo {modelname} eliminado correctamente', 'success')
        else:
            flash('El modelo no existe', 'danger')
    except Exception as e:
        flash(f'Error al eliminar el modelo: {str(e)}', 'danger')
    
    return redirect(url_for('modelos'))

@app.route('/metricas_modelo/<nombre>')
def metricas_modelo(nombre):
    model_path = os.path.join(MODEL_FOLDER, nombre)
    if not os.path.exists(model_path):
        return jsonify({"error": "Modelo no encontrado"}), 404
    
    model_data = joblib.load(model_path)
    return jsonify({
        'nombre': nombre,
        'precision': model_data.get('accuracy', 0),
        'fecha': model_data['training_date'],
        'metricas': model_data.get('report', {})
    })

@app.route('/seleccionar_modelo/<nombre>', methods=['POST'])
def seleccionar_modelo(nombre):
    session['modelo_seleccionado'] = nombre  # Usa Flask-Session
    return jsonify({
        'status': 'success',
        'modelo': nombre,
        'message': 'Modelo seleccionado para predicciones'
    })