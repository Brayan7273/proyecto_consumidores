from flask import abort, jsonify, render_template, request, redirect, send_file, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime, time
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
from flask import session
from flask import send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from flask import g  # Añade esto al inicio con los otros imports
from apscheduler.schedulers.background import BackgroundScheduler
import atexit

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'app', 'data', 'uploads')
MODEL_FOLDER = os.path.join(BASE_DIR, 'app', 'data', 'models')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'app', 'data', 'outputs')
REPORT_FOLDER = os.path.join(BASE_DIR, 'app', 'data', 'reports')
# Crear carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

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
app.config['REPORT_FOLDER'] = REPORT_FOLDER

df_global = None

# Extensiones permitidas
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv','pkl', 'joblib', 'sav'}

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
    datos_template = None
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
            # Guardar archivo temporalmente
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Leer datos
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Validar estructura
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
            
            # Preparar datos para template y reporte
            datos_template = {
                'columnas': df.columns.tolist(),
                'filas': df.head(10).to_dict('records'),
                'total_registros': len(df),
                'dataframe': df.to_json()  # Guardamos el DataFrame serializado
            }
            
            modelo_info = {
                'algoritmo': 'Random Forest',
                'accuracy': accuracy,
                'fecha': timestamp,
                'tamanio': os.path.getsize(model_path)/(1024*1024),  # MB
                'metricas': report,
                'distribucion': df['Tipo_Consumidor'].value_counts().to_dict()
            }
            
            # Guardar datos en sesión para el reporte
            session['modelo_info'] = modelo_info
            session['datos_reporte'] = datos_template
            
            flash('Modelo entrenado exitosamente!', 'success')
            
        except Exception as e:
            flash(f'Error al procesar el archivo: {str(e)}', 'danger')
            return redirect(url_for('entrenar'))
    
    return render_template(
        'entrenar.html',
        datos=datos_template,
        modelo_info=modelo_info
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

def allowed_file(filename):
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    print(f"Extensión del archivo: {ext}")
    print(f"Permitir extensiones: {ALLOWED_EXTENSIONS}")
    return ext in ALLOWED_EXTENSIONS

@app.route('/cargar_modelo', methods=['POST'])
def cargar_modelo():
    if 'modeloArchivo' not in request.files:
        print("No hay archivo en request.files")
        flash('No se seleccionó ningún archivo', 'danger')
        return redirect(url_for('modelos'))
    
    file = request.files['modeloArchivo']
    print(f"Archivo recibido: {file.filename}")
    
    if file.filename == '':
        print("Nombre de archivo vacío")
        flash('No se seleccionó ningún archivo', 'danger')
        return redirect(url_for('modelos'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(MODEL_FOLDER, filename)
        print(f"Guardando archivo en: {save_path}")
        try:
            file.save(save_path)
            print("Archivo guardado correctamente")
            flash(f'Modelo "{filename}" cargado correctamente', 'success')
        except Exception as e:
            print(f"Error guardando archivo: {e}")
            flash(f'Error al guardar el archivo: {e}', 'danger')
    else:
        print("Archivo no permitido")
        flash('Archivo no permitido. Solo formatos: .pkl, .joblib, .sav', 'danger')
    
    return redirect(url_for('modelos'))



@app.route('/descargar_modelo_seleccionado/<modelname>')
def descargar_modelo_seleccionado(modelname):
    try:
        # Asegurarse que el archivo exista
        if not os.path.isfile(os.path.join(MODEL_FOLDER, modelname)):
            abort(404, description="Archivo no encontrado")

        # Enviar el archivo para descarga
        return send_from_directory(MODEL_FOLDER, modelname, as_attachment=True)
    except Exception as e:
        # Manejo básico de errores
        abort(500, description=f"Error al descargar el archivo: {str(e)}")

@app.route('/generar_reporte/<formato>')
def generar_reporte(formato):
    # Obtener datos de la sesión
    modelo_info = session.get('modelo_info')
    datos_reporte = session.get('datos_reporte')
    
    if not modelo_info or not datos_reporte:
        flash('No hay datos de modelo para generar reporte', 'danger')
        return redirect(url_for('entrenar'))
    
    try:
        # Convertir datos a DataFrame
        df = pd.read_json(datos_reporte['dataframe'])
        
        # Crear gráficos adicionales
        plt.switch_backend('Agg')  # Para evitar problemas con matplotlib en threads
        
        # Gráfico 1: Distribución de tipos
        fig1 = plt.figure(figsize=(8, 6))
        labels = list(modelo_info['distribucion'].keys())
        sizes = list(modelo_info['distribucion'].values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.title('Distribución de Tipos de Consumidores')
        plt.axis('equal')
        buf1 = io.BytesIO()
        plt.savefig(buf1, format='png', bbox_inches='tight')
        plt.close(fig1)
        
        # Gráfico 2: Matriz de correlación
        fig2 = plt.figure(figsize=(10, 8))
        numeric_cols = [col for col in df.columns if col.startswith('P')]
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación entre Variables')
        buf2 = io.BytesIO()
        plt.savefig(buf2, format='png', bbox_inches='tight')
        plt.close(fig2)
        
        # Gráfico 3: Boxplot de variables importantes
        fig3 = plt.figure(figsize=(10, 6))
        top_vars = corr.abs().mean().sort_values(ascending=False).head(5).index
        df_melt = pd.melt(df[list(top_vars) + ['Tipo_Consumidor']], 
                         id_vars='Tipo_Consumidor',
                         var_name='Variable',
                         value_name='Valor')
        sns.boxplot(x='Variable', y='Valor', hue='Tipo_Consumidor', data=df_melt)
        plt.title('Distribución de Variables por Tipo de Consumidor')
        plt.xticks(rotation=45)
        buf3 = io.BytesIO()
        plt.savefig(buf3, format='png', bbox_inches='tight')
        plt.close(fig3)
        
        if formato == 'pdf':
            # Crear PDF
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            
            styles = getSampleStyleSheet()
            elements.append(Paragraph("Reporte del Modelo de Consumidores", styles['Title']))
            elements.append(Paragraph(f"Fecha: {modelo_info['fecha']}", styles['Normal']))
            
            # Información del modelo
            model_data = [
                ["Algoritmo", modelo_info['algoritmo']],
                ["Precisión", f"{modelo_info['accuracy']*100:.2f}%"],
                ["Tamaño del modelo", f"{modelo_info['tamanio']:.2f} MB"]
            ]
            model_table = Table(model_data)
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(model_table)
            
            # Métricas detalladas
            elements.append(Paragraph("\nMétricas Detalladas", styles['Heading2']))
            metric_data = [["Tipo", "Precisión", "Recall", "F1-Score"]]
            
            for tipo, metrica in modelo_info['metricas'].items():
                if tipo not in ['accuracy', 'macro avg', 'weighted avg']:
                    metric_data.append([
                        tipo,
                        f"{metrica['precision']:.2f}",
                        f"{metrica['recall']:.2f}",
                        f"{metrica['f1-score']:.2f}"
                    ])
            
            metric_table = Table(metric_data)
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 0), (-1, -1), 10)
            ]))
            elements.append(metric_table)
            
            # Agregar gráficos al PDF
            elements.append(Paragraph("\nAnálisis Gráfico", styles['Heading2']))
            
            for i, (buf, title) in enumerate(zip(
                [buf1, buf2, buf3],
                ['Distribución de Tipos', 'Matriz de Correlación', 'Boxplot por Tipo']
            )):
                buf.seek(0)
                elements.append(Paragraph(title, styles['Heading3']))
                img = Image(buf, width=400, height=300)
                elements.append(img)
            
            # Construir PDF
            doc.build(elements)
            buffer.seek(0)
            
            # Guardar copia del reporte
            report_filename = f"reporte_{modelo_info['fecha']}.pdf"
            report_path = os.path.join(REPORT_FOLDER, report_filename)
            with open(report_path, 'wb') as f:
                f.write(buffer.getbuffer())
            
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'reporte_modelo_{modelo_info["fecha"]}.pdf',
                mimetype='application/pdf'
            )
            
        elif formato == 'excel':
            # Crear Excel con gráficos
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                # Hoja 1: Datos
                df.to_excel(writer, sheet_name='Datos', index=False)
                
                # Hoja 2: Métricas
                metricas = []
                for tipo, metrica in modelo_info['metricas'].items():
                    if tipo not in ['accuracy', 'macro avg', 'weighted avg']:
                        metricas.append({
                            'Tipo': tipo,
                            'Precisión': metrica['precision'],
                            'Recall': metrica['recall'],
                            'F1-Score': metrica['f1-score']
                        })
                pd.DataFrame(metricas).to_excel(writer, sheet_name='Métricas', index=False)
                
                # Hoja 3: Resumen del Modelo
                resumen_data = [
                    ['Algoritmo', modelo_info['algoritmo']],
                    ['Precisión', modelo_info['accuracy']],
                    ['Fecha', modelo_info['fecha']],
                    ['Tamaño del modelo (MB)', modelo_info['tamanio']]
                ]
                resumen_df = pd.DataFrame(resumen_data, columns=['Atributo', 'Valor'])
                resumen_df.to_excel(writer, sheet_name='Resumen', index=False)
                
                # Agregar gráficos
                workbook = writer.book
                worksheet = workbook.add_worksheet('Gráficos')
                
                # Insertar imágenes
                for i, (buf, title) in enumerate(zip(
                    [buf1, buf2, buf3],
                    ['Distribución de Tipos', 'Matriz de Correlación', 'Boxplot por Tipo']
                )):
                    buf.seek(0)
                    worksheet.insert_image(
                        f'A{1+i*20}',
                        '',
                        {'image_data': buf, 'x_scale': 0.5, 'y_scale': 0.5}
                    )
                    worksheet.write(f'A{1+i*20+15}', title)
            
            buffer.seek(0)
            
            # Guardar copia del reporte
            report_filename = f"reporte_{modelo_info['fecha']}.xlsx"
            report_path = os.path.join(REPORT_FOLDER, report_filename)
            with open(report_path, 'wb') as f:
                f.write(buffer.getbuffer())
            
            buffer.seek(0)
            return send_file(
                buffer,
                as_attachment=True,
                download_name=f'reporte_modelo_{modelo_info["fecha"]}.xlsx',
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
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

@app.route('/eliminar_modelo/<modelname>', methods=['POST'])
def eliminar_modelo(modelname):
    try:
        model_path = os.path.join(MODEL_FOLDER, modelname)
        if os.path.exists(model_path):
            os.remove(model_path)
            return jsonify({"status": "success", "message": f"Modelo {modelname} eliminado correctamente"})
        else:
            return jsonify({"status": "error", "message": "El modelo no existe"}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error al eliminar el modelo: {str(e)}"}), 500

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

@app.route('/seleccionar_modelo/<nombre>', methods=['GET'])
def seleccionar_modelo(nombre):
    session['modelo_seleccionado'] = nombre
    return redirect(url_for('/predecir'))  


def limpiar_reportes_temporales():
    temp_dir = os.path.join(app.root_path, 'temp_reports')
    if os.path.exists(temp_dir):
        now = time.time()
        for f in os.listdir(temp_dir):
            f_path = os.path.join(temp_dir, f)
            if os.stat(f_path).st_mtime < now - 3600:  # Eliminar archivos con más de 1 hora
                os.remove(f_path)

# Programar limpieza cada hora
scheduler = BackgroundScheduler()
scheduler.add_job(func=limpiar_reportes_temporales, trigger='interval', hours=1)
scheduler.start()

# Asegurar que el programador se apague al salir
atexit.register(lambda: scheduler.shutdown())