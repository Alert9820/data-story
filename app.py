import os
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import google.generativeai as genai
import json
import warnings
import io
from decimal import Decimal
warnings.filterwarnings('ignore')

app = Flask(__name__, template_folder='templates')

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
app.config['CLEANED_FOLDER'] = '/tmp/cleaned'

# Create folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CLEANED_FOLDER'], exist_ok=True)

# Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')

ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    elif isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_data(df):
    """Complete data cleaning pipeline"""
    original_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Detect and convert dates
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    # Handle missing values
    for col in df.columns:
        missing_pct = df[col].isnull().sum() / len(df) * 100
        if missing_pct > 50:
            df = df.drop(columns=[col])
        elif df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
    
    # Remove outliers (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    # Remove empty rows/columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    cleaning_report = {
        'original_rows': int(original_shape[0]),
        'original_cols': int(original_shape[1]),
        'cleaned_rows': int(len(df)),
        'cleaned_cols': int(len(df.columns)),
        'rows_removed': int(original_shape[0] - len(df)),
        'cols_removed': int(original_shape[1] - len(df.columns))
    }
    
    return df, cleaning_report

def generate_eda(df):
    """Generate EDA report"""
    eda = {
        'total_rows': int(len(df)),
        'total_cols': int(len(df.columns)),
        'numeric_cols': int(len(df.select_dtypes(include=[np.number]).columns)),
        'categorical_cols': int(len(df.select_dtypes(include=['object']).columns)),
        'date_cols': int(len(df.select_dtypes(include=['datetime64']).columns)),
        'total_missing': int(df.isnull().sum().sum())
    }
    
    # Missing values by column
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    eda['missing_by_column'] = convert_to_serializable(missing_data.to_dict())
    
    # Numeric statistics
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        stats = {}
        for col in numeric_df.columns:
            stats[col] = {
                'mean': float(round(numeric_df[col].mean(), 2)),
                'std': float(round(numeric_df[col].std(), 2)),
                'min': float(round(numeric_df[col].min(), 2)),
                'max': float(round(numeric_df[col].max(), 2))
            }
        eda['numeric_stats'] = stats
        
        # Correlations
        corr_matrix = numeric_df.corr()
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    correlations.append({
                        'col1': str(corr_matrix.columns[i]),
                        'col2': str(corr_matrix.columns[j]),
                        'value': float(round(corr_value, 2))
                    })
        eda['top_correlations'] = correlations[:5]
    
    return eda

def generate_charts(df):
    """Generate Plotly charts"""
    charts = {}
    
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(include=['object'])
    date_cols = df.select_dtypes(include=['datetime64']).columns
    
    # Correlation Heatmap
    if len(numeric_df.columns) >= 2:
        corr_matrix = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title='Correlation Heatmap',
            template='plotly_dark',
            height=500,
            title_font_size=16
        )
        charts['heatmap'] = json.loads(plotly.io.to_json(fig))
    
    # Histograms for numeric columns
    if not numeric_df.empty:
        cols = list(numeric_df.columns)[:3]
        for col in cols:
            fig = px.histogram(
                df, x=col, 
                title=f'Distribution of {col}',
                template='plotly_dark',
                color_discrete_sequence=['#00ff88']
            )
            fig.update_layout(height=400, showlegend=False)
            charts[f'hist_{col}'] = json.loads(plotly.io.to_json(fig))
    
    # Bar charts for categorical
    if not categorical_df.empty:
        col = categorical_df.columns[0]
        top_values = df[col].value_counts().head(10)
        fig = px.bar(
            x=top_values.index, 
            y=top_values.values,
            title=f'Top Values in {col}',
            template='plotly_dark',
            color_discrete_sequence=['#ff6b6b']
        )
        fig.update_layout(height=400, xaxis_title=col, yaxis_title='Count')
        charts['bar_chart'] = json.loads(plotly.io.to_json(fig))
    
    # Time series
    if len(date_cols) > 0 and not numeric_df.empty:
        date_col = date_cols[0]
        numeric_col = numeric_df.columns[0]
        time_data = df.groupby(date_col)[numeric_col].mean().reset_index()
        fig = px.line(
            time_data, x=date_col, y=numeric_col,
            title=f'{numeric_col} Trend Over Time',
            template='plotly_dark',
            color_discrete_sequence=['#4ecdc4']
        )
        fig.update_layout(height=400)
        charts['timeseries'] = json.loads(plotly.io.to_json(fig))
    
    # Pie chart
    if not categorical_df.empty:
        col = categorical_df.columns[0]
        top_cats = df[col].value_counts().head(5)
        fig = px.pie(
            values=top_cats.values, names=top_cats.index,
            title=f'{col} Distribution',
            template='plotly_dark',
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        fig.update_layout(height=400)
        charts['pie_chart'] = json.loads(plotly.io.to_json(fig))
    
    # Scatter plot (removed trendline to avoid statsmodels)
    if len(numeric_df.columns) >= 2:
        fig = px.scatter(
            df, x=numeric_df.columns[0], y=numeric_df.columns[1],
            title=f'{numeric_df.columns[0]} vs {numeric_df.columns[1]}',
            template='plotly_dark',
            color_discrete_sequence=['#ffe66d']
        )
        fig.update_layout(height=400)
        charts['scatter'] = json.loads(plotly.io.to_json(fig))
    
    return charts

def generate_insights(df, eda, cleaning_report):
    """Generate AI insights"""
    if not GEMINI_API_KEY:
        return "⚠️ Gemini API key not configured. Please add GEMINI_API_KEY to environment variables for AI-powered insights."
    
    context = f"""
    Dataset: {eda['total_rows']} rows, {eda['total_cols']} columns
    Cleaning: Removed {cleaning_report['rows_removed']} rows, {cleaning_report['cols_removed']} columns
    Data Types: {eda['numeric_cols']} numeric, {eda['categorical_cols']} categorical, {eda['date_cols']} date
    Missing Values: {eda['total_missing']} total
    """
    
    prompt = f"""
    As a business analyst, provide insights for this dataset:
    {context}
    
    Provide in this exact format:
    
    📊 EXECUTIVE SUMMARY:
    (2-3 sentences on key findings)
    
    💡 KEY INSIGHTS:
    • Insight 1
    • Insight 2
    • Insight 3
    • Insight 4
    
    🎯 RECOMMENDATIONS:
    • Recommendation 1
    • Recommendation 2
    • Recommendation 3
    
    ⚠️ RISKS & OPPORTUNITIES:
    • Risk/Opportunity 1
    • Risk/Opportunity 2
    
    📈 KPI SUGGESTIONS:
    • KPI 1
    • KPI 2
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠️ AI Insight Generation Failed: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use CSV or Excel.'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read file
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        
        # Clean data
        cleaned_df, cleaning_report = clean_data(df)
        
        # Save cleaned data
        cleaned_path = os.path.join(app.config['CLEANED_FOLDER'], f'cleaned_{filename}.csv')
        cleaned_df.to_csv(cleaned_path, index=False)
        
        # Generate EDA
        eda_report = generate_eda(cleaned_df)
        
        # Generate charts
        charts = generate_charts(cleaned_df)
        
        # Generate insights
        insights = generate_insights(cleaned_df, eda_report, cleaning_report)
        
        # KPI metrics
        numeric_df = cleaned_df.select_dtypes(include=[np.number])
        kpis = {
            'total_rows': int(len(cleaned_df)),
            'total_cols': int(len(cleaned_df.columns)),
            'completeness': float(round((1 - cleaned_df.isnull().sum().sum() / (len(cleaned_df) * len(cleaned_df.columns))) * 100, 2)),
            'numeric_columns': int(len(numeric_df.columns))
        }
        
        if not numeric_df.empty:
            kpis['avg_value'] = float(round(numeric_df[numeric_df.columns[0]].mean(), 2))
            kpis['max_value'] = float(round(numeric_df[numeric_df.columns[0]].max(), 2))
        
        # Convert preview data to serializable format
        preview_data = cleaned_df.head(20).to_dict('records')
        preview_data = convert_to_serializable(preview_data)
        
        # Store for download
        app.config['CURRENT_CLEANED'] = cleaned_path
        app.config['CURRENT_INSIGHTS'] = insights
        
        response_data = {
            'success': True,
            'filename': filename,
            'cleaning_report': cleaning_report,
            'eda_report': eda_report,
            'charts': charts,
            'insights': insights,
            'kpis': kpis,
            'preview': preview_data,
            'columns': [str(col) for col in cleaned_df.columns.tolist()]
        }
        
        return jsonify(convert_to_serializable(response_data))
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/cleaned')
def download_cleaned():
    if 'CURRENT_CLEANED' in app.config and os.path.exists(app.config['CURRENT_CLEANED']):
        return send_file(app.config['CURRENT_CLEANED'], as_attachment=True, download_name='cleaned_data.csv')
    return jsonify({'error': 'No cleaned data available'}), 400

@app.route('/download/insights')
def download_insights():
    if 'CURRENT_INSIGHTS' in app.config:
        insights_text = app.config['CURRENT_INSIGHTS']
        return send_file(
            io.BytesIO(insights_text.encode()),
            as_attachment=True,
            download_name='ai_insights.txt',
            mimetype='text/plain'
        )
    return jsonify({'error': 'No insights available'}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
