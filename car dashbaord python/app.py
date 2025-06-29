from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import os
from google.cloud import dialogflow_v2 as dialogflow

# Set Matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend

import matplotlib.pyplot as plt

app = Flask(__name__)

# Set the path to your Dialogflow service account key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/service-account-file.json'

# Function to generate plots
def create_plot(data, title, filename):
    plt.figure(figsize=(10, 6))
    plt.bar(data.keys(), data.values(), color='skyblue')
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Function to predict maintenance needs
def predict_maintenance(df):
    maintenance_data = {}
    for index, row in df.iterrows():
        if row['miles_driven'] > 10000:
            maintenance_data[row['vehicle_id']] = 'Maintenance needed soon'
    return maintenance_data

# Function to process the uploaded CSV file
def process_data(file_path):
    df = pd.read_csv(file_path)

    sales_profit = int(df['sales_profit'].sum())
    
    monthly_sales = df.groupby('month')['sales_profit'].sum().astype(int).to_dict()
    type_counts = df['vehicle_type'].value_counts().astype(int).to_dict()
    age_distribution = df['customer_age'].value_counts().astype(int).to_dict()
    sales_channel_counts = df['sales_channel'].value_counts().astype(int).to_dict()
    gender_distribution = df['customer_gender'].value_counts().astype(int).to_dict()
    income_distribution = {
        'min': float(df['customer_income'].min()),
        'max': float(df['customer_income'].max()),
        'mean': float(df['customer_income'].mean())
    }
    color_distribution = df['vehicle_color'].value_counts().astype(int).to_dict()

    maintenance_predictions = predict_maintenance(df)

    create_plot(monthly_sales, 'Monthly Sales Profit', 'static/monthly_sales.png')
    create_plot(type_counts, 'Vehicle Type Counts', 'static/type_counts.png')
    create_plot(age_distribution, 'Customer Age Distribution', 'static/age_distribution.png')

    location_data = {
         'Area 1': {'lat': 13.067439, 'lng': 80.237617, 'details': '10 SUVs sold'},
        'Area 2': {'lat': 13.048926, 'lng': 80.218171, 'details': '5 Sedans sold'},
        'Area 3': {'lat': 13.058475, 'lng': 80.270160, 'details': '8 Hatchbacks sold'},
        'Area 4': {'lat': 13.100000, 'lng': 80.200000, 'details': '7 SUVs sold'},
        'Area 5': {'lat': 13.067439, 'lng': 80.287617, 'details': '12 Sedans sold'},
        'Area 6': {'lat': 13.047926, 'lng': 80.218171, 'details': '15 Hatchbacks sold'},
        'Area 7': {'lat': 13.068475, 'lng': 80.250160, 'details': '9 SUVs sold'},
        'Area 8': {'lat': 13.090000, 'lng': 80.210000, 'details': '14 Vans sold'},
        'Area 9': {'lat': 13.110000, 'lng': 80.230000, 'details': '11 SUVs sold'},
        'Area 10': {'lat': 13.130000, 'lng': 80.240000, 'details': '6 Coupes sold'},
        'Area 11': {'lat': 13.140000, 'lng': 80.250000, 'details': '13 Trucks sold'},
        'Area 12': {'lat': 13.150000, 'lng': 80.260000, 'details': '10 Minivans sold'},
        'Area 13': {'lat': 13.160000, 'lng': 80.270000, 'details': '4 Convertibles sold'},
        'Area 14': {'lat': 13.170000, 'lng': 80.280000, 'details': '9 Sports Cars sold'},
        'Area 15': {'lat': 13.180000, 'lng': 80.290000, 'details': '8 Crossovers sold'},
    }

    # Prepare data for the 3D chart
    x_values = df['sales_profit'].tolist()  # Example x values
    y_values = df['customer_age'].tolist()  # Example y values
    z_values = df['customer_income'].tolist()  # Example z values

    return {
        'sales_profit': sales_profit,
        'monthly_sales': monthly_sales,
        'maintenance_predictions': maintenance_predictions,
        'type_counts': type_counts,
        'age_distribution': age_distribution,
        'sales_channel_counts': sales_channel_counts,
        'gender_distribution': gender_distribution,
        'income_distribution': income_distribution,
        'color_distribution': color_distribution,
        'locations': location_data,
        'x': x_values,
        'y': y_values,
        'z': z_values,
    }

# Function to detect intent from user input
def detect_intent_texts(project_id, session_id, texts, language_code):
    session_client = dialogflow.SessionsClient()
    session = session_client.session_path(project_id, session_id)

    text_input = dialogflow.TextInput(text=texts, language_code=language_code)
    query_input = dialogflow.QueryInput(text=text_input)

    response = session_client.detect_intent(session=session, query_input=query_input)
    return response.query_result.fulfillment_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File is not a CSV'}), 400

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        data = process_data(file_path)
        return jsonify(data)
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'No data in the file'}), 400
    except pd.errors.ParserError:
        return jsonify({'error': 'Error parsing the file'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.route('/plots/<filename>')
def plots(filename):
    file_path = os.path.join('static', filename)
    if os.path.exists(file_path):
        return send_file(file_path, mimetype='image/png')
    return jsonify({'error': 'File not found'}), 404

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message')
    project_id = 'your-project-id'
    session_id = 'unique-session-id'
    language_code = 'en'

    response_text = detect_intent_texts(project_id, session_id, user_message, language_code)
    return jsonify({'response': response_text})

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, threaded=False)
