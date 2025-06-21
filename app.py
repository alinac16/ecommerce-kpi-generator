# app.py (updated)
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import requests
import os
import pandas as pd
from io import StringIO, BytesIO
import base64
import json # Import json for parsing AI response

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# IMPORTANT: Adjust origins to your frontend URL in production
# For local development, allow specific origins including IPv6 localhost
CORS(app, origins=[
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "http://[::]:8000",
    "http://[::1]:8000",
    "http://[::]:5500",
    "http://[::1]:5500",
    "https://nanacoffeeroasters.github.io" # Example GitHub Pages URL
])

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")
    # In a real production app, you might want to raise an error and prevent startup

# --- AI KPI Suggestion Endpoint ---
@app.route('/suggest-kpis', methods=['POST'])
def suggest_kpis():
    if not GEMINI_API_KEY:
        return jsonify({"error": "API Key not configured on server."}), 500

    try:
        data = request.get_json()
        headers = data.get('headers')

        if not headers:
            return jsonify({"error": "Missing 'headers' in request."}), 400

        headers_str = ", ".join(headers)
        prompt = f"""You are an expert e-commerce data analyst assistant.
Given the following CSV headers from an e-commerce orders table, suggest 3 to 5 common and useful Key Performance Indicators (KPIs) that can be calculated from this data.
For each KPI, provide its name, a simple equation using the provided headers, a brief English description of what the KPI shows and what a good number generally looks like for an e-commerce business, and a brief Thai description.

Format your response as a JSON array of objects. Each object must have the following keys:
"name": (string, e.g., "Total Sales")
"equation": (string, e.g., "SUM(Price)")
"english_description": (string, e.g., "Total revenue generated from all sales. A higher number indicates more sales activity and revenue.")
"thai_description": (string, e.g., "รายได้รวมที่มาจากการขายทั้งหมด ตัวเลขที่สูงขึ้นแสดงถึงกิจกรรมการขายและรายได้ที่มากขึ้น")

Example Headers: Order ID, Product Name, Quantity, Price, Customer ID, Order Date, Category, City

Example Output (JSON):
```json
[
  {{
    "name": "Total Sales",
    "equation": "SUM(Price)",
    "english_description": "Total revenue generated from all sales. A higher number indicates more sales activity and revenue.",
    "thai_description": "รายได้รวมที่มาจากการขายทั้งหมด ตัวเลขที่สูงขึ้นแสดงถึงกิจกรรมการขายและรายได้ที่มากขึ้น"
  }},
  {{
    "name": "Average Order Value (AOV)",
    "equation": "SUM(Price) / COUNT(DISTINCT Order ID)",
    "english_description": "The average amount spent per order. A higher AOV means customers are spending more each time they purchase.",
    "thai_description": "ยอดใช้จ่ายเฉลี่ยต่อคำสั่งซื้อ AOV ที่สูงขึ้นหมายความว่าลูกค้าใช้จ่ายมากขึ้นในการซื้อแต่ละครั้ง"
  }},
  {{
    "name": "Number of Unique Customers",
    "equation": "COUNT(DISTINCT Customer ID)",
    "english_description": "The total count of unique customers who made purchases. A growing number suggests successful customer acquisition.",
    "thai_description": "จำนวนลูกค้ารวมที่ไม่ซ้ำกันที่ทำการซื้อ ตัวเลขที่เพิ่มขึ้นแสดงถึงการหาลูกค้าใหม่ที่ประสบความสำเร็จ"
  }}
]
```

Current Headers: {headers_str}
Suggested KPIs (JSON Array):
"""
        chat_history = []
        chat_history.append({ "role": "user", "parts": [{ "text": prompt }] })

        gemini_payload = {
            "contents": chat_history,
            "generationConfig": {
                "temperature": 0.1, # Keep temperature low for more consistent JSON/structured output
            },
        }

        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers={'Content-Type': 'application/json'},
            json=gemini_payload
        )
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        gemini_response = response.json()

        suggested_kpis_raw_text = gemini_response['candidates'][0]['content']['parts'][0]['text'].strip()

        # Attempt to parse JSON. Gemini might wrap it in markdown.
        if suggested_kpis_raw_text.startswith('```json'):
            suggested_kpis_raw_text = suggested_kpis_raw_text[len('```json'):]
            if suggested_kpis_raw_text.endswith('```'):
                suggested_kpis_raw_text = suggested_kpis_raw_text[:-len('```')]
        
        suggested_kpis_raw_text = suggested_kpis_raw_text.strip()
        
        try:
            suggested_kpis_list = json.loads(suggested_kpis_raw_text)
            # Basic validation to ensure it's a list of dicts with expected keys
            if not isinstance(suggested_kpis_list, list) or \
               not all(isinstance(item, dict) and 
                       all(key in item for key in ["name", "equation", "english_description", "thai_description"])
                       for item in suggested_kpis_list):
                raise ValueError("AI returned unexpected JSON structure.")

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            print(f"Raw AI Response: {suggested_kpis_raw_text}")
            return jsonify({"error": "AI response was not valid JSON. Please try again or refine headers."}), 500
        except ValueError as e:
            print(f"AI response validation error: {e}")
            return jsonify({"error": "AI returned unexpected KPI structure. Please try again."}), 500


        return jsonify({"kpis": suggested_kpis_list}), 200

    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API for KPI suggestions: {e}")
        return jsonify({"error": f"Failed to get KPI suggestions from AI: {e}"}), 500
    except Exception as e:
        print(f"An unexpected error occurred in /suggest-kpis: {e}")
        import traceback
        traceback.print_exc() # For debugging
        return jsonify({"error": f"Internal server error: {e}"}), 500

# --- KPI Calculation and Excel Generation Endpoint ---
@app.route('/generate-excel', methods=['POST'])
def generate_excel():
    try:
        data = request.get_json()
        base64_csv_data = data.get('csv_data')
        selected_kpis = data.get('selected_kpis') # These are just names, not full objects

        if not base64_csv_data or not selected_kpis:
            return jsonify({"error": "Missing CSV data or selected KPIs."}), 400

        # Decode Base64 CSV data
        csv_bytes = base64.b64decode(base64_csv_data)
        csv_string = csv_bytes.decode('utf-8')
        df = pd.read_csv(StringIO(csv_string))

        # Ensure 'Order Date' is datetime if present for time-based KPIs
        if 'Order Date' in df.columns:
            df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
            df.dropna(subset=['Order Date'], inplace=True) # Drop rows where date conversion failed

        # Create an Excel writer object in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Raw Data
            df.to_excel(writer, sheet_name='Raw Data', index=False)

            # Sheet 2: Calculated KPIs
            kpi_results = {}

            # --- KPI Calculation Logic ---
            # This logic will need to handle the KPI names coming from the AI's suggestions.
            # You must ensure the KPI names generated by the AI match these conditions
            # and that the necessary columns exist in your CSV.
            # IMPORTANT: Expand this section based on the range of KPIs AI might suggest.

            if 'Total Sales' in selected_kpis and 'Price' in df.columns:
                kpi_results['Total Sales'] = df['Price'].sum()
            elif 'Total Sales' in selected_kpis and 'Quantity' in df.columns and 'Price' in df.columns:
                # Fallback if 'Price' not directly available but 'Quantity' and 'Price' (per unit) are
                df['Total Item Price'] = df['Quantity'] * df['Price']
                kpi_results['Total Sales'] = df['Total Item Price'].sum()

            if 'Total Quantity Sold' in selected_kpis and 'Quantity' in df.columns:
                kpi_results['Total Quantity Sold'] = df['Quantity'].sum()

            if 'Average Order Value (AOV)' in selected_kpis and 'Price' in df.columns and 'Order ID' in df.columns:
                sales_per_order = df.groupby('Order ID')['Price'].sum()
                kpi_results['Average Order Value (AOV)'] = sales_per_order.mean()

            if 'Number of Unique Customers' in selected_kpis and 'Customer ID' in df.columns:
                kpi_results['Number of Unique Customers'] = df['Customer ID'].nunique()

            if 'Top Selling Products' in selected_kpis and 'Item Name' in df.columns and 'Price' in df.columns:
                top_products = df.groupby('Item Name')['Price'].sum().nlargest(10) # Top 10 by sales
                kpi_results['Top 10 Selling Products (by Sales)'] = top_products.to_frame()

            if 'Sales by Product Category' in selected_kpis and 'Product Category' in df.columns and 'Price' in df.columns:
                sales_by_category = df.groupby('Product Category')['Price'].sum()
                kpi_results['Sales by Product Category'] = sales_by_category.to_frame()

            if 'Monthly Sales Trend' in selected_kpis and 'Order Date' in df.columns and 'Price' in df.columns:
                df_monthly = df.set_index('Order Date').resample('M')['Price'].sum().to_frame(name='Total Sales')
                kpi_results['Monthly Sales Trend'] = df_monthly

            # Add more KPI calculations here based on what you expect the AI to suggest and what your data contains
            # For each KPI result that is a DataFrame/Series, write it to the Excel sheet
            kpi_dataframes = {}
            kpi_scalars = {}

            for kpi_name, value in kpi_results.items():
                if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
                    kpi_dataframes[kpi_name] = value
                else:
                    kpi_scalars[kpi_name] = value

            # Write scalar KPIs first (e.g., Total Sales, AOV)
            if kpi_scalars:
                scalar_df = pd.DataFrame.from_dict(kpi_scalars, orient='index', columns=['Value'])
                scalar_df.index.name = 'KPI'
                # Ensure we have data before writing
                if not scalar_df.empty:
                    scalar_df.to_excel(writer, sheet_name='Calculated KPIs', startrow=0, startcol=0)
                    next_row = len(scalar_df) + 2 # Start next table after 2 blank rows
                else:
                    next_row = 0
            else:
                next_row = 0

            # Write DataFrame KPIs below (e.g., Top Products, Monthly Trends)
            for kpi_name, df_value in kpi_dataframes.items():
                if not df_value.empty:
                    # Add a header for this section
                    df_header = pd.DataFrame([kpi_name], columns=[''])
                    df_header.to_excel(writer, sheet_name='Calculated KPIs', startrow=next_row, startcol=0, header=False, index=False)
                    next_row += 1 # for the header row

                    # Write the DataFrame itself
                    df_value.to_excel(writer, sheet_name='Calculated KPIs', startrow=next_row, startcol=0)
                    next_row += len(df_value) + 3 # Add space for next table

        output.seek(0) # Rewind to the beginning of the stream

        return send_file(output, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', as_attachment=True, download_name='ecommerce_kpis.xlsx')

    except Exception as e:
        print(f"An error occurred during Excel generation: {e}")
        import traceback
        traceback.print_exc() # For debugging
        return jsonify({"error": f"Failed to generate Excel file: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
