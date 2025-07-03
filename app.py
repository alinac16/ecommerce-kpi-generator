import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pandas.api.types as pd_types
import io
import json
import re
from dotenv import load_dotenv
import httpx

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

API_KEY = os.getenv("API_KEY", "")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

@app.route('/')
def index():
    return render_template('index.html')

async def call_gemini_api(prompt, generation_config=None):
    headers = {
        'Content-Type': 'application/json'
    }
    params = f'?key={API_KEY}' if API_KEY else ''

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }

    if generation_config:
        payload["generationConfig"] = generation_config
    else:
        payload["generationConfig"] = {"temperature": 0.1}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{GEMINI_API_URL}{params}", headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        if result.get('candidates') and result['candidates'][0].get('content') and result['candidates'][0]['content'].get('parts'):
            raw_text = result['candidates'][0]['content']['parts'][0]['text']
            if raw_text.startswith('```json'):
                raw_text = raw_text[len('```json'):]
                if raw_text.endswith('```'):
                    raw_text = raw_text[:-len('```')]
            return json.loads(raw_text.strip())
        else:
            print(f"AI API Response: {result}")
            return {"error": "Unexpected AI API response structure or no content."}
    except httpx.RequestError as e:
        print(f"Error calling Gemini API: {e}")
        return {"error": f"Failed to connect to AI service: {e}"}
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error from AI service: {e.response.status_code} - {e.response.text}")
        return {"error": f"AI service returned an error: {e.response.status_code}"}
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e} - Raw AI Response: {raw_text if 'raw_text' in locals() else 'N/A'}")
        return {"error": f"Failed to parse AI response: {e}"}
    except Exception as e:
        print(f"An unexpected error occurred in call_gemini_api: {e}")
        return {"error": f"An unexpected error occurred: {e}"}


def clean_column_name(col_name):
    """Cleans a column name for easier programmatic access."""
    cleaned = re.sub(r'\W+', '_', col_name).strip('_').lower()
    return cleaned

CURRENCY_PATTERN = re.compile(
    r'(?P<currency>'
    r'thb|yen|cad|usd|eur|gbp|jpy|cny|inr|rub|krw|pln|try|idr|sgd|myr|php|vnd|ils|mxn|chf|aud|nzd|zar|'
    r'\$|€|£|¥|₹|₽|฿|₩|zł|₺|rp|s\$|rm|₱|₫|₪|kr|a\$|c\$|nz\$)',
    re.IGNORECASE
)

def extract_currency_and_amount(value):
    """
    Extracts currency code/symbol from a string value and returns the cleaned numeric string
    along with the extracted currency.
    Returns a pandas Series (cleaned_numeric_string, currency_code_found).
    """
    if pd.isna(value):
        return pd.Series([None, None])
    s_value = str(value).strip()

    currency_found = None
    numeric_part = s_value

    match = CURRENCY_PATTERN.search(s_value)
    if match:
        currency_found = match.group('currency').upper()
        numeric_part = s_value.replace(match.group(0), '', 1).strip()

    cleaned_numeric_str = re.sub(r'[^\d.-]+', '', numeric_part)

    if cleaned_numeric_str.count('.') > 1:
        parts = cleaned_numeric_str.split('.')
        cleaned_numeric_str = parts[0] + '.' + ''.join(parts[1:])
        
    if cleaned_numeric_str.count('-') > 1:
        cleaned_numeric_str = '-' + cleaned_numeric_str.replace('-', '')

    number = None
    if cleaned_numeric_str:
        try:
            number = int(float(cleaned_numeric_str))
        except ValueError:
            number = None

    return pd.Series([currency_found, number])


# --- DEFINITIONS FOR AI MAPPING AND KPI CALCULATION ---

# Conceptual roles for columns (for AI mapping)
CONCEPTUAL_COLUMN_ROLES = {
    "Order ID": ["order_id", "invoice_id", "transaction_id"],
    "Product Name": ["product_name", "item_name", "product"],
    "Variation": ["variation", "product_variation"],
    "Quantity": ["quantity", "units", "items_sold", "item_count"],
    "Price": ["price", "item_price", "unit_price", "sku_unit_original_price"],
    "Sales Amount": ["sales", "revenue", "amount", "total", "subtotal", "order_total", "line_total", 
                     "sku_subtotal_before_discount", "sku_subtotal_after_discount", "order_amount",
                     "sku_platform_discount", "sku_seller_discount", "shipping_fee_after_discount", 
                     "original_shipping_fee", "shipping_fee_seller_discount", 
                     "shipping_fee_platform_discount", "taxes", "order_refund_amount"],
    "Customer ID": ["customer_id", "user_id", "buyer_id"],
    "Order Date": ["order_date", "date", "created_time", "paid_time", "rts_time", "shipped_time", 
                   "delivered_time", "cancelled_time", "transaction_date", "timestamp"],
    "Category": ["product_category", "category", "item_category"],
    "Brand": ["brand", "product_brand"],
    "Shipping Cost": ["shipping_cost", "delivery_fee"],
    "Discount Amount": ["discount_amount", "coupon_discount"],
    "Payment Method": ["payment_method", "payment_type"],
    "Country": ["country", "billing_country", "shipping_country"],
    "Province": ["province", "state"],
    "City": ["city", "billing_city", "shipping_city"],
    "Website Visits": ["website_visits", "sessions", "traffic", "page_views"],
    "Taxes": ["taxes"],
    "Order Refund Amount": ["order_refund_amount"],
    "Weight": ["weight_kg", "weight_g", "weight"],
    "Order Status": ["order_status", "order_status"],
    "Order Substatus": ["order_substatus", "order_substatus"],
    "Fulfillment Type": ["fulfillment_type", "fulfillment_type"],
    "Warehouse Name": ["warehouse_name", "warehouse_name"],
    "Tracking ID": ["tracking_id", "tracking_id"],
    "Delivery Option": ["delivery_option", "delivery_option"],
    "Shipping Provider Name": ["shipping_provider_name", "shipping_provider_name"],
    "Buyer Message": ["buyer_message", "buyer_message"],
    "Buyer Username": ["buyer_username", "username"],
    "Recipient": ["recipient", "customer_name"],
    "Phone #": ["phone_number", "phone"],
    "Zipcode": ["zipcode", "postcode"],
    "Detail Address": ["detail_address", "detail_address"],
    "Additional Address Information": ["additional_address_information", "additional_address_information"],
    "Package ID": ["package_id", "package_id"],
    "Seller Note": ["seller_note", "seller_note"],
    "Checked Status": ["checked_status", "checked_status"],
    "Checked Marked by": ["checked_marked_by", "checked_marked_by"],
    "SKU ID": ["sku_id"],
    "Seller SKU": ["seller_sku"],
    "Small Order Fee": ["small_order_fee"],
    "Cancel By": ["cancel_by"],
    "Cancel Reason": ["cancel_reason"]
}

# Simplified mapping for identifying columns that should be treated as numeric in cleaning
NUMERIC_CONCEPTUAL_ROLES_FOR_CLEANING = {
    "Sales Amount": CONCEPTUAL_COLUMN_ROLES["Sales Amount"],
    "Quantity": CONCEPTUAL_COLUMN_ROLES["Quantity"],
    "Price": CONCEPTUAL_COLUMN_ROLES["Price"],
    "Shipping Cost": CONCEPTUAL_COLUMN_ROLES["Shipping Cost"],
    "Discount Amount": CONCEPTUAL_COLUMN_ROLES["Discount Amount"],
    "Website Visits": CONCEPTUAL_COLUMN_ROLES["Website Visits"],
    "Taxes": CONCEPTUAL_COLUMN_ROLES["Taxes"],
    "Order Refund Amount": CONCEPTUAL_COLUMN_ROLES["Order Refund Amount"],
    "Weight": CONCEPTUAL_COLUMN_ROLES["Weight"],
    "Small Order Fee": CONCEPTUAL_COLUMN_ROLES["Small Order Fee"]
}


def perform_data_cleaning(df_original):
    """
    Performs comprehensive data cleaning and standardization on the DataFrame.
    Includes enhanced currency extraction and numeric conversion, creating new
    '[original_column_name]_currency' columns for extracted currencies.
    """
    df = df_original.copy()
    df.columns = [clean_column_name(col) for col in df.columns]

    new_currency_columns = []
    
    potential_numeric_cols_in_df = set()
    for role_aliases in NUMERIC_CONCEPTUAL_ROLES_FOR_CLEANING.values():
        for alias in role_aliases:
            for col in df.columns:
                if alias in col:
                    potential_numeric_cols_in_df.add(col)
    
    for col in potential_numeric_cols_in_df:
        if pd_types.is_object_dtype(df[col]) or pd_types.is_string_dtype(df[col]):
            currency_and_numeric_parts = df[col].apply(extract_currency_and_amount)
            
            extracted_currencies_for_col = currency_and_numeric_parts.iloc[:, 0]
            numeric_values_for_col = currency_and_numeric_parts.iloc[:, 1]

            currency_col_name = f"{col}_currency"
            df[currency_col_name] = extracted_currencies_for_col
            new_currency_columns.append(currency_col_name)
            
            df[col] = numeric_values_for_col

    for col in df.columns:
        if col in potential_numeric_cols_in_df:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            except Exception:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    possible_date_cols = ['order_date', 'date', 'transaction_date', 'timestamp', 'created_time', 'paid_time', 'rts_time', 'shipped_time', 'delivered_time', 'cancelled_time']
    for col_name in possible_date_cols:
        if col_name in df.columns:
            original_series = df[col_name].copy()
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce', dayfirst=True)
            if df[col_name].isnull().all() and not original_series.isnull().all():
                df[col_name] = pd.to_datetime(original_series, errors='coerce')
                if df[col_name].isnull().all() and not original_series.isnull().all():
                     df[col_name] = original_series
            
            df[col_name] = df[col_name].dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')


    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    for col in df.columns:
        if pd_types.is_numeric_dtype(df[col]) and col in potential_numeric_cols_in_df:
            df[col] = df[col].fillna(0)
        if col in new_currency_columns:
            df[col] = df[col].fillna('N/A')

    original_cols_ordered = [col for col in df.columns if col not in new_currency_columns]
    df = df[original_cols_ordered + new_currency_columns]

    return df

# Collect all unique conceptual roles that are required by any KPI
CORE_CONCEPTUAL_ROLES_FOR_AI_MAPPING = sorted(list(set(
    role for kpi_def in KPI_DEFINITIONS.values() for role in kpi_def["required_columns_roles"].keys()
)))

async def map_columns_with_ai(cleaned_headers_list):
    """
    Uses AI to suggest conceptual column mappings.
    The AI maps from actual cleaned headers to conceptual roles.

    Args:
        cleaned_headers_list (list): A list of column headers from the cleaned DataFrame.

    Returns:
        tuple: A tuple containing:
            - dict: AI's suggested column mappings (actual_header_name: conceptual_role or 'Ignore').
            - list: Sorted list of all available conceptual roles for UI dropdowns (plus 'Ignore').
    """
    if not cleaned_headers_list:
        return {}, []

    # AI Prompt: Map FROM actual headers TO conceptual roles
    conceptual_roles_for_prompt_list = sorted(list(CONCEPTUAL_COLUMN_ROLES.keys()))
    conceptual_roles_for_prompt_str = ", ".join(f"'{role}'" for role in conceptual_roles_for_prompt_list)


    prompt = f"""You are an intelligent data mapping assistant for e-commerce data.
    Given a list of actual cleaned CSV headers, map each relevant header to one of the following standard conceptual e-commerce column types.
    Only map headers that clearly correspond to these conceptual types. If a header doesn't fit a conceptual type, you **must** map it to 'Ignore'.
    The output should be a JSON object where keys are the original cleaned CSV headers and values are the mapped conceptual column types or 'Ignore'.
    You **must** provide a mapping for every actual cleaned CSV header provided.

    **Standard Conceptual Column Types:**
    {conceptual_roles_for_prompt_str}

    **Actual Cleaned CSV Headers to map:**
    {json.dumps(cleaned_headers_list)}

    Your output **must be a JSON object** only, formatted exactly as shown in the example. Do not include any other text or markdown fences outside the JSON.

    Example Output:
    ```json
    {{
      "order_id": "Order ID",
      "product_name": "Product Name",
      "quantity": "Quantity",
      "sku_unit_original_price": "Price",
      "order_amount": "Sales Amount",
      "buyer_username": "Customer ID",
      "created_time": "Order Date",
      "product_category": "Category",
      "country": "Country",
      "shipping_fee_after_discount": "Shipping Cost",
      "sku_seller_discount": "Discount Amount",
      "payment_method": "Payment Method",
      "taxes": "Taxes",
      "order_refund_amount": "Order Refund Amount",
      "weight_kg": "Weight",
      "unimportant_column": "Ignore"
    }}
    ```
    Your Mapping:
    """
    
    mapping_generation_config = {
        "responseMimeType": "application/json",
        "responseSchema": {
            "type": "OBJECT",
            "patternProperties": {
                ".*": {"type": "string"}
            },
            "additionalProperties": True
        }
    }

    try:
        # ai_column_mappings_raw will be actual_header: conceptual_role (or Ignore)
        ai_column_mappings_raw = await call_gemini_api(prompt, generation_config=mapping_generation_config)
    except Exception as e:
        print(f"Error during AI call for mapping: {e}")
        ai_column_mappings_raw = {"error": str(e)}

    # Ensure all original headers are present in the final output, defaulting to 'Ignore'
    final_ai_mappings = {}
    if isinstance(ai_column_mappings_raw, dict) and not ai_column_mappings_raw.get("error"):
        for header in cleaned_headers_list:
            mapped_role = ai_column_mappings_raw.get(header, "Ignore")
            # Validate AI's suggested role against our known conceptual roles
            if mapped_role not in CONCEPTUAL_COLUMN_ROLES.keys() and mapped_role != "Ignore":
                mapped_role = "Ignore" # Correct any hallucinated conceptual roles
            final_ai_mappings[header] = mapped_role
    else:
        # Fallback if AI fails: map all cleaned headers to 'Ignore'
        final_ai_mappings = {header: "Ignore" for header in cleaned_headers_list}


    # For the frontend dropdown, provide all conceptual roles + 'Ignore' option
    available_conceptual_roles_for_dropdown = sorted(list(CONCEPTUAL_COLUMN_ROLES.keys()) + ["Ignore"])

    return final_ai_mappings, available_conceptual_roles_for_dropdown


# KPI_DEFINITIONS remains the same as it maps conceptual roles internally
KPI_DEFINITIONS = {
    "Total Revenue": {
        "description": "The total amount of money generated from sales.",
        "equation_template": "SUM({Sales Amount})",
        "required_columns_roles": {"Sales Amount": CONCEPTUAL_COLUMN_ROLES["Sales Amount"]},
        "calculation_func": lambda df, mapped_cols: df[mapped_cols["Sales Amount"]].sum() if "Sales Amount" in mapped_cols and not df[mapped_cols["Sales Amount"]].empty else 0,
        "unit": "$",
        "time_series_agg_role": "Sales Amount"
    },
    "Number of Orders": {
        "description": "The total count of unique orders placed.",
        "equation_template": "COUNT_DISTINCT({Order ID})",
        "required_columns_roles": {"Order ID": CONCEPTUAL_COLUMN_ROLES["Order ID"]},
        "calculation_func": lambda df, mapped_cols: df[mapped_cols["Order ID"]].nunique() if "Order ID" in mapped_cols and not df[mapped_cols["Order ID"]].empty else 0,
        "unit": "orders",
        "time_series_agg_role": "Order ID"
    },
    "Average Order Value (AOV)": {
        "description": "The average amount of money spent per order.",
        "equation_template": "Total Revenue / Number of Orders",
        "required_columns_roles": {
            "Sales Amount": CONCEPTUAL_COLUMN_ROLES["Sales Amount"],
            "Order ID": CONCEPTUAL_COLUMN_ROLES["Order ID"]
        },
        "calculation_func": lambda df, mapped_cols: (df[mapped_cols["Sales Amount"]].sum() / df[mapped_cols["Order ID"]].nunique()) if "Sales Amount" in mapped_cols and "Order ID" in mapped_cols and df[mapped_cols["Order ID"]].nunique() > 0 else 0,
        "unit": "$",
        "time_series_agg_role": "Sales Amount"
    },
    "Units Sold": {
        "description": "The total quantity of items sold across all orders.",
        "equation_template": "SUM({Quantity})",
        "required_columns_roles": {"Quantity": CONCEPTUAL_COLUMN_ROLES["Quantity"]},
        "calculation_func": lambda df, mapped_cols: df[mapped_cols["Quantity"]].sum() if "Quantity" in mapped_cols and not df[mapped_cols["Quantity"]].empty else 0,
        "unit": "units",
        "time_series_agg_role": "Quantity"
    },
    "Number of Unique Customers": {
        "description": "The total count of distinct customers who made a purchase.",
        "equation_template": "COUNT_DISTINCT({Customer ID})",
        "required_columns_roles": {"Customer ID": CONCEPTUAL_COLUMN_ROLES["Customer ID"]},
        "calculation_func": lambda df, mapped_cols: df[mapped_cols["Customer ID"]].nunique() if "Customer ID" in mapped_cols and not df[mapped_cols["Customer ID"]].empty else 0,
        "unit": "customers",
        "time_series_agg_role": "Customer ID"
    },
    "Average Selling Price (ASP)": {
        "description": "The average price at which a single unit of a product is sold.",
        "equation_template": "Total Revenue / Units Sold",
        "required_columns_roles": {
            "Sales Amount": CONCEPTUAL_COLUMN_ROLES["Sales Amount"],
            "Quantity": CONCEPTUAL_COLUMN_ROLES["Quantity"]
        },
        "calculation_func": lambda df, mapped_cols: (df[mapped_cols["Sales Amount"]].sum() / df[mapped_cols["Quantity"]].sum()) if "Sales Amount" in mapped_cols and "Quantity" in mapped_cols and df[mapped_cols["Quantity"]].sum() > 0 else 0,
        "unit": "$",
        "time_series_agg_role": "Sales Amount"
    },
    "Conversion Rate": {
        "description": "The percentage of website visitors who complete a desired goal, such as making a purchase. Requires a column for website visits/sessions.",
        "equation_template": "({Number of Orders} / {Website Visits}) * 100",
        "required_columns_roles": {
            "Order ID": CONCEPTUAL_COLUMN_ROLES["Order ID"],
            "Website Visits": CONCEPTUAL_COLUMN_ROLES["Website Visits"]
        },
        "calculation_func": lambda df, mapped_cols: (df[mapped_cols["Order ID"]].nunique() / df[mapped_cols["Website Visits"]].sum()) * 100 if "Order ID" in mapped_cols and "Website Visits" in mapped_cols and df[mapped_cols["Website Visits"]].sum() > 0 else 0,
        "unit": "percent",
        "time_series_agg_role": "Order ID"
    },
    "Customer Lifetime Value (CLTV)": {
        "description": "The average revenue expected from a single customer account over their lifetime. This calculation uses average revenue per unique customer.",
        "equation_template": "({Total Revenue} / {Number of Unique Customers})",
        "required_columns_roles": {
            "Sales Amount": CONCEPTUAL_COLUMN_ROLES["Sales Amount"],
            "Customer ID": CONCEPTUAL_COLUMN_ROLES["Customer ID"]
        },
        "calculation_func": lambda df, mapped_cols: (df[mapped_cols["Sales Amount"]].sum() / df[mapped_cols["Customer ID"]].nunique()) if "Sales Amount" in mapped_cols and "Customer ID" in mapped_cols and df[mapped_cols["Customer ID"]].nunique() > 0 else 0,
        "unit": "$",
        "time_series_agg_role": "Sales Amount"
    },
    "Repeat Purchase Rate": {
        "description": "The percentage of customers who have made more than one purchase.",
        "equation_template": "(Number of Returning Customers / Total Unique Customers) * 100",
        "required_columns_roles": {
            "Customer ID": CONCEPTUAL_COLUMN_ROLES["Customer ID"],
            "Order ID": CONCEPTUAL_COLUMN_ROLES["Order ID"]
        },
        "calculation_func": lambda df, mapped_cols: (df.groupby(mapped_cols["Customer ID"])[mapped_cols["Order ID"]].nunique() > 1).sum() / df[mapped_cols["Customer ID"]].nunique() * 100 if "Customer ID" in mapped_cols and "Order ID" in mapped_cols and df[mapped_cols["Customer ID"]].nunique() > 0 else 0,
        "unit": "percent",
        "time_series_agg_role": "Customer ID"
    },
    "Average Items Per Order": {
        "description": "The average number of items purchased in a single order.",
        "equation_template": "{Total Units Sold} / {Number of Orders}",
        "required_columns_roles": {
            "Quantity": CONCEPTUAL_COLUMN_ROLES["Quantity"],
            "Order ID": CONCEPTUAL_COLUMN_ROLES["Order ID"]
        },
        "calculation_func": lambda df, mapped_cols: (df[mapped_cols["Quantity"]].sum() / df[mapped_cols["Order ID"]].nunique()) if "Quantity" in mapped_cols and "Order ID" in mapped_cols and df[mapped_cols["Order ID"]].nunique() > 0 else 0,
        "unit": "items",
        "time_series_agg_role": "Quantity"
    },
    # New KPIs for aggregation by location/user/etc.
    "Revenue by Country": {
        "description": "Total revenue aggregated by country.",
        "equation_template": "SUM({Sales Amount}) GROUP BY {Country}",
        "required_columns_roles": {
            "Sales Amount": CONCEPTUAL_COLUMN_ROLES["Sales Amount"],
            "Country": CONCEPTUAL_COLUMN_ROLES["Country"]
        },
        "calculation_func": lambda df, mapped_cols: df.groupby(mapped_cols["Country"])[mapped_cols["Sales Amount"]].sum() if "Country" in mapped_cols and "Sales Amount" in mapped_cols and not df[mapped_cols["Sales Amount"]].empty else pd.Series(dtype='float64'),
        "unit": "$",
        "group_by_role": "Country"
    },
    "Orders by Payment Method": {
        "description": "Number of unique orders aggregated by payment method.",
        "equation_template": "COUNT_DISTINCT({Order ID}) GROUP BY {Payment Method}",
        "required_columns_roles": {
            "Order ID": CONCEPTUAL_COLUMN_ROLES["Order ID"],
            "Payment Method": CONCEPTUAL_COLUMN_ROLES["Payment Method"]
        },
        "calculation_func": lambda df, mapped_cols: df.groupby(mapped_cols["Payment Method"])[mapped_cols["Order ID"]].nunique() if "Payment Method" in mapped_cols and "Order ID" in mapped_cols and not df[mapped_cols["Order ID"]].empty else pd.Series(dtype='int64'),
        "unit": "orders",
        "group_by_role": "Payment Method"
    },
    "Top Products by Revenue": {
        "description": "List of products with their total revenue, sorted by highest revenue.",
        "equation_template": "SUM({Sales Amount}) GROUP BY {Product Name}",
        "required_columns_roles": {
            "Sales Amount": CONCEPTUAL_COLUMN_ROLES["Sales Amount"],
            "Product Name": CONCEPTUAL_COLUMN_ROLES["Product Name"]
        },
        "calculation_func": lambda df, mapped_cols: df.groupby(mapped_cols["Product Name"])[mapped_cols["Sales Amount"]].sum().nlargest(10).reset_index() if "Product Name" in mapped_cols and "Sales Amount" in mapped_cols and not df[mapped_cols["Sales Amount"]].empty else pd.DataFrame(columns=['Product Name', 'Sales Amount']),
        "unit": "$",
        "group_by_role": "Product Name",
        "is_top_n": True
    },
    "Products Sold by Category": {
        "description": "Total units sold for each product category.",
        "equation_template": "SUM({Quantity}) GROUP BY {Category}",
        "required_columns_roles": {
            "Quantity": CONCEPTUAL_COLUMN_ROLES["Quantity"],
            "Category": CONCEPTUAL_COLUMN_ROLES["Category"]
        },
        "calculation_func": lambda df, mapped_cols: df.groupby(mapped_cols["Category"])[mapped_cols["Quantity"]].sum() if "Category" in mapped_cols and "Quantity" in mapped_cols and not df[mapped_cols["Quantity"]].empty else pd.Series(dtype='int64'),
        "unit": "units",
        "group_by_role": "Category"
    }
}


@app.route('/upload_csv', methods=['POST'])
async def upload_csv():
    data = request.json
    csv_data_str = data.get('csv_data', '')

    if not csv_data_str:
        return jsonify({"error": "No CSV data provided."}), 400

    try:
        df_raw = pd.read_csv(io.StringIO(csv_data_str))
        df_cleaned = perform_data_cleaning(df_raw.copy())
        cleaned_csv_data_str = df_cleaned.to_csv(index=False)

        cleaned_headers_list = df_cleaned.columns.tolist()

        # Call the dedicated AI mapping function
        # This returns actual_header -> conceptual_role and list of conceptual roles
        ai_column_mappings, available_conceptual_roles_for_dropdown = await map_columns_with_ai(cleaned_headers_list)

        return jsonify({
            "cleaned_csv_data": cleaned_csv_data_str,
            "ai_column_mappings": ai_column_mappings, # actual_header -> conceptual_role
            "cleaned_headers": cleaned_headers_list, # List of actual cleaned headers
            "available_conceptual_roles": available_conceptual_roles_for_dropdown # List of conceptual roles + Ignore
        })

    except pd.errors.EmptyDataError:
        return jsonify({"error": "Uploaded CSV is empty."}), 400
    except Exception as e:
        print(f"Server error during upload/cleaning/mapping: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during data upload, cleaning, or mapping: {str(e)}"}), 500


@app.route('/get_kpi_suggestions', methods=['POST'])
async def get_kpi_suggestions():
    data = request.json
    cleaned_csv_data_str = data.get('cleaned_csv_data', '')
    user_confirmed_mappings = data.get('user_confirmed_mappings', {}) # actual_header -> conceptual_role

    if not cleaned_csv_data_str:
        return jsonify({"error": "No cleaned CSV data provided."}), 400
    if not user_confirmed_mappings:
        return jsonify({"error": "No column mappings provided."}), 400

    try:
        df = pd.read_csv(io.StringIO(cleaned_csv_data_str))

        suggested_kpis_for_selection = []
        for kpi_name, kpi_def in KPI_DEFINITIONS.items():
            is_kpi_calculable = True
            kpi_specific_mapped_cols = {} # Conceptual role -> actual_header (for calculation function)
            column_mapping_details = [] # Human-readable for frontend

            for required_role_conceptual in kpi_def["required_columns_roles"].keys():
                # Find the actual column name from user_confirmed_mappings
                # We need to reverse lookup: conceptual_role is the target, find the actual header that maps to it
                actual_col_name = None
                for header, role in user_confirmed_mappings.items():
                    if role == required_role_conceptual:
                        actual_col_name = header
                        break
                
                if actual_col_name and actual_col_name != "Ignore" and actual_col_name in df.columns:
                    kpi_specific_mapped_cols[required_role_conceptual] = actual_col_name
                    column_mapping_details.append(f"'{actual_col_name}' (for {required_role_conceptual})")
                else:
                    is_kpi_calculable = False
                    break

            if is_kpi_calculable:
                suggested_kpis_for_selection.append({
                    "kpi_name": kpi_name,
                    "equation": kpi_def["equation_template"],
                    "description": kpi_def["description"],
                    "matched_columns": kpi_specific_mapped_cols, # Conceptual role -> actual_header
                    "column_mapping_details": column_mapping_details
                })
        
        suggested_kpis_for_selection.sort(key=lambda x: x['kpi_name'])

        return jsonify({"kpis": suggested_kpis_for_selection})

    except pd.errors.EmptyDataError:
        return jsonify({"error": "Uploaded CSV is empty."}), 400
    except Exception as e:
        print(f"Server error during KPI suggestion based on mapping: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during KPI suggestion: {str(e)}"}), 500


@app.route('/analyze_kpis', methods=['POST'])
async def analyze_kpis():
    data = request.json
    cleaned_csv_data_str = data.get('cleaned_csv_data', '')
    user_confirmed_mappings = data.get('user_confirmed_mappings', {}) # actual_header -> conceptual_role
    selected_kpis_info = data.get('selected_kpis', [])

    if not cleaned_csv_data_str:
        return jsonify({"error": "No cleaned CSV data provided."}), 400
    if not selected_kpis_info:
        return jsonify({"error": "No KPIs selected for analysis."}), 400
    if not user_confirmed_mappings:
        return jsonify({"error": "No column mappings provided."}), 400

    try:
        df = pd.read_csv(io.StringIO(cleaned_csv_data_str))

        calculated_results = []
        for kpi_info in selected_kpis_info:
            kpi_name = kpi_info['kpi_name']
            # kpi_specific_mapped_cols from selected_kpis_info is already conceptual_role -> actual_header
            kpi_specific_mapped_cols = kpi_info['matched_columns'] 

            kpi_def = KPI_DEFINITIONS.get(kpi_name)
            if not kpi_def:
                continue

            can_calculate_kpi = True
            for required_role_key in kpi_def["required_columns_roles"].keys():
                actual_col_name = kpi_specific_mapped_cols.get(required_role_key)
                if not actual_col_name or actual_col_name == "Ignore" or actual_col_name not in df.columns:
                    can_calculate_kpi = False
                    break

            if not can_calculate_kpi:
                calculated_results.append({
                    "kpi_name": kpi_name,
                    "value": None,
                    "display_value": "N/A",
                    "unit": kpi_def.get('unit', ''),
                    "interpretation_insight": f"Calculation failed: Missing required column mappings for {kpi_name} or columns not found in data. Please check your column mappings.",
                    "visual_type": "text",
                    "is_time_series": False,
                    "chart_data": None,
                    "chart_options": None
                })
                continue


            calculated_value = None
            chart_data = None
            is_time_series = False
            
            date_col_actual_name = kpi_specific_mapped_cols.get("Order Date")
            date_col_is_valid = False
            if date_col_actual_name and date_col_actual_name != "Ignore" and date_col_actual_name in df.columns:
                if pd_types.is_datetime64_any_dtype(df[date_col_actual_name]) and not df[date_col_actual_name].isnull().all():
                    date_col_is_valid = True
            
            calculated_value = kpi_def["calculation_func"](df, kpi_specific_mapped_cols)
            
            if date_col_is_valid and kpi_def.get("time_series_agg_role"):
                target_col_for_ts_role = kpi_def["time_series_agg_role"]
                target_col_for_ts = kpi_specific_mapped_cols.get(target_col_for_ts_role)
                
                if target_col_for_ts and target_col_for_ts != "Ignore" and pd_types.is_numeric_dtype(df[target_col_for_ts]):
                    is_time_series = True
                    df_ts = df.dropna(subset=[target_col_for_ts, date_col_actual_name]).copy()
                    df_ts['month'] = df_ts[date_col_actual_name].dt.to_period('M')

                    if kpi_name in ["Total Revenue", "Units Sold", "Customer Lifetime Value (CLTV)"]:
                        ts_series = df_ts.groupby('month')[target_col_for_ts].sum().sort_index()
                    elif kpi_name in ["Number of Orders", "Number of Unique Customers"]:
                        ts_series = df_ts.groupby('month')[target_col_for_ts].nunique().sort_index()
                    elif kpi_name == "Average Order Value (AOV)":
                        sales_col = kpi_specific_mapped_cols.get("Sales Amount")
                        order_id_col = kpi_specific_mapped_cols.get("Order ID")
                        if sales_col and sales_col != "Ignore" and order_id_col and order_id_col != "Ignore":
                            ts_series = df_ts.groupby('month').apply(lambda x: x[sales_col].sum() / x[order_id_col].nunique() if x[order_id_col].nunique() > 0 else 0).sort_index()
                        else: is_time_series = False
                    elif kpi_name == "Average Selling Price (ASP)":
                        sales_col = kpi_specific_mapped_cols.get("Sales Amount")
                        qty_col = kpi_specific_mapped_cols.get("Quantity")
                        if sales_col and sales_col != "Ignore" and qty_col and qty_col != "Ignore":
                            ts_series = df_ts.groupby('month').apply(lambda x: x[sales_col].sum() / x[qty_col].sum() if x[qty_col].sum() > 0 else 0).sort_index()
                        else: is_time_series = False
                    elif kpi_name == "Conversion Rate":
                        order_id_col = kpi_specific_mapped_cols.get("Order ID")
                        visits_col = kpi_specific_mapped_cols.get("Website Visits")
                        if order_id_col and order_id_col != "Ignore" and visits_col and visits_col != "Ignore":
                             ts_series = df_ts.groupby('month').apply(lambda x: (x[order_id_col].nunique() / x[visits_col].sum()) * 100 if x[visits_col].sum() > 0 else 0).sort_index()
                        else: is_time_series = False
                    elif kpi_name == "Repeat Purchase Rate":
                        customer_id_col = kpi_specific_mapped_cols.get("Customer ID")
                        order_id_col = kpi_specific_mapped_cols.get("Order ID")
                        if customer_id_col and customer_id_col != "Ignore" and order_id_col and order_id_col != "Ignore":
                            ts_series = df_ts.groupby('month').apply(lambda x: (x.groupby(customer_id_col)[order_id_col].nunique() > 1).sum() / x[customer_id_col].nunique() * 100 if x[customer_id_col].nunique() > 0 else 0).sort_index()
                        else: is_time_series = False
                    elif kpi_name == "Average Items Per Order":
                        quantity_col = kpi_specific_mapped_cols.get("Quantity")
                        order_id_col = kpi_specific_mapped_cols.get("Order ID")
                        if quantity_col and quantity_col != "Ignore" and order_id_col and order_id_col != "Ignore":
                            ts_series = df_ts.groupby('month').apply(lambda x: (x[quantity_col].sum() / x[order_id_col].nunique()) if x[order_id_col].nunique() > 0 else 0).sort_index()
                        else: is_time_series = False
                    else:
                        is_time_series = False
                        
                    if is_time_series and not ts_series.empty:
                        chart_data = {
                            "labels": [str(m) for m in ts_series.index],
                            "datasets": [{
                                "label": kpi_name,
                                "data": ts_series.values.tolist(),
                                "backgroundColor": "rgba(224, 108, 45, 0.7)",
                                "borderColor": "rgba(224, 108, 45, 1)",
                                "borderWidth": 2,
                                "fill": False,
                                "tension": 0.4
                            }]
                        }
                        if kpi_def.get('unit') == 'percent':
                            chart_data["datasets"][0]["backgroundColor"] = "rgba(92, 136, 79, 0.7)"
                            chart_data["datasets"][0]["borderColor"] = "rgba(92, 136, 79, 1)"
                else:
                    is_time_series = False
            
            elif kpi_def.get("group_by_role"):
                group_by_col_role = kpi_def["group_by_role"]
                group_by_col = kpi_specific_mapped_cols.get(group_by_col_role)
                
                target_col_for_agg = None
                for role_key in kpi_def["required_columns_roles"].keys():
                    if role_key != group_by_col_role:
                        target_col_for_agg = kpi_specific_mapped_cols.get(role_key)
                        break

                if group_by_col and group_by_col != "Ignore" and \
                   target_col_for_agg and target_col_for_agg != "Ignore" and \
                   pd_types.is_numeric_dtype(df[target_col_for_agg]):
                    
                    grouped_series_or_df = kpi_def["calculation_func"](df, kpi_specific_mapped_cols)
                    if not grouped_series_or_df.empty:
                        if kpi_def.get("is_top_n"):
                             chart_data = {
                                 "labels": grouped_series_or_df[group_by_col].tolist(),
                                 "datasets": [{
                                     "label": kpi_name,
                                     "data": grouped_series_or_df[target_col_for_agg].tolist(),
                                     "backgroundColor": "rgba(224, 108, 45, 0.7)",
                                     "borderColor": "rgba(224, 108, 45, 1)",
                                     "borderWidth": 1,
                                 }]
                             }
                        else:
                             chart_data = {
                                 "labels": grouped_series_or_df.index.tolist(),
                                 "datasets": [{
                                     "label": kpi_name,
                                     "data": grouped_series_or_df.values.tolist(),
                                     "backgroundColor": "rgba(58, 47, 47, 0.7)",
                                     "borderColor": "rgba(58, 47, 47, 1)",
                                     "borderWidth": 1,
                                 }]
                             }
                        visual_type = 'bar'
                    else:
                        chart_data = None
                else:
                    chart_data = None


            display_value = "N/A"
            if calculated_value is not None:
                if isinstance(calculated_value, (int, float)):
                    if kpi_def.get('unit') == '$':
                        display_value = f"${calculated_value:.2f}".strip()
                    elif kpi_def.get('unit') == 'percent':
                        display_value = f"{calculated_value:.2f}%".strip()
                    else:
                        display_value = f"{calculated_value:.0f} {kpi_def.get('unit', '')}".strip()
                elif isinstance(calculated_value, pd.Series) and not calculated_value.empty:
                    display_value = f"Grouped Data: {len(calculated_value)} categories"
                elif isinstance(calculated_value, pd.DataFrame) and not calculated_value.empty:
                     display_value = f"Top {len(calculated_value)} entries."
                
            prompt = f"""
            The user selected the KPI: "{kpi_name}" with a calculated value of {calculated_value} {kpi_def.get('unit', '')}.
            The original CSV headers were: {list(df.columns)}.
            {"The data contains a time series (date column detected on column '{date_col_actual_name}')." if date_col_actual_name else "The data is not time-series based."}
            {"This KPI is grouped by '{group_by_col_role}'." if kpi_def.get('group_by_role') else ""}

            Provide a concise interpretation of this KPI's value, explaining what it means, what insights can be drawn, and potential next steps or business implications.
            Also, suggest the most appropriate chart type (e.g., 'bar', 'line', 'pie', 'doughnut', 'polarArea') for this KPI given its value and whether it's time-series, or grouped data. If it's time-series, a 'line' chart is usually best. For grouped data, 'bar' or 'pie' charts are often good.
            Ensure the interpretation is actionable and clear.

            Provide the output in a JSON object format with 'interpretation_insight' and 'visual_type' keys.
            Example JSON:
            {{
              "interpretation_insight": "A Total Revenue of $150,000 indicates strong sales. Consider segmenting by product category or region for deeper insights.",
              "visual_type": "line"
            }}
            """
            ai_response = await call_gemini_api(prompt, generation_config={
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "interpretation_insight": {"type": "STRING"},
                        "visual_type": {"type": "STRING"}
                    },
                    "required": ["interpretation_insight", "visual_type"]
                }
            })

            if isinstance(ai_response, dict) and ai_response.get("error"):
                interpretation = "Could not get AI insight."
                visual_type_ai_suggested = "bar"
            else:
                interpretation = ai_response.get('interpretation_insight', 'No specific AI insight available.')
                visual_type_ai_suggested = ai_response.get('visual_type', 'bar')
            
            if is_time_series:
                final_visual_type = 'line'
            elif kpi_def.get('group_by_role'):
                final_visual_type = visual_type_ai_suggested
                if not final_visual_type:
                    final_visual_type = 'bar'
            else:
                final_visual_type = visual_type_ai_suggested
                if final_visual_type not in ['bar', 'line', 'pie', 'doughnut', 'polarArea', 'radar']:
                    final_visual_type = 'bar'


            calculated_results.append({
                "kpi_name": kpi_name,
                "value": calculated_value,
                "display_value": display_value,
                "unit": kpi_def.get('unit', ''),
                "interpretation_insight": interpretation,
                "visual_type": final_visual_type,
                "is_time_series": is_time_series,
                "chart_data": chart_data,
                "chart_options": {
                    "responsive": True,
                    "maintainAspectRatio": False,
                    "plugins": {
                        "legend": {
                            "position": "top",
                        },
                        "title": {
                            "display": True,
                            "text": kpi_name
                        }
                    },
                    "scales": {
                        "y": {
                            "beginAtZero": True,
                            "ticks": {
                                "callback": "function(value) {return value + '%';}" if kpi_def.get('unit') == 'percent' else None
                            },
                            "max": 100 if kpi_def.get('unit') == 'percent' else None
                        }
                    }
                }
            })

        return jsonify({"calculated_kpis": calculated_results})

    except pd.errors.EmptyDataError:
        return jsonify({"error": "Uploaded CSV is empty."}), 400
    except Exception as e:
        print(f"Server error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"An error occurred during analysis: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
