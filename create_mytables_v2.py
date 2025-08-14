# create_mytables_v2.py
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.types import String, Integer, DateTime, DECIMAL, Text

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR            # CSVs live in the same folder as this script
DB_USER = "txt2sql_user"
DB_PASS = "StrongPassword!123"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "txt2sql_v2"

def p(name: str) -> str:
    """Absolute path to a CSV under DATA_DIR; raises if missing."""
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return str(path)

print(f"📁 Using DATA_DIR: {DATA_DIR}")

# ---------- CREATE DB IF NEEDED ----------
server_engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}")
with server_engine.begin() as conn:
    conn.execute(text(
        f"CREATE DATABASE IF NOT EXISTS {DB_NAME} "
        "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
    ))
print(f"✅ Ensured database exists: {DB_NAME}")

# ---------- CONNECT TO TARGET DB ----------
engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ---- ORDERS ----
orders = pd.read_csv(
    p('olist_orders_dataset.csv'),
    parse_dates=[
        'order_purchase_timestamp','order_approved_at',
        'order_delivered_carrier_date','order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]
)
orders.to_sql(
    'orders', engine, if_exists='replace', index=False,
    dtype={
        'order_id': String(64),
        'customer_id': String(64),
        'order_status': String(32),
        'order_purchase_timestamp': DateTime(),
        'order_approved_at': DateTime(),
        'order_delivered_carrier_date': DateTime(),
        'order_delivered_customer_date': DateTime(),
        'order_estimated_delivery_date': DateTime(),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: orders")

# ---- ORDER_PAYMENTS ----
op = pd.read_csv(p('olist_order_payments_dataset.csv'))
op.to_sql(
    'order_payments', engine, if_exists='replace', index=False,
    dtype={
        'order_id': String(64),
        'payment_sequential': Integer(),
        'payment_type': String(32),
        'payment_installments': Integer(),
        'payment_value': DECIMAL(12, 2),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: order_payments")

# ---- ORDER_ITEMS ----
oi = pd.read_csv(p('olist_order_items_dataset.csv'), parse_dates=['shipping_limit_date'])
oi.to_sql(
    'order_items', engine, if_exists='replace', index=False,
    dtype={
        'order_id': String(64),
        'order_item_id': Integer(),
        'product_id': String(64),
        'seller_id': String(64),
        'shipping_limit_date': DateTime(),
        'price': DECIMAL(12, 2),
        'freight_value': DECIMAL(12, 2),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: order_items")

# ---- ORDER_REVIEWS ----
orv = pd.read_csv(
    p('olist_order_reviews_dataset.csv'),
    parse_dates=['review_creation_date','review_answer_timestamp']
)
orv.to_sql(
    'order_reviews', engine, if_exists='replace', index=False,
    dtype={
        'review_id': String(64),
        'order_id': String(64),
        'review_score': Integer(),
        'review_comment_title': String(255),
        # Use Text() for long messages
        'review_comment_message': Text(),
        'review_creation_date': DateTime(),
        'review_answer_timestamp': DateTime(),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: order_reviews")

# ---- CUSTOMER ----
cust = pd.read_csv(p('olist_customers_dataset.csv'))
cust.to_sql(
    'customer', engine, if_exists='replace', index=False,
    dtype={
        'customer_id': String(64),
        'customer_unique_id': String(64),
        'customer_zip_code_prefix': Integer(),
        'customer_city': String(128),
        'customer_state': String(4),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: customer")

# ---- PRODUCTS ----
prod = pd.read_csv(p('olist_products_dataset.csv'))
for col in [
    'product_name_lenght','product_description_lenght','product_photos_qty',
    'product_weight_g','product_length_cm','product_height_cm','product_width_cm'
]:
    prod[col] = pd.to_numeric(prod[col], errors='coerce').astype('Int64')

prod.to_sql(
    'products', engine, if_exists='replace', index=False,
    dtype={
        'product_id': String(64),
        'product_category_name': String(128),
        'product_name_lenght': Integer(),
        'product_description_lenght': Integer(),
        'product_photos_qty': Integer(),
        'product_weight_g': Integer(),
        'product_length_cm': Integer(),
        'product_height_cm': Integer(),
        'product_width_cm': Integer(),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: products")

# ---- SELLERS ----
sel = pd.read_csv(p('olist_sellers_dataset.csv'))
sel.to_sql(
    'sellers', engine, if_exists='replace', index=False,
    dtype={
        'seller_id': String(64),
        'seller_zip_code_prefix': Integer(),
        'seller_city': String(128),
        'seller_state': String(4),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: sellers")

# ---- CATEGORY_TRANSLATION ----
ct = pd.read_csv(p('product_category_name_translation.csv'))
ct.to_sql(
    'category_translation', engine, if_exists='replace', index=False,
    dtype={
        'product_category_name': String(128),
        'product_category_name_english': String(128),
    },
    chunksize=50, method='multi'
)
print("✅ Loaded: category_translation")

print(f"🎉 All tables loaded into {DB_NAME} with correct types.")
