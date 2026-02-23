"""
================================================================================
E-Commerce Data Joining & Validation
================================================================================
Purpose: Join Orders, Customers, and Order Items datasets
Author: Data Analyst
Date: 2026-01-20

Join Strategy:
1. Orders LEFT JOIN Customers ON customer_id
2. Result LEFT JOIN Order_Items ON order_id
================================================================================
"""

import pandas as pd
import numpy as np
import sys

# Output to both console and file
class OutputWriter:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.file.flush()
    
    def close(self):
        self.file.close()

output = OutputWriter('joining_results.txt')
sys.stdout = output

print("=" * 70)
print("E-COMMERCE DATA JOINING REPORT")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD DATASETS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOADING DATASETS")
print("=" * 70)

orders = pd.read_csv('olist_orders_dataset.csv')
order_items = pd.read_csv('olist_order_items_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')

print(f"\nOrders:      {len(orders):,} rows")
print(f"Order Items: {len(order_items):,} rows")
print(f"Customers:   {len(customers):,} rows")

# Store original counts for validation
original_orders_count = len(orders)
original_unique_order_ids = orders['order_id'].nunique()

print(f"\nOriginal unique order_id in Orders: {original_unique_order_ids:,}")

# =============================================================================
# STEP 2: JOIN ORDERS WITH CUSTOMERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: JOIN ORDERS WITH CUSTOMERS (LEFT JOIN on customer_id)")
print("=" * 70)

# Check key relationship before joining
orders_customer_ids = set(orders['customer_id'].unique())
customers_customer_ids = set(customers['customer_id'].unique())

matching_ids = orders_customer_ids.intersection(customers_customer_ids)
orders_only_ids = orders_customer_ids - customers_customer_ids
customers_only_ids = customers_customer_ids - orders_customer_ids

print(f"\nKey Analysis (customer_id):")
print(f"  - Customer IDs in Orders: {len(orders_customer_ids):,}")
print(f"  - Customer IDs in Customers: {len(customers_customer_ids):,}")
print(f"  - Matching IDs: {len(matching_ids):,}")
print(f"  - IDs only in Orders: {len(orders_only_ids):,}")
print(f"  - IDs only in Customers: {len(customers_only_ids):,}")

# Perform LEFT JOIN
orders_customers = orders.merge(
    customers,
    on='customer_id',
    how='left',
    validate='one_to_one'  # Expecting 1:1 relationship
)

print(f"\nAfter JOIN:")
print(f"  - Rows before: {len(orders):,}")
print(f"  - Rows after:  {len(orders_customers):,}")
print(f"  - Row change:  {len(orders_customers) - len(orders):,}")

if len(orders_customers) == len(orders):
    print("  - STATUS: NO ROW EXPLOSION - Perfect 1:1 join!")
else:
    print("  - STATUS: WARNING - Row count changed!")

# Check for nulls introduced by left join
customer_nulls = orders_customers['customer_unique_id'].isnull().sum()
print(f"  - Unmatched customers (nulls): {customer_nulls:,}")

# =============================================================================
# STEP 3: JOIN WITH ORDER ITEMS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: JOIN WITH ORDER ITEMS (LEFT JOIN on order_id)")
print("=" * 70)

# Check key relationship before joining
orders_order_ids = set(orders_customers['order_id'].unique())
items_order_ids = set(order_items['order_id'].unique())

matching_order_ids = orders_order_ids.intersection(items_order_ids)
orders_only_order_ids = orders_order_ids - items_order_ids
items_only_order_ids = items_order_ids - orders_order_ids

print(f"\nKey Analysis (order_id):")
print(f"  - Order IDs in Orders: {len(orders_order_ids):,}")
print(f"  - Order IDs in Order Items: {len(items_order_ids):,}")
print(f"  - Matching IDs: {len(matching_order_ids):,}")
print(f"  - IDs only in Orders: {len(orders_only_order_ids):,}")
print(f"  - IDs only in Order Items: {len(items_only_order_ids):,}")

# Calculate expected rows
items_per_order = order_items.groupby('order_id').size()
print(f"\nItems per order distribution:")
print(f"  - Min:  {items_per_order.min()}")
print(f"  - Max:  {items_per_order.max()}")
print(f"  - Mean: {items_per_order.mean():.2f}")
print(f"  - Most orders have: {items_per_order.mode().values[0]} item(s)")

# Perform LEFT JOIN
# Note: This is a one-to-many join (one order can have multiple items)
full_dataset = orders_customers.merge(
    order_items,
    on='order_id',
    how='left'
)

print(f"\nAfter JOIN:")
print(f"  - Rows before: {len(orders_customers):,}")
print(f"  - Rows after:  {len(full_dataset):,}")
print(f"  - Row change:  {len(full_dataset) - len(orders_customers):+,}")

# This expansion is EXPECTED because one order can have multiple items
print(f"\nEXPECTED BEHAVIOR: Row count increased because:")
print(f"  - One order can contain multiple products")
print(f"  - Each order row is duplicated for each item in that order")

# =============================================================================
# STEP 4: VALIDATE NO UNINTENDED ROW EXPLOSION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: VALIDATE JOIN INTEGRITY")
print("=" * 70)

# Check unique order_id count
final_unique_order_ids = full_dataset['order_id'].nunique()

print(f"\nOrder ID Comparison:")
print(f"  - Unique order_id BEFORE joins: {original_unique_order_ids:,}")
print(f"  - Unique order_id AFTER joins:  {final_unique_order_ids:,}")

if original_unique_order_ids == final_unique_order_ids:
    print("  - STATUS: VALID - No orders lost or duplicated incorrectly!")
else:
    diff = final_unique_order_ids - original_unique_order_ids
    print(f"  - STATUS: DIFFERENCE of {diff:,} order IDs")

# Validate row count matches order_items (since we did left join from orders)
print(f"\nRow Count Validation:")
print(f"  - Final dataset rows: {len(full_dataset):,}")
print(f"  - Order items rows:   {len(order_items):,}")

# Check for orders that have items
orders_with_items = full_dataset[full_dataset['product_id'].notna()]
orders_without_items = full_dataset[full_dataset['product_id'].isna()]

print(f"\nOrder-Item Coverage:")
print(f"  - Rows with item data: {len(orders_with_items):,}")
print(f"  - Rows without item data: {len(orders_without_items):,}")

if len(orders_without_items) > 0:
    # Find which orders don't have items
    missing_item_orders = orders_without_items['order_id'].unique()
    print(f"  - Orders missing item data: {len(missing_item_orders):,}")

# =============================================================================
# STEP 5: FINAL DATASET STRUCTURE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: FINAL DATASET STRUCTURE")
print("=" * 70)

print(f"\nFinal Dataset Shape: {full_dataset.shape[0]:,} rows x {full_dataset.shape[1]} columns")

print("\nColumn List:")
for i, col in enumerate(full_dataset.columns, 1):
    dtype = full_dataset[col].dtype
    nulls = full_dataset[col].isnull().sum()
    print(f"  {i:2}. {col:<35} | Type: {str(dtype):<10} | Nulls: {nulls:,}")

# =============================================================================
# STEP 6: EXPLAIN WHY LEFT JOINS ARE USED
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: WHY LEFT JOINS?")
print("=" * 70)

print("""
LEFT JOIN RATIONALE (Business Perspective):
-------------------------------------------

1. ORDERS LEFT JOIN CUSTOMERS
   - We want to keep ALL orders, even if customer data is missing
   - Business reason: An order without customer details is still a sale
   - Alternative (INNER JOIN) would lose orders with missing customer records

2. RESULT LEFT JOIN ORDER_ITEMS  
   - We want to keep ALL orders, even those without item details
   - Business reason: Some orders might be cancelled before items recorded
   - These edge cases are valuable for understanding order funnel

3. WHY NOT INNER JOIN?
   - INNER JOIN only keeps rows that match in BOTH tables
   - We would lose valuable data about incomplete orders
   - For analytics, we often need to analyze "what went wrong" cases

4. WHY NOT RIGHT JOIN?
   - We want orders to be the "anchor" table
   - All analysis should be from the order perspective
   - Order Items without orders would be orphan records (data quality issue)

SUMMARY: Left joins preserve our complete order history while enriching
         it with customer and item details where available.
""")

# =============================================================================
# STEP 7: SAVE CLEAN DATASET
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: SAVE CLEAN DATASET")
print("=" * 70)

# Save the merged dataset
full_dataset.to_csv('merged_ecommerce_data.csv', index=False)
print(f"\nSaved merged dataset to: merged_ecommerce_data.csv")
print(f"Final size: {len(full_dataset):,} rows x {full_dataset.shape[1]} columns")

# Also create a summary of the dataset
print("\nQuick Stats:")
print(f"  - Total Orders: {full_dataset['order_id'].nunique():,}")
print(f"  - Total Customers: {full_dataset['customer_unique_id'].nunique():,}")
print(f"  - Total Products: {full_dataset['product_id'].nunique():,}")
print(f"  - Total Sellers: {full_dataset['seller_id'].nunique():,}")

# Revenue preview
if 'price' in full_dataset.columns:
    total_revenue = full_dataset['price'].sum()
    total_freight = full_dataset['freight_value'].sum()
    print(f"  - Total Product Revenue: ${total_revenue:,.2f}")
    print(f"  - Total Freight Revenue: ${total_freight:,.2f}")

print("\n" + "=" * 70)
print("DATA JOINING COMPLETE")
print("=" * 70)

# Close output
output.close()
sys.stdout = output.terminal

print("\nResults also saved to: joining_results.txt")
print("Merged dataset saved to: merged_ecommerce_data.csv")
