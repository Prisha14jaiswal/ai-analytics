"""
================================================================================
E-Commerce Customer Repeat Analysis
================================================================================
Purpose: Analyze repeat customers vs new customers
Author: Data Analyst
Date: 2026-01-20

Key: Using customer_unique_id (not customer_id) for true customer identity
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Output writer
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

output = OutputWriter('customer_repeat_analysis_results.txt')
sys.stdout = output

print("=" * 70)
print("CUSTOMER REPEAT ANALYSIS")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD DATA & UNDERSTAND THE IDs
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: UNDERSTANDING customer_id vs customer_unique_id")
print("=" * 70)

df = pd.read_csv('merged_ecommerce_with_funnel.csv')
print(f"\nLoaded dataset: {len(df):,} rows")

# Get order-level data
orders_df = df.drop_duplicates('order_id')[['order_id', 'customer_id', 'customer_unique_id', 'customer_state']]

# Merge with order revenue
order_revenue = df.groupby('order_id').agg({
    'price': 'sum',
    'freight_value': 'sum'
}).reset_index()
order_revenue['total_value'] = order_revenue['price'] + order_revenue['freight_value']

orders_df = orders_df.merge(order_revenue, on='order_id')

# Count unique IDs
unique_customer_id = orders_df['customer_id'].nunique()
unique_customer_unique_id = orders_df['customer_unique_id'].nunique()
total_orders = len(orders_df)

print(f"""
ID COMPARISON:
--------------
  Total Orders:           {total_orders:,}
  Unique customer_id:     {unique_customer_id:,}
  Unique customer_unique_id: {unique_customer_unique_id:,}
  
OBSERVATION:
  - customer_id count = order count (1:1 relationship)
  - customer_unique_id count < order count (some customers ordered multiple times)
""")

# =============================================================================
# STEP 2: WHY customer_unique_id vs customer_id
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: WHY USE customer_unique_id INSTEAD OF customer_id?")
print("=" * 70)

print("""
EXPLANATION:
============

customer_id:
  - Generated for EACH ORDER
  - Even the same person gets a NEW customer_id per order
  - Cannot track customer behavior across orders
  - Essentially an "order-customer" identifier

customer_unique_id:
  - Identifies the ACTUAL PERSON
  - Same person = same customer_unique_id across all their orders
  - Enables tracking of repeat purchases
  - Essential for customer lifetime value analysis

ANALOGY:
  - customer_id = Your receipt number (different each visit)
  - customer_unique_id = Your loyalty card number (same every visit)

BUSINESS IMPORTANCE:
  - To calculate repeat rate, we MUST use customer_unique_id
  - Using customer_id would show 0% repeat customers (all unique)
  - This is a common data modeling pattern in e-commerce
""")

# Example: Show a repeat customer
repeat_example = orders_df.groupby('customer_unique_id').size()
repeat_customer = repeat_example[repeat_example > 1].index[0]
example_orders = orders_df[orders_df['customer_unique_id'] == repeat_customer]

print(f"\nEXAMPLE - Repeat Customer:")
print(f"  customer_unique_id: {repeat_customer}")
print(f"  Number of orders: {len(example_orders)}")
print(f"  Different customer_id values: {example_orders['customer_id'].nunique()}")
print("  (Same person, different customer_id per order)")

# =============================================================================
# STEP 3: IDENTIFY REPEAT CUSTOMERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: IDENTIFY REPEAT CUSTOMERS")
print("=" * 70)

# Count orders per unique customer
customer_orders = orders_df.groupby('customer_unique_id').agg({
    'order_id': 'count',
    'total_value': 'sum',
    'customer_state': 'first'
}).reset_index()

customer_orders.columns = ['customer_unique_id', 'order_count', 'total_revenue', 'state']

# Classify customers
customer_orders['customer_type'] = customer_orders['order_count'].apply(
    lambda x: 'New (1 order)' if x == 1 else 'Repeat (2+ orders)'
)

# Separate new vs repeat
new_customers = customer_orders[customer_orders['order_count'] == 1]
repeat_customers = customer_orders[customer_orders['order_count'] > 1]

total_unique_customers = len(customer_orders)
new_count = len(new_customers)
repeat_count = len(repeat_customers)
repeat_percentage = repeat_count / total_unique_customers * 100

print(f"""
CUSTOMER BREAKDOWN:
-------------------
  Total Unique Customers: {total_unique_customers:,}
  
  New Customers (1 order):     {new_count:,} ({new_count/total_unique_customers*100:.2f}%)
  Repeat Customers (2+ orders): {repeat_count:,} ({repeat_percentage:.2f}%)
""")

# =============================================================================
# STEP 4: AVERAGE ORDERS PER CUSTOMER
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: AVERAGE ORDERS PER CUSTOMER")
print("=" * 70)

avg_orders_all = customer_orders['order_count'].mean()
avg_orders_repeat = repeat_customers['order_count'].mean()
max_orders = customer_orders['order_count'].max()

print(f"""
ORDER FREQUENCY METRICS:
------------------------
  Average orders per customer (all): {avg_orders_all:.2f}
  Average orders per repeat customer: {avg_orders_repeat:.2f}
  Maximum orders by single customer: {max_orders}
  
ORDER COUNT DISTRIBUTION:
""")

order_count_dist = customer_orders['order_count'].value_counts().sort_index()
for orders, count in order_count_dist.head(10).items():
    pct = count / total_unique_customers * 100
    bar = '█' * int(pct / 2)
    print(f"  {orders:>2} order(s): {count:>6,} customers ({pct:>5.2f}%) {bar}")

if max_orders > 10:
    high_frequency = customer_orders[customer_orders['order_count'] > 10]
    print(f"\n  11+ orders: {len(high_frequency):>6,} customers")

# =============================================================================
# STEP 5: REVENUE CONTRIBUTION COMPARISON
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: REVENUE CONTRIBUTION: NEW vs REPEAT CUSTOMERS")
print("=" * 70)

# Revenue metrics
new_revenue = new_customers['total_revenue'].sum()
repeat_revenue = repeat_customers['total_revenue'].sum()
total_revenue = new_revenue + repeat_revenue

new_avg_value = new_customers['total_revenue'].mean()
repeat_avg_value = repeat_customers['total_revenue'].mean()
repeat_avg_per_order = repeat_customers['total_revenue'].sum() / repeat_customers['order_count'].sum()

print(f"""
REVENUE BREAKDOWN:
------------------
  Total Revenue: ${total_revenue:,.2f}
  
  New Customers:
    - Revenue: ${new_revenue:,.2f} ({new_revenue/total_revenue*100:.1f}%)
    - Customers: {new_count:,}
    - Avg Lifetime Value: ${new_avg_value:,.2f}
  
  Repeat Customers:
    - Revenue: ${repeat_revenue:,.2f} ({repeat_revenue/total_revenue*100:.1f}%)
    - Customers: {repeat_count:,}
    - Avg Lifetime Value: ${repeat_avg_value:,.2f}
    - Avg per Order: ${repeat_avg_per_order:,.2f}
""")

# Value per customer comparison
value_multiplier = repeat_avg_value / new_avg_value

print(f"""
KEY INSIGHT:
------------
  Repeat customers are worth {value_multiplier:.1f}x more than new customers!
  
  Even though repeat customers are only {repeat_percentage:.1f}% of the customer base,
  they contribute {repeat_revenue/total_revenue*100:.1f}% of total revenue.
""")

# =============================================================================
# STEP 6: DETAILED REPEAT CUSTOMER ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: REPEAT CUSTOMER SEGMENTS")
print("=" * 70)

# Segment repeat customers
repeat_customers['segment'] = pd.cut(
    repeat_customers['order_count'],
    bins=[1, 2, 3, 5, 100],
    labels=['2 orders', '3 orders', '4-5 orders', '6+ orders']
)

segment_summary = repeat_customers.groupby('segment').agg({
    'customer_unique_id': 'count',
    'total_revenue': ['sum', 'mean'],
    'order_count': 'sum'
}).reset_index()

segment_summary.columns = ['segment', 'customers', 'revenue', 'avg_ltv', 'orders']

print("\nREPEAT CUSTOMER SEGMENTS:")
print("-" * 80)
print(f"{'Segment':<15} {'Customers':>12} {'Revenue':>15} {'Avg LTV':>12} {'Orders':>10}")
print("-" * 80)

for _, row in segment_summary.iterrows():
    print(f"{row['segment']:<15} {row['customers']:>12,} ${row['revenue']:>14,.0f} ${row['avg_ltv']:>11,.2f} {row['orders']:>10,}")

# =============================================================================
# STEP 7: CREATE VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: CREATE VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Customer Repeat Analysis', fontsize=16, fontweight='bold')

# --- Chart 1: Customer Type Distribution ---
ax1 = axes[0, 0]
types = ['New Customers\n(1 order)', 'Repeat Customers\n(2+ orders)']
counts = [new_count, repeat_count]
colors = ['#3498db', '#27ae60']
explode = (0, 0.05)

wedges, texts, autotexts = ax1.pie(counts, explode=explode, labels=types, colors=colors,
                                    autopct='%1.1f%%', startangle=90, 
                                    textprops={'fontsize': 11})
ax1.set_title('Customer Type Distribution', fontsize=13, fontweight='bold')

# --- Chart 2: Revenue Contribution ---
ax2 = axes[0, 1]
revenue_types = ['New Customers', 'Repeat Customers']
revenues = [new_revenue / 1000000, repeat_revenue / 1000000]
bars = ax2.bar(revenue_types, revenues, color=['#3498db', '#27ae60'], edgecolor='white', linewidth=2)

for bar, rev, pct in zip(bars, revenues, [new_revenue/total_revenue*100, repeat_revenue/total_revenue*100]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'${rev:.2f}M\n({pct:.1f}%)', ha='center', fontsize=11, fontweight='bold')

ax2.set_ylabel('Revenue (Millions $)', fontsize=11)
ax2.set_title('Revenue Contribution by Customer Type', fontsize=13, fontweight='bold')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Chart 3: Order Count Distribution ---
ax3 = axes[1, 0]
order_dist = customer_orders['order_count'].value_counts().sort_index()
order_dist_capped = order_dist.head(10)  # Cap at 10 for visualization

ax3.bar(order_dist_capped.index.astype(str), order_dist_capped.values, color='#9b59b6', edgecolor='white')
ax3.set_xlabel('Number of Orders', fontsize=11)
ax3.set_ylabel('Number of Customers', fontsize=11)
ax3.set_title('Distribution of Orders per Customer', fontsize=13, fontweight='bold')
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add percentage labels
for i, (orders, count) in enumerate(order_dist_capped.items()):
    pct = count / total_unique_customers * 100
    ax3.text(i, count + 500, f'{pct:.1f}%', ha='center', fontsize=9)

# --- Chart 4: LTV Comparison ---
ax4 = axes[1, 1]
categories = ['New Customer\nAvg Value', 'Repeat Customer\nAvg LTV']
values = [new_avg_value, repeat_avg_value]
bars4 = ax4.bar(categories, values, color=['#3498db', '#27ae60'], edgecolor='white', linewidth=2)

for bar in bars4:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             f'${bar.get_height():.2f}', ha='center', fontsize=12, fontweight='bold')

ax4.set_ylabel('Average Value ($)', fontsize=11)
ax4.set_title('Average Customer Value: New vs Repeat', fontsize=13, fontweight='bold')
ax4.axhline(y=new_avg_value, color='gray', linestyle='--', alpha=0.3)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add multiplier annotation
ax4.annotate(f'{value_multiplier:.1f}x higher!', 
             xy=(1, repeat_avg_value), 
             xytext=(1.3, repeat_avg_value - 50),
             fontsize=12, fontweight='bold', color='#27ae60',
             arrowprops=dict(arrowstyle='->', color='#27ae60'))

plt.tight_layout()
plt.savefig('customer_repeat_analysis_charts.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\nVisualization saved: customer_repeat_analysis_charts.png")

# =============================================================================
# STEP 8: EXECUTIVE SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 8: EXECUTIVE SUMMARY")
print("=" * 70)

print(f"""
CUSTOMER REPEAT ANALYSIS SUMMARY
================================

WHY customer_unique_id?
  - customer_id changes per order (can't track repeat behavior)
  - customer_unique_id identifies the actual person across orders

KEY METRICS:
  Total Unique Customers: {total_unique_customers:,}
  Repeat Customer Rate:   {repeat_percentage:.2f}%
  Avg Orders per Customer: {avg_orders_all:.2f}

REVENUE COMPARISON:
  +-------------------+----------+-----------+-------------+
  | Customer Type     | Count    | Revenue   | Avg Value   |
  +-------------------+----------+-----------+-------------+
  | New (1 order)     | {new_count:>8,} | ${new_revenue/1000000:>7.2f}M | ${new_avg_value:>10.2f} |
  | Repeat (2+ orders)| {repeat_count:>8,} | ${repeat_revenue/1000000:>7.2f}M | ${repeat_avg_value:>10.2f} |
  +-------------------+----------+-----------+-------------+

BUSINESS INSIGHTS:
  1. {repeat_percentage:.1f}% repeat rate is LOW for e-commerce
     → Opportunity to improve customer retention
  
  2. Repeat customers are worth {value_multiplier:.1f}x more
     → Focus on converting new to repeat customers
  
  3. Most repeat customers only order 2 times
     → Implement loyalty programs to increase frequency

RECOMMENDATIONS:
  - Invest in retention marketing (email, loyalty programs)
  - Analyze why customers don't return
  - Set target to increase repeat rate to 5%+
  - Every 1% increase in repeat rate = significant revenue growth
""")

print("\n" + "=" * 70)
print("CUSTOMER REPEAT ANALYSIS COMPLETE")
print("=" * 70)

# Close output
output.close()
sys.stdout = output.terminal

print("\nResults saved to: customer_repeat_analysis_results.txt")
print("Visualization saved to: customer_repeat_analysis_charts.png")
