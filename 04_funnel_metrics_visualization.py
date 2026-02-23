"""
================================================================================
E-Commerce Funnel Metrics & Visualization
================================================================================
Purpose: Calculate conversion rates, drop-off rates, and visualize funnel
Author: Data Analyst
Date: 2026-01-20
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# Output to file
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

output = OutputWriter('funnel_metrics_results.txt')
sys.stdout = output

print("=" * 70)
print("FUNNEL METRICS & VISUALIZATION")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOAD DATA")
print("=" * 70)

df = pd.read_csv('merged_ecommerce_with_funnel.csv')
print(f"\nLoaded dataset: {len(df):,} rows")

# =============================================================================
# STEP 2: CALCULATE FUNNEL METRICS USING COUNT(DISTINCT order_id)
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: FUNNEL METRICS (COUNT DISTINCT order_id)")
print("=" * 70)

# Get unique orders at each stage
stage_1_orders = df[df['funnel_order_placed'] == 1]['order_id'].nunique()
stage_2_orders = df[df['funnel_payment_approved'] == 1]['order_id'].nunique()
stage_3_orders = df[df['funnel_shipped'] == 1]['order_id'].nunique()
stage_4_orders = df[df['funnel_delivered'] == 1]['order_id'].nunique()

# Store in a structured format
funnel_data = {
    'Stage': ['1. Order Placed', '2. Payment Approved', '3. Shipped', '4. Delivered'],
    'Stage_Short': ['Placed', 'Approved', 'Shipped', 'Delivered'],
    'Orders': [stage_1_orders, stage_2_orders, stage_3_orders, stage_4_orders]
}

funnel_df = pd.DataFrame(funnel_data)

# Calculate metrics
funnel_df['Pct_of_Stage_1'] = (funnel_df['Orders'] / stage_1_orders * 100).round(2)

# Conversion rate (from previous stage)
conversion_rates = [100.0]  # First stage is 100%
for i in range(1, len(funnel_df)):
    prev_orders = funnel_df.loc[i-1, 'Orders']
    curr_orders = funnel_df.loc[i, 'Orders']
    conv_rate = round(curr_orders / prev_orders * 100, 2)
    conversion_rates.append(conv_rate)
funnel_df['Conversion_Rate'] = conversion_rates

# Drop-off count and percentage
dropoff_counts = [0]  # First stage has no drop-off
for i in range(1, len(funnel_df)):
    prev_orders = funnel_df.loc[i-1, 'Orders']
    curr_orders = funnel_df.loc[i, 'Orders']
    dropoff = prev_orders - curr_orders
    dropoff_counts.append(dropoff)
funnel_df['Dropoff_Count'] = dropoff_counts

dropoff_pcts = [0.0]
for i in range(1, len(funnel_df)):
    prev_orders = funnel_df.loc[i-1, 'Orders']
    dropoff = funnel_df['Dropoff_Count'].iloc[i]
    dropoff_pct = round(dropoff / prev_orders * 100, 2)
    dropoff_pcts.append(dropoff_pct)
funnel_df['Dropoff_Pct'] = dropoff_pcts

print("\nFUNNEL METRICS TABLE:")
print("-" * 90)
print(f"{'Stage':<25} {'Orders':>12} {'% of Start':>12} {'Conv Rate':>12} {'Drop-off':>12} {'Drop %':>12}")
print("-" * 90)

for _, row in funnel_df.iterrows():
    print(f"{row['Stage']:<25} {row['Orders']:>12,} {row['Pct_of_Stage_1']:>11.2f}% {row['Conversion_Rate']:>11.2f}% {int(row['Dropoff_Count']):>12,} {row['Dropoff_Pct']:>11.2f}%")

print("-" * 90)

# =============================================================================
# STEP 3: IDENTIFY HIGHEST DROP-OFF STAGE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: HIGHEST DROP-OFF ANALYSIS")
print("=" * 70)

# Find stage with highest drop-off (excluding stage 1)
dropoff_stages = funnel_df[funnel_df['Dropoff_Count'] > 0]
max_dropoff_idx = dropoff_stages['Dropoff_Count'].idxmax()
max_dropoff_stage = funnel_df.loc[max_dropoff_idx]

print(f"""
HIGHEST DROP-OFF IDENTIFIED:
============================

Stage: {max_dropoff_stage['Stage']}
Drop-off Count: {int(max_dropoff_stage['Dropoff_Count']):,} orders
Drop-off Rate: {max_dropoff_stage['Dropoff_Pct']:.2f}%

This means {int(max_dropoff_stage['Dropoff_Count']):,} orders were lost between 
the previous stage and this stage.
""")

# Determine which transition has highest drop-off
if max_dropoff_idx == 1:
    transition = "Order Placed → Payment Approved"
    business_reasons = """
POSSIBLE BUSINESS REASONS FOR DROP-OFF (Placed → Approved):
------------------------------------------------------------
1. PAYMENT FAILURES
   - Credit card declined
   - Insufficient funds
   - Payment gateway errors
   
2. ABANDONED CHECKOUT
   - Customer left before completing payment
   - Unexpected shipping costs revealed at checkout
   - Complicated checkout process
   
3. FRAUD DETECTION
   - Payment flagged as potentially fraudulent
   - Additional verification required

RECOMMENDED ACTIONS:
- Analyze failed payment error codes
- Implement cart abandonment email campaigns
- Simplify checkout process
- Offer multiple payment methods
"""

elif max_dropoff_idx == 2:
    transition = "Payment Approved → Shipped"
    business_reasons = """
POSSIBLE BUSINESS REASONS FOR DROP-OFF (Approved → Shipped):
------------------------------------------------------------
1. INVENTORY ISSUES
   - Product out of stock after payment
   - Supplier delays
   - Warehouse fulfillment problems
   
2. SELLER ISSUES
   - Seller unable to fulfill order
   - Seller cancellation
   - Seller shipping deadline missed
   
3. CUSTOMER CANCELLATIONS
   - Customer requested cancellation after payment
   - Buyer's remorse
   
4. ORDER PROCESSING DELAYS
   - Manual review required
   - Address verification issues

RECOMMENDED ACTIONS:
- Implement real-time inventory tracking
- Set seller performance metrics
- Improve warehouse operations
- Automate order processing
"""

elif max_dropoff_idx == 3:
    transition = "Shipped → Delivered"
    business_reasons = """
POSSIBLE BUSINESS REASONS FOR DROP-OFF (Shipped → Delivered):
------------------------------------------------------------
1. DELIVERY FAILURES
   - Customer not available to receive
   - Incorrect address
   - Access issues (gated communities, apartments)
   
2. SHIPPING PROBLEMS
   - Package lost in transit
   - Damaged during shipping
   - Delayed beyond acceptable window
   
3. RETURNS/REFUSALS
   - Customer refused delivery
   - Package returned to sender
   
4. DATA GAPS
   - Orders still in transit (not yet delivered)
   - Delivery confirmation not recorded

RECOMMENDED ACTIONS:
- Partner with reliable shipping carriers
- Implement delivery tracking notifications
- Offer flexible delivery options
- Improve address validation at checkout
"""

print(f"TRANSITION WITH HIGHEST DROP-OFF: {transition}")
print(business_reasons)

# =============================================================================
# STEP 4: CREATE FUNNEL VISUALIZATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: CREATE FUNNEL VISUALIZATION")
print("=" * 70)

# Set up the figure
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('E-Commerce Order Lifecycle Funnel Analysis', fontsize=16, fontweight='bold')

# --- Chart 1: Funnel Bar Chart ---
ax1 = axes[0]

stages = funnel_df['Stage_Short'].tolist()
orders = funnel_df['Orders'].tolist()
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

# Create horizontal bar chart (funnel style)
bars = ax1.barh(stages[::-1], orders[::-1], color=colors[::-1], edgecolor='white', linewidth=2)

# Add value labels
for i, (bar, order) in enumerate(zip(bars, orders[::-1])):
    pct = funnel_df.iloc[3-i]['Pct_of_Stage_1']
    ax1.text(bar.get_width() + 1000, bar.get_y() + bar.get_height()/2, 
             f'{order:,} ({pct:.1f}%)', va='center', fontsize=11, fontweight='bold')

ax1.set_xlabel('Number of Orders', fontsize=12)
ax1.set_title('Funnel Stages - Order Count', fontsize=14, fontweight='bold')
ax1.set_xlim(0, max(orders) * 1.25)

# Remove top and right spines
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Chart 2: Conversion & Drop-off Rates ---
ax2 = axes[1]

x = np.arange(len(stages) - 1)  # We have 3 transitions
transitions = ['Placed→Approved', 'Approved→Shipped', 'Shipped→Delivered']
conv_rates = funnel_df['Conversion_Rate'].tolist()[1:]  # Skip first stage
dropoff_rates = funnel_df['Dropoff_Pct'].tolist()[1:]

# Create grouped bar chart
width = 0.35
bars1 = ax2.bar(x - width/2, conv_rates, width, label='Conversion Rate %', color='#27ae60')
bars2 = ax2.bar(x + width/2, dropoff_rates, width, label='Drop-off Rate %', color='#e74c3c')

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#27ae60')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.2f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')

# Highlight highest drop-off
max_dropoff_transition_idx = dropoff_rates.index(max(dropoff_rates))
bars2[max_dropoff_transition_idx].set_edgecolor('black')
bars2[max_dropoff_transition_idx].set_linewidth(3)

ax2.set_xlabel('Funnel Transition', fontsize=12)
ax2.set_ylabel('Percentage', fontsize=12)
ax2.set_title('Conversion vs Drop-off Rates by Stage', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(transitions, fontsize=10)
ax2.legend(loc='upper right')
ax2.set_ylim(0, 105)
ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.3)

# Remove top and right spines
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('funnel_visualization.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\nFunnel visualization saved as: funnel_visualization.png")

# =============================================================================
# STEP 5: CREATE DETAILED DROP-OFF CHART
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: DROP-OFF WATERFALL CHART")
print("=" * 70)

fig2, ax = plt.subplots(figsize=(12, 7))

# Waterfall chart data
stages_waterfall = ['Order\nPlaced', 'Drop-off 1', 'Payment\nApproved', 
                    'Drop-off 2', 'Shipped', 'Drop-off 3', 'Delivered']
values = [stage_1_orders, 
          -(stage_1_orders - stage_2_orders),
          stage_2_orders,
          -(stage_2_orders - stage_3_orders),
          stage_3_orders,
          -(stage_3_orders - stage_4_orders),
          stage_4_orders]

# Calculate cumulative for waterfall positioning
cumulative = [0]
running = 0
for i, v in enumerate(values):
    if i % 2 == 0:  # Stage bars
        cumulative.append(0)
        running = v
    else:  # Drop-off bars
        cumulative.append(running + v)
        running = running + v

# Colors
bar_colors = []
for i, v in enumerate(values):
    if i % 2 == 0:  # Stage
        bar_colors.append('#3498db')
    else:  # Drop-off
        bar_colors.append('#e74c3c')

# Create bars
x_pos = np.arange(len(stages_waterfall))
bottom_vals = []
for i in range(len(values)):
    if i == 0:
        bottom_vals.append(0)
    elif i % 2 == 1:  # Drop-off - starts from previous stage value
        bottom_vals.append(values[i-1] + values[i])
    else:  # Stage after drop-off
        bottom_vals.append(0)

for i, (stage, val) in enumerate(zip(stages_waterfall, values)):
    if i % 2 == 0:  # Stage bars
        ax.bar(x_pos[i], val, color='#3498db', edgecolor='white', linewidth=2)
        ax.text(x_pos[i], val + 1500, f'{val:,}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    else:  # Drop-off bars  
        ax.bar(x_pos[i], abs(val), bottom=values[i-1] + val, color='#e74c3c', 
               edgecolor='white', linewidth=2)
        ax.text(x_pos[i], values[i-1] + val/2, f'-{abs(val):,}\n({abs(val)/values[i-1]*100:.1f}%)', 
                ha='center', va='center', fontsize=9, fontweight='bold', color='white')

ax.set_xticks(x_pos)
ax.set_xticklabels(stages_waterfall, fontsize=10)
ax.set_ylabel('Number of Orders', fontsize=12)
ax.set_title('Order Funnel Waterfall - Where Orders Are Lost', fontsize=14, fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add legend
blue_patch = mpatches.Patch(color='#3498db', label='Orders at Stage')
red_patch = mpatches.Patch(color='#e74c3c', label='Orders Lost (Drop-off)')
ax.legend(handles=[blue_patch, red_patch], loc='upper right')

plt.tight_layout()
plt.savefig('funnel_dropoff_waterfall.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("Waterfall chart saved as: funnel_dropoff_waterfall.png")

# =============================================================================
# STEP 6: SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: EXECUTIVE SUMMARY")
print("=" * 70)

print(f"""
FUNNEL PERFORMANCE SUMMARY
==========================

OVERALL METRICS:
- Total orders in funnel: {stage_1_orders:,}
- Successfully delivered: {stage_4_orders:,}
- Overall success rate: {stage_4_orders/stage_1_orders*100:.2f}%
- Total orders lost: {stage_1_orders - stage_4_orders:,} ({(stage_1_orders-stage_4_orders)/stage_1_orders*100:.2f}%)

STAGE-BY-STAGE PERFORMANCE:
1. Order Placed → Payment Approved: {funnel_df.iloc[1]['Conversion_Rate']:.2f}% conversion
   - Lost {int(funnel_df.iloc[1]['Dropoff_Count']):,} orders
   
2. Payment Approved → Shipped: {funnel_df.iloc[2]['Conversion_Rate']:.2f}% conversion
   - Lost {int(funnel_df.iloc[2]['Dropoff_Count']):,} orders
   
3. Shipped → Delivered: {funnel_df.iloc[3]['Conversion_Rate']:.2f}% conversion
   - Lost {int(funnel_df.iloc[3]['Dropoff_Count']):,} orders

BIGGEST OPPORTUNITY:
The highest drop-off occurs at: {max_dropoff_stage['Stage']}
- {int(max_dropoff_stage['Dropoff_Count']):,} orders lost ({max_dropoff_stage['Dropoff_Pct']:.2f}%)
- Addressing this stage first would have the highest impact

KEY INSIGHT:
Despite the drop-off, {stage_4_orders/stage_1_orders*100:.1f}% of all orders reach delivery -
this is a STRONG conversion rate indicating healthy operations.
The main opportunity is in the final mile delivery process.
""")

print("\n" + "=" * 70)
print("FUNNEL METRICS ANALYSIS COMPLETE")
print("=" * 70)

# Close output
output.close()
sys.stdout = output.terminal

print("\nResults saved to: funnel_metrics_results.txt")
print("Visualizations saved:")
print("  - funnel_visualization.png")
print("  - funnel_dropoff_waterfall.png")
