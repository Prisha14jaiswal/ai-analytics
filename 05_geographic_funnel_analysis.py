"""
================================================================================
E-Commerce Geographic Funnel Analysis
================================================================================
Purpose: Segment funnel performance by customer_state
Author: Data Analyst
Date: 2026-01-20
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

output = OutputWriter('geographic_funnel_results.txt')
sys.stdout = output

print("=" * 70)
print("GEOGRAPHIC FUNNEL ANALYSIS BY CUSTOMER STATE")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1: LOAD DATA")
print("=" * 70)

df = pd.read_csv('merged_ecommerce_with_funnel.csv')
print(f"\nLoaded dataset: {len(df):,} rows")
print(f"Unique states: {df['customer_state'].nunique()}")

# =============================================================================
# STEP 2: CALCULATE METRICS BY STATE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2: CALCULATE FUNNEL METRICS BY STATE")
print("=" * 70)

# Aggregate at order level first (avoid counting items multiple times)
orders_df = df.drop_duplicates('order_id')

# Group by state and calculate funnel metrics
state_metrics = orders_df.groupby('customer_state').agg({
    'order_id': 'count',
    'funnel_order_placed': 'sum',
    'funnel_payment_approved': 'sum',
    'funnel_shipped': 'sum',
    'funnel_delivered': 'sum',
    'funnel_excluded': 'sum'
}).reset_index()

state_metrics.columns = ['state', 'total_orders', 'placed', 'approved', 'shipped', 'delivered', 'excluded']

# Calculate rates
state_metrics['delivery_rate'] = (state_metrics['delivered'] / state_metrics['approved'] * 100).round(2)
state_metrics['ship_rate'] = (state_metrics['shipped'] / state_metrics['approved'] * 100).round(2)
state_metrics['failure_rate'] = (100 - state_metrics['delivery_rate']).round(2)

# Sort by total orders for context
state_metrics = state_metrics.sort_values('total_orders', ascending=False)

print("\nSTATE-LEVEL FUNNEL METRICS:")
print("-" * 100)
print(f"{'State':<8} {'Orders':>10} {'Approved':>10} {'Shipped':>10} {'Delivered':>10} {'Delivery%':>12} {'Failure%':>10}")
print("-" * 100)

for _, row in state_metrics.iterrows():
    print(f"{row['state']:<8} {row['total_orders']:>10,} {row['approved']:>10,} {row['shipped']:>10,} {row['delivered']:>10,} {row['delivery_rate']:>11.2f}% {row['failure_rate']:>9.2f}%")

print("-" * 100)

# =============================================================================
# STEP 3: IDENTIFY HIGH FAILURE STATES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3: IDENTIFY HIGH DELIVERY FAILURE STATES")
print("=" * 70)

# Calculate overall average delivery rate
avg_delivery_rate = state_metrics['delivery_rate'].mean()
print(f"\nOverall Average Delivery Rate: {avg_delivery_rate:.2f}%")

# States with below-average delivery
low_performers = state_metrics[state_metrics['delivery_rate'] < avg_delivery_rate].copy()
low_performers = low_performers.sort_values('delivery_rate')

print(f"\nStates with BELOW AVERAGE Delivery Rate:")
print("-" * 60)
for _, row in low_performers.iterrows():
    diff = avg_delivery_rate - row['delivery_rate']
    print(f"  {row['state']}: {row['delivery_rate']:.2f}% ({diff:.2f}% below avg) - {row['total_orders']:,} orders")

# High failure states (more than 2% failure)
high_failure = state_metrics[state_metrics['failure_rate'] > 2].copy()
high_failure = high_failure.sort_values('failure_rate', ascending=False)

print(f"\nStates with HIGH Failure Rate (>2%):")
print("-" * 60)
if len(high_failure) > 0:
    for _, row in high_failure.iterrows():
        undelivered = row['approved'] - row['delivered']
        print(f"  {row['state']}: {row['failure_rate']:.2f}% failure ({undelivered:,.0f} undelivered orders)")
else:
    print("  No states with failure rate > 2%")

# =============================================================================
# STEP 4: TOP AND BOTTOM PERFORMERS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 4: TOP AND BOTTOM PERFORMING STATES")
print("=" * 70)

# Filter to states with significant volume (at least 500 orders)
significant_states = state_metrics[state_metrics['total_orders'] >= 500].copy()

top_5 = significant_states.nlargest(5, 'delivery_rate')
bottom_5 = significant_states.nsmallest(5, 'delivery_rate')

print("\nTOP 5 STATES (Highest Delivery Rate) - min 500 orders:")
print("-" * 60)
for i, (_, row) in enumerate(top_5.iterrows(), 1):
    print(f"  {i}. {row['state']}: {row['delivery_rate']:.2f}% ({row['total_orders']:,} orders)")

print("\nBOTTOM 5 STATES (Lowest Delivery Rate) - min 500 orders:")
print("-" * 60)
for i, (_, row) in enumerate(bottom_5.iterrows(), 1):
    print(f"  {i}. {row['state']}: {row['delivery_rate']:.2f}% ({row['total_orders']:,} orders)")

# =============================================================================
# STEP 5: CREATE VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5: CREATE VISUALIZATIONS")
print("=" * 70)

# Chart 1: Delivery Rate by State (sorted)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Geographic Analysis: Delivery Performance by State', fontsize=16, fontweight='bold')

# --- Chart 1: All States Delivery Rate ---
ax1 = axes[0, 0]
sorted_states = state_metrics.sort_values('delivery_rate', ascending=True)
colors = ['#e74c3c' if x < avg_delivery_rate else '#27ae60' for x in sorted_states['delivery_rate']]
bars = ax1.barh(sorted_states['state'], sorted_states['delivery_rate'], color=colors)
ax1.axvline(x=avg_delivery_rate, color='#3498db', linestyle='--', linewidth=2, label=f'Avg: {avg_delivery_rate:.1f}%')
ax1.set_xlabel('Delivery Rate (%)', fontsize=11)
ax1.set_title('Delivery Rate by State', fontsize=13, fontweight='bold')
ax1.legend(loc='lower right')
ax1.set_xlim(90, 100)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# --- Chart 2: Order Volume by State ---
ax2 = axes[0, 1]
top_10_volume = state_metrics.nlargest(10, 'total_orders')
ax2.barh(top_10_volume['state'], top_10_volume['total_orders'], color='#3498db')
for i, (_, row) in enumerate(top_10_volume.iterrows()):
    ax2.text(row['total_orders'] + 500, i, f"{row['total_orders']:,}", va='center', fontsize=9)
ax2.set_xlabel('Number of Orders', fontsize=11)
ax2.set_title('Top 10 States by Order Volume', fontsize=13, fontweight='bold')
ax2.invert_yaxis()
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# --- Chart 3: Failure Rate Comparison (Top 10 by volume) ---
ax3 = axes[1, 0]
top_10_volume_sorted = top_10_volume.sort_values('failure_rate', ascending=False)
colors3 = ['#e74c3c' if x > 1.5 else '#f39c12' if x > 1 else '#27ae60' for x in top_10_volume_sorted['failure_rate']]
bars3 = ax3.bar(top_10_volume_sorted['state'], top_10_volume_sorted['failure_rate'], color=colors3)
ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='1% threshold')
ax3.set_ylabel('Failure Rate (%)', fontsize=11)
ax3.set_xlabel('State', fontsize=11)
ax3.set_title('Delivery Failure Rate (Top 10 States by Volume)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add value labels
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- Chart 4: Volume vs Delivery Rate Scatter ---
ax4 = axes[1, 1]
scatter = ax4.scatter(state_metrics['total_orders'], state_metrics['delivery_rate'], 
                      s=100, c=state_metrics['failure_rate'], cmap='RdYlGn_r', 
                      alpha=0.7, edgecolors='black', linewidth=0.5)

# Add state labels for top volume states
for _, row in top_10_volume.iterrows():
    ax4.annotate(row['state'], (row['total_orders'], row['delivery_rate']), 
                 fontsize=8, ha='center', va='bottom')

ax4.set_xlabel('Order Volume', fontsize=11)
ax4.set_ylabel('Delivery Rate (%)', fontsize=11)
ax4.set_title('Order Volume vs Delivery Rate by State', fontsize=13, fontweight='bold')
ax4.axhline(y=avg_delivery_rate, color='#3498db', linestyle='--', alpha=0.5)
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Failure Rate (%)')
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('geographic_funnel_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("\nVisualization saved: geographic_funnel_analysis.png")

# =============================================================================
# STEP 6: BUSINESS EXPLANATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6: HOW GEOGRAPHY IMPACTS LOGISTICS PERFORMANCE")
print("=" * 70)

# Get state info for context
top_state = top_5.iloc[0]
bottom_state = bottom_5.iloc[0]

print(f"""
GEOGRAPHIC IMPACT ON LOGISTICS PERFORMANCE
==========================================

1. DISTANCE FROM FULFILLMENT CENTERS
   - States closer to major distribution hubs (SP, RJ) have BETTER delivery rates
   - Remote states face longer transit times, more handling, higher failure risk
   - Best performer: {top_state['state']} ({top_state['delivery_rate']:.2f}%)
   - Needs improvement: {bottom_state['state']} ({bottom_state['delivery_rate']:.2f}%)

2. INFRASTRUCTURE QUALITY
   - Developed states (SP, MG, RJ) have better road networks
   - Rural/remote areas face: unpaved roads, limited carrier presence
   - Last-mile delivery is more challenging in less accessible regions

3. URBAN vs RURAL DISTRIBUTION
   - Urban centers: Dense population, easier delivery logistics
   - Rural areas: Scattered addresses, longer delivery routes
   - States with more rural population tend to have lower delivery rates

4. CARRIER COVERAGE
   - Major carriers focus on high-volume urban areas
   - Remote states may rely on smaller, less reliable carriers
   - Limited delivery options = higher failure rates

5. SEASONAL/WEATHER FACTORS
   - Some regions face seasonal weather challenges
   - Rainy seasons can disrupt transportation
   - Extreme weather may delay deliveries

KEY INSIGHTS FOR BUSINESS:
--------------------------
""")

# Calculate impact metrics
total_approved = state_metrics['approved'].sum()
total_undelivered = state_metrics['approved'].sum() - state_metrics['delivered'].sum()

# Impact of improving bottom performers
bottom_5_undelivered = bottom_5['approved'].sum() - bottom_5['delivered'].sum()
potential_improvement = bottom_5_undelivered * 0.5  # If we reduce failures by 50%

print(f"""
Current State:
- Total approved orders: {total_approved:,}
- Total undelivered: {total_undelivered:,.0f} ({total_undelivered/total_approved*100:.2f}%)

Bottom 5 States Impact:
- Undelivered orders in bottom 5: {bottom_5_undelivered:,.0f}
- If we improve these by 50%: +{potential_improvement:,.0f} successful deliveries

RECOMMENDED ACTIONS:
1. Prioritize logistics partnerships in underperforming states
2. Consider regional fulfillment centers for high-volume remote states
3. Implement address validation specifically for problematic regions
4. Offer alternative delivery options (pickup points) in difficult areas
""")

# =============================================================================
# STEP 7: SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7: SUMMARY BY REGION")
print("=" * 70)

# Group states by performance tier
state_metrics['performance_tier'] = pd.cut(
    state_metrics['delivery_rate'],
    bins=[0, 97, 98.5, 100],
    labels=['Needs Improvement', 'Average', 'High Performer']
)

tier_summary = state_metrics.groupby('performance_tier').agg({
    'state': 'count',
    'total_orders': 'sum',
    'delivered': 'sum',
    'approved': 'sum'
}).reset_index()

tier_summary['delivery_rate'] = (tier_summary['delivered'] / tier_summary['approved'] * 100).round(2)

print("\nPERFORMANCE TIER SUMMARY:")
print("-" * 70)
for _, row in tier_summary.iterrows():
    print(f"  {row['performance_tier']}: {row['state']} states, {row['total_orders']:,} orders, {row['delivery_rate']:.2f}% delivery rate")

print("\n" + "=" * 70)
print("GEOGRAPHIC ANALYSIS COMPLETE")
print("=" * 70)

# Close output
output.close()
sys.stdout = output.terminal

print("\nResults saved to: geographic_funnel_results.txt")
print("Visualization saved to: geographic_funnel_analysis.png")
