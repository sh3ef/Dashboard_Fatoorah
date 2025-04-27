

# 📊 Inventory and Sales Analysis Dashboard

A Streamlit-based dashboard was added to analyze inventory and sales data. The dashboard provides insights into stock levels, sales performance, and product efficiency.
A dashboard was added based on current inventory analysis.
The products table was used to calculate inventory from quantity column.
Sales were calculated based on invoices.
## Data Sources
- **Products Table**: Used to calculate current stock levels from the `quantity` column.
- **Sales Invoices**: Used to analyze sales data.
- **Sales Invoice Details**: Used to calculate sold quantities and revenue.

## Key Features
1. **Data Processing**:
   - Calculated sold quantities and revenue for each product.
   - Merged sales data with current stock levels.
   - Computed performance metrics such as margin, margin percentage, and stock efficiency.

2. **Visualizations**:
   - **Stock vs Sales Comparison**: Top 20 products.
   - **Efficiency Distribution**: Oversupplied, Balanced, Undersupplied.
   - **Revenue Analysis**: Includes COGS and margin.
   - **Current Stock vs Sales**: Comparison using bars and lines.
   - **Efficiency Comparison**: Stock and sales by efficiency categories.
   - **Pareto Analysis**: Identifies top 20% of products contributing to 80% of sales.
   - **Pricing Analysis**: Relationship between sale price and demand.
   - **Restock Analysis**: Identifies products needing restocking based on a minimum stock threshold.


3. **Interactive UI**:
   - Built using Streamlit for an interactive and user-friendly experience.
   - Visualizations are displayed in a multi-column layout for better organization.

## How to Run
1. Install dependencies:
   ```bash
   >> pip install streamlit plotly pandas
   >> python -m streamlit run Dashboard_based_on_current_stock.py
