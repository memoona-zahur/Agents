import json
from agents.tool import function_tool
from datetime import datetime, timedelta

@function_tool("analyze_sales_data")
def analyze_sales_data(product_name: str):
    """
    Analyzes historical sales data from sales.json for a specific product.
    Returns a summary of past sales, including total quantity sold and the number of sales records.
    """
    try:
        with open("D:\Hackathon_Project\EcoSync_Logistics_Agent\src\ecosync_logistics_agent\custom_tools\sales.json", "r") as f:
            sales_data = json.load(f)
    except FileNotFoundError:
        return "Error: sales.json file not found."
    except json.JSONDecodeError:
        return "Error: Could not decode sales.json."

    product_sales = [sale for sale in sales_data if sale["product_name"].lower() == product_name.lower()]

    if not product_sales:
        return f"No sales data found for '{product_name}'."

    total_quantity_sold = sum(sale["quantity_sold"] for sale in product_sales)
    number_of_sales = len(product_sales)

    summary = f"Historical sales analysis for '{product_name}':\n"
    summary += f"- Number of sales records found: {number_of_sales}\n"
    summary += f"- Total quantity sold across all records: {total_quantity_sold}\n"

    if number_of_sales > 0:
        average_quantity_per_sale = total_quantity_sold / number_of_sales
        summary += f"- Average quantity per sale: {average_quantity_per_sale:.2f}\n"

        # You could add more analysis here if needed, like recent trends
        # For example, to get sales in the last month (assuming 'sale_date' is in YYYY-MM-DD format):
        one_month_ago = datetime.now().date().replace(day=1) - timedelta(days=1)
        recent_sales = [sale for sale in product_sales if datetime.strptime(sale["sale_date"], "%Y-%m-%d").date() > one_month_ago]
        if recent_sales:
            recent_total_quantity = sum(sale["quantity_sold"] for sale in recent_sales)
            recent_sales_count = len(recent_sales)
            summary += f"- Quantity sold in the last month (approximately): {recent_total_quantity} across {recent_sales_count} records.\n"
        else:
            summary += "- No sales recorded in the last month.\n"

    else:
        summary += "- No sales records available for this product.\n"

    return summary