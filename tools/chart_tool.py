# /Users/surfiniaburger/Desktop/app/tools/chart_tool.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI errors in server environments
import matplotlib.pyplot as plt
import os
import uuid
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

STATIC_CHARTS_DIR = "static/charts"
# Ensure the directory exists when the module is loaded
os.makedirs(STATIC_CHARTS_DIR, exist_ok=True)

def generate_simple_bar_chart(
    data: List[Dict[str, Any]],
    category_field: str,
    value_field: str,
    title: Optional[str] = "Bar Chart",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
) -> Optional[str]:
    """
    Generates a simple bar chart from a list of dictionaries and saves it as a PNG.
    The chart image is saved to the 'static/charts/' directory.

    Args:
        data: A list of dictionaries, e.g., [{"category": "A", "value": 10}, ...].
        category_field: The key in the dictionaries to use for categories (x-axis).
        value_field: The key in the dictionaries to use for values (y-axis).
        title: The title of the chart.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.

    Returns:
        The relative web URL of the saved chart image (e.g., "/static/charts/chart_uuid.png"),
        or None if an error occurs or no data is provided.
    """
    if not data:
        logger.warning("No data provided for bar chart generation.")
        return None
    try:
        categories = [item.get(category_field) for item in data]
        values_str = [item.get(value_field) for item in data]

        if any(c is None for c in categories) or any(v is None for v in values_str):
            logger.error(f"Missing '{category_field}' or '{value_field}' in some data items for bar chart.")
            return None
        
        try:
            values = [float(v) for v in values_str]
        except (ValueError, TypeError) as e:
            logger.error(f"Values for '{value_field}' are not all numeric for bar chart: {e}")
            return None

        plt.figure(figsize=(10, 6))
        plt.bar(categories, values)
        plt.title(title or "Bar Chart")
        plt.xlabel(xlabel or category_field)
        plt.ylabel(ylabel or value_field)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        filename = f"chart_{uuid.uuid4().hex}.png"
        filepath = os.path.join(STATIC_CHARTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()  # Close the figure to free memory
        logger.info(f"Bar chart saved to {filepath}")
        return f"/{filepath.replace(os.sep, '/')}" # Ensure forward slashes for URL
    except Exception as e:
        logger.error(f"Error generating bar chart: {e}", exc_info=True)
        return None

def generate_simple_line_chart(
    data: List[Dict[str, Any]],
    x_field: str,
    y_field: str,
    title: Optional[str] = "Line Chart",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
) -> Optional[str]:
    """
    Generates a simple line chart from a list of dictionaries and saves it as a PNG.
    The chart image is saved to the 'static/charts/' directory.
    Data should ideally be sorted by x_field for a meaningful line.
    """
    if not data:
        logger.warning("No data provided for line chart generation.")
        return None
    try:
        x_values_orig = [item.get(x_field) for item in data]
        y_values_str = [item.get(y_field) for item in data]

        if any(x is None for x in x_values_orig) or any(y is None for y in y_values_str):
            logger.error(f"Missing '{x_field}' or '{y_field}' in some data items for line chart.")
            return None

        # Attempt to convert x_values to numeric if possible, otherwise treat as categorical
        # For line charts, numeric x-values are common, but categorical can also work.
        # If they are strings that should be numeric (e.g., years as strings), convert them.
        # If they are truly categorical strings, matplotlib will handle them.
        try:
            # Try converting to float first, then int if that fails but they look like ints
            x_values = [float(x) for x in x_values_orig]
        except ValueError:
            x_values = x_values_orig # Keep as original if not purely numeric

        try:
            y_values = [float(y) for y in y_values_str]
        except (ValueError, TypeError) as e:
            logger.error(f"Y-values for '{y_field}' are not all numeric for line chart: {e}")
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='o', linestyle='-') # Added marker and linestyle
        plt.title(title or "Line Chart")
        plt.xlabel(xlabel or x_field)
        plt.ylabel(ylabel or y_field)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, linestyle='--', alpha=0.7) # Added grid for better readability
        plt.tight_layout()

        filename = f"chart_{uuid.uuid4().hex}.png"
        filepath = os.path.join(STATIC_CHARTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Line chart saved to {filepath}")
        return f"/{filepath.replace(os.sep, '/')}"
    except Exception as e:
        logger.error(f"Error generating line chart: {e}", exc_info=True)
        return None

def generate_pie_chart(
    data: List[Dict[str, Any]],
    labels_field: str,
    values_field: str,
    title: Optional[str] = "Pie Chart"
) -> Optional[str]:
    """
    Generates a simple pie chart from a list of dictionaries and saves it as a PNG.
    The chart image is saved to the 'static/charts/' directory.

    Args:
        data: A list of dictionaries, e.g., [{"label": "Apples", "value": 30}, ...].
        labels_field: The key in the dictionaries to use for pie slice labels.
        values_field: The key in the dictionaries to use for pie slice values.
        title: The title of the chart.

    Returns:
        The relative web URL of the saved chart image (e.g., "/static/charts/chart_uuid.png"),
        or None if an error occurs or no data is provided.
    """
    if not data:
        logger.warning("No data provided for pie chart generation.")
        return None
    try:
        labels = [item.get(labels_field) for item in data]
        values_str = [item.get(values_field) for item in data]

        if any(l is None for l in labels) or any(v is None for v in values_str):
            logger.error(f"Missing '{labels_field}' or '{values_field}' in some data items for pie chart.")
            return None

        try:
            values = [float(v) for v in values_str]
        except (ValueError, TypeError) as e:
            logger.error(f"Values for '{values_field}' are not all numeric for pie chart: {e}")
            return None

        plt.figure(figsize=(8, 8)) # Pie charts often look better square
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
        plt.title(title or "Pie Chart")
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()

        filename = f"chart_{uuid.uuid4().hex}.png"
        filepath = os.path.join(STATIC_CHARTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Pie chart saved to {filepath}")
        return f"/{filepath.replace(os.sep, '/')}"
    except Exception as e:
        logger.error(f"Error generating pie chart: {e}", exc_info=True)
        return None