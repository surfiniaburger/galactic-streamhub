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
    



# Add this new function to your tools/chart_tool.py file
import numpy as np

def generate_grouped_bar_chart(
    data: List[Dict[str, Any]],
    group_field: str,
    title: Optional[str] = "Grouped Bar Chart",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None
) -> Optional[str]:
    """
    Generates a grouped bar chart from a list of dictionaries and saves it as a PNG.
    Each dictionary in the list represents a main group (e.g., 'Chemotherapy').
    The other keys in the dictionary represent the sub-bars within that group.

    Args:
        data: e.g., [{"group": "Chemo", "Stage I": 50, "Stage II": 30}, {"group": "Immuno", "Stage I": 60, "Stage II": 45}]
        group_field: The key that identifies the main group label (e.g., "group").
        title: The title of the chart.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.

    Returns:
        The relative web URL of the saved chart image, or None on error.
    """
    if not data:
        logger.warning("No data provided for grouped bar chart generation.")
        return None
    try:
        main_groups = [item.get(group_field) for item in data]
        if any(g is None for g in main_groups):
            logger.error(f"Missing '{group_field}' in some data items for grouped bar chart.")
            return None

        # Dynamically discover the sub-categories (the bars within each group)
        # and collect all numeric data.
        sub_categories = set()
        for item in data:
            for key in item:
                if key != group_field and isinstance(item[key], (int, float)):
                    sub_categories.add(key)
        
        sub_categories = sorted(list(sub_categories))
        if not sub_categories:
            logger.error("No numeric sub-categories found in the data for grouped bar chart.")
            return None

        # Prepare data for plotting
        values_by_subcategory = {sub: [] for sub in sub_categories}
        for group in main_groups:
            item = next(d for d in data if d.get(group_field) == group)
            for sub in sub_categories:
                values_by_subcategory[sub].append(item.get(sub, 0)) # Default to 0 if sub-category is missing for a group

        x = np.arange(len(main_groups))  # The label locations
        width = 0.8 / len(sub_categories)  # The width of the bars, adjusted for number of sub-categories
        multiplier = 0

        fig, ax = plt.subplots(figsize=(12, 7))

        for subcategory, values in values_by_subcategory.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, values, width, label=subcategory)
            ax.bar_label(rects, padding=3, fmt='%.1f')
            multiplier += 1

        # Add some text for labels, title and axes ticks
        ax.set_ylabel(ylabel or "Values")
        ax.set_xlabel(xlabel or group_field)
        ax.set_title(title or "Grouped Bar Chart")
        ax.set_xticks(x + width * (len(sub_categories) - 1) / 2, main_groups)
        ax.legend(loc='upper left', ncols=len(sub_categories))
        ax.margins(y=0.1) # Add some margin to the top
        plt.tight_layout()

        filename = f"chart_{uuid.uuid4().hex}.png"
        filepath = os.path.join(STATIC_CHARTS_DIR, filename)
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Grouped bar chart saved to {filepath}")
        return f"/{filepath.replace(os.sep, '/')}"

    except Exception as e:
        logger.error(f"Error generating grouped bar chart: {e}", exc_info=True)
        return None