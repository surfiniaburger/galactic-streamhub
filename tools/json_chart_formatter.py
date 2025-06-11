# /Users/surfiniaburger/Desktop/app/tools/json_chart_formatter.py
import json
from typing import Any, Dict, List, Literal, Union
import logging

class JSONChartFormatter:
    """A tool to format chart data into a JSON object for Chart.js."""

    def __call__(self, chart_type: Literal["bar", "line"], data: List[Dict[str, Union[str, int, float]]], category_field: str, value_field: str, title: str, xlabel: str, ylabel: str) -> str:
        """
        Formats chart data into a JSON object for Chart.js.

        Args:
            chart_type: The type of chart to generate (e.g., "bar", "line").
            data: The data for the chart, as a list of dictionaries.
            category_field: The name of the field containing the category labels.
            value_field: The name of the field containing the values.
            title: The title of the chart.
            xlabel: The label for the x-axis.
            ylabel: The label for the y-axis.

        Returns:
            A JSON string representing the Chart.js configuration.
        """
        try:
            labels = [item[category_field] for item in data]
            values = [item[value_field] for item in data]

            chart_config = {
                "type": chart_type,
                "data": {
                    "labels": labels,
                    "datasets": [{
                        "label": title,
                        "data": values,
                        "backgroundColor": 'rgba(75, 192, 192, 0.2)',
                        "borderColor": 'rgba(75, 192, 192, 1)',
                        "borderWidth": 1
                    }]
                },
                "options": {
                    "scales": {
                        "x": {
                            "title": {
                                "display": True,
                                "text": xlabel
                            }
                        },
                        "y": {
                            "title": {
                                "display": True,
                                "text": ylabel
                            },
                            "beginAtZero": True
                        }
                    }
                }
            }
            return json.dumps(chart_config)
        except Exception as e:
            logging.error(f"Error formatting chart data: {e}")
            return json.dumps({"error": "Error formatting chart data."})
