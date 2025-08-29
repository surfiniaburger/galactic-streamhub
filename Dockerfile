# Use the official Python 3.12 slim image, which is based on Debian
FROM python:3.12-slim

# Use bash and exit on pipeline failures
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install uv, a modern Python package installer, from its official release
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# --- MODIFIED SECTION ---
# Install system dependencies.
# Version pinning is removed to allow apt to resolve the correct packages for the
# underlying OS (Debian 13 "Trixie"), making the build more robust.
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For matplotlib/Pillow
    libfreetype6-dev \
    libpng-dev \
    pkg-config \
    # For opencv-python (cv2). libgl1 is the modern replacement for libgl1-mesa-glx
    libgl1 \
    libglib2.0-0 \
    # General build tools and network utilities
    curl \
    build-essential \
    # Install NodeSource repo for a recent version of Node.js
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    # Install Node.js from the newly added repository
    && apt-get install -y --no-install-recommends nodejs \
    # Clean up APT caches to keep the final image smaller
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the application directory
WORKDIR /app

# Copy the rest of the application source code
COPY . /app

# Add the virtual environment's bin directory to the system's PATH
# This allows running installed Python packages directly (like uvicorn)
ENV PATH="/app/.venv/bin:$PATH"

# Install Python dependencies using the locked requirements file for reproducibility
RUN uv sync --frozen

# Command to run the Uvicorn server.
# It will listen on all network interfaces on port 8080, which is standard for Cloud Run.
CMD ["/app/.venv/bin/uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]