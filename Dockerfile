# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirment.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirment.txt

# Copy application code
COPY server.py .
COPY .env* ./

# Create necessary directories
RUN mkdir -p faiss_index

# Expose port (Railway/Render will override this)
EXPOSE 8080

# Set environment variables for cloud deployment
ENV PYTHONUNBUFFERED=1
ENV ENABLE_MCP_TOOLS=false
ENV ENABLE_FAISS_MEMORY=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Run the application
CMD ["python", "server.py"]
