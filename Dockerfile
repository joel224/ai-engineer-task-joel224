# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory in the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.lock.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.lock.txt

# 4. Copy the application code into the container
COPY simple_agent.py .

# 5. Make port 8000 available to the world outside this container
EXPOSE 8000

# 6. Define environment variables needed by the app.
# These can be overridden at runtime.
ENV PDF_PATH="/data/document.pdf"

# 7. Run the application when the container launches
# Use 0.0.0.0 to make it accessible from outside the container
CMD ["uvicorn", "simple_agent:app", "--host", "0.0.0.0", "--port", "8000"]
