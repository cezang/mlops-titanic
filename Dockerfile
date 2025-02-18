# Base image Python 3.10 Slim (mniejszy obraz)
FROM python:3.10-slim

# Install Poetry
RUN pip install --upgrade pip && pip install poetry

# Set working directory
WORKDIR /app

# Set PYTHONPATH to ensure your app finds the modules
ENV PYTHONPATH="/app"

# Copy dependency management files to the container
COPY pyproject.toml poetry.lock* ./ 

# Skopiowanie pliku modelu do odpowiedniej lokalizacji w obrazie
# Jeśli model jest zbyt duży, najlepiej skorzystać z Persistent Storage lub innej lokalizacji
# COPY classification_model/trained_models/random_forest_0.1.0.pkl /app/classification_model/trained_models/random_forest_0.1.0.pkl

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application code
COPY . .

# Ensure the static directory exists
RUN mkdir -p /app/api/static

# Remove unnecessary cache files to reduce image size
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose the port on which the application will run (optional)
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
