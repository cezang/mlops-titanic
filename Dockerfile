# Base image Python 3.10
FROM python:3.10

# Install Poetry
RUN pip install --upgrade pip && pip install poetry

# Set working directory
WORKDIR /app

# Copy dependency management files to the container
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create a virtual environment
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy the rest of the application code
COPY . .

# Ensure the static directory exists
RUN mkdir -p /app/api/static

# Expose the port on which the application will run (optional)
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]