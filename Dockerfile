FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY profiles ./profiles
COPY knowledge ./knowledge

RUN useradd --create-home falak
USER falak
EXPOSE 8000
CMD ["falak", "serve"]
