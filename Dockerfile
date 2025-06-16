# Nexus Forge Production Dockerfile - Multi-stage build for optimized production deployment

# =============================================================================
# Stage 1: Base image with system dependencies
# =============================================================================
FROM python:3.11-slim as base

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user early
RUN groupadd --gid 1000 nexus && \
    useradd --uid 1000 --gid nexus --shell /bin/bash --create-home nexus

# Set security-focused environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# =============================================================================
# Stage 2: Build dependencies and install packages
# =============================================================================
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set work directory
WORKDIR /app

# Copy and install Python dependencies
COPY pyproject.toml ./
COPY README.md ./

# Install dependencies to virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -e .[production]

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM base as production

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set work directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/logs /app/cache /app/uploads /tmp && \
    chown -R nexus:nexus /app /tmp

# Copy application code
COPY --chown=nexus:nexus src/ ./src/
COPY --chown=nexus:nexus config/ ./config/

# Copy static files if they exist
COPY --chown=nexus:nexus frontend/build/ ./static/ 2>/dev/null || mkdir -p ./static

# Set up proper permissions
RUN chmod -R 755 /app && \
    chmod -R 750 /app/logs /app/cache /app/uploads

# Switch to non-root user
USER nexus

# Set production environment
ENV ENVIRONMENT=production \
    PORT=8080 \
    WORKERS=4 \
    LOG_LEVEL=info

# Health check with proper endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose ports
EXPOSE 8080 8090

# Use exec form for better signal handling
CMD ["python", "-m", "src.backend.main"]

# =============================================================================
# Stage 4: Development image (optional)
# =============================================================================
FROM production as development

# Switch back to root for development tools installation
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    ipython \
    jupyter

# Switch back to nexus user
USER nexus

# Override for development
ENV ENVIRONMENT=development \
    DEBUG=true \
    LOG_LEVEL=debug

# Development command
CMD ["python", "-m", "src.backend.main", "--reload"]

# =============================================================================
# Labels for metadata
# =============================================================================
LABEL maintainer="Nexus Forge Team" \
      version="1.0.0" \
      description="Nexus Forge - AI-powered content generation platform" \
      org.opencontainers.image.title="Nexus Forge" \
      org.opencontainers.image.description="Production-ready AI content generation service" \
      org.opencontainers.image.version="1.0.0" \
      org.opencontainers.image.authors="Nexus Forge Team" \
      org.opencontainers.image.vendor="Nexus Forge" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.url="https://github.com/nexus-forge/nexus-forge" \
      org.opencontainers.image.source="https://github.com/nexus-forge/nexus-forge" \
      org.opencontainers.image.documentation="https://docs.nexus-forge.com"