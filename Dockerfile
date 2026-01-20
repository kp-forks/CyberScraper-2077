# Use Python 3.12 for better performance and compatibility
FROM python:3.12-slim-bookworm

# Set the working directory in the container
WORKDIR /app

# Install system dependencies including browser dependencies for Playwright/Patchright
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    git \
    tor \
    tor-geoipdb \
    netcat-traditional \
    curl \
    build-essential \
    python3-dev \
    libffi-dev \
    procps \
    # Browser dependencies for Playwright/Patchright
    libglib2.0-0 \
    libnspr4 \
    libnss3 \
    libdbus-1-3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libcairo2 \
    libpango-1.0-0 \
    libasound2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configure Tor - Simplified configuration
RUN echo "SocksPort 9050" >> /etc/tor/torrc && \
    echo "ControlPort 9051" >> /etc/tor/torrc && \
    echo "CookieAuthentication 1" >> /etc/tor/torrc && \
    echo "DataDirectory /var/lib/tor" >> /etc/tor/torrc

# Set correct permissions for Tor
RUN chown -R debian-tor:debian-tor /var/lib/tor && \
    chmod 700 /var/lib/tor

# Copy local files
COPY . .

# Create and activate a virtual environment
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Install Python dependencies (includes PySocks for Tor support)
# Added retries and timeout for network reliability
RUN pip install --no-cache-dir --timeout=120 --retries=3 -r requirements.txt

# Install patchright browsers (package already in requirements.txt)
RUN patchright install chromium && patchright install chrome

# Create run script with proper Tor startup
RUN echo '#!/bin/bash\n\
\n\
# Start Tor service\n\
echo "Starting Tor service..."\n\
service tor start\n\
\n\
# Wait for Tor to be ready\n\
echo "Waiting for Tor to start..."\n\
for i in {1..30}; do\n\
    if ps aux | grep -v grep | grep -q /usr/bin/tor; then\n\
        echo "Tor process is running"\n\
        if nc -z localhost 9050; then\n\
            echo "Tor SOCKS port is listening"\n\
            break\n\
        fi\n\
    fi\n\
    if [ $i -eq 30 ]; then\n\
        echo "Warning: Tor might not be ready, but continuing..."\n\
    fi\n\
    sleep 1\n\
done\n\
\n\
# Verify Tor status\n\
echo "Checking Tor service status:"\n\
service tor status\n\
\n\
# Export API key if provided\n\
if [ ! -z "$OPENAI_API_KEY" ]; then\n\
    export OPENAI_API_KEY=$OPENAI_API_KEY\n\
    echo "OpenAI API key configured"\n\
fi\n\
\n\
if [ ! -z "$GOOGLE_API_KEY" ]; then\n\
    export GOOGLE_API_KEY=$GOOGLE_API_KEY\n\
    echo "Google API key configured"\n\
fi\n\
\n\
# Start the application with explicit host binding\n\
echo "Starting CyberScraper 2077..."\n\
streamlit run --server.address 0.0.0.0 --server.port 8501 main.py\n\
' > /app/run.sh

RUN chmod +x /app/run.sh

# Expose ports
EXPOSE 8501 9050 9051

# Set the entrypoint
ENTRYPOINT ["/app/run.sh"]