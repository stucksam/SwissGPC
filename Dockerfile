FROM python:3.12.9
LABEL authors="Samuel Stucki"

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

RUN echo 'deb https://deb.debian.org/debian stable non-free contrib' >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y espeak-ng festival ffmpeg git hdf5-tools mbrola && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .
# Install Python dependencies from requirements.txt
# see https://github.com/SYSTRAN/faster-whisper/issues/1236 and https://github.com/m-bain/whisperX/issues/967 for CTranslate
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt && \
    python3 -m pip install CTranslate2==4.5.0 && \
    rm -Rf /root/.cache/pip

# Set the default command to execute main.py
CMD ["python3", "main.py"]
