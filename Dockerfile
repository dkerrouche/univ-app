# # Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# # you will also find guides on how best to write your Dockerfile

# FROM python:3.9.7

# RUN useradd -m -u 1000 user
# USER user
# ENV PATH="/home/user/.local/bin:$PATH"

# WORKDIR /app

# COPY --chown=user ./requirements.txt requirements.txt
# RUN pip install --no-cache-dir --upgrade -r requirements.txt

# COPY --chown=user . /app

# # RUN python ingest.py

# CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7860"]


FROM python:3.9.7

# Create a non-root user and set up environment
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy dependencies and install them
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the app code
COPY --chown=user . /app

# Expose the port (for local debugging and deployment)
# EXPOSE 7860

# # Dynamic port assignment for Hugging Face
# CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "${PORT}"]
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "7865"]
