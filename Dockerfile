FROM registry.deez.re/research/python-gpu-11-2:3.9

# Install dependencies with Poetry
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_CACHE_DIR=/var/cache
RUN mkdir -p /var/cache

WORKDIR /workspace
COPY pyproject.toml ./
COPY poetry.lock ./#
RUN poetry install --only main --no-root
RUN python -m spacy download en_core_web_sm

# -------------------------------
# Jupyter setup
EXPOSE 8888
RUN poetry install --only notebook --no-root
#ENTRYPOINT ["python", "main.py", "--experiment=all", "--min_utterances_for_query=1"]
#ENTRYPOINT ["python", "process_results.py", "--name=_queryminsize.5"]
#USER deezer