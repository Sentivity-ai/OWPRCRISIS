#!/bin/bash
set -e

echo "=== Installing packages ==="
pip install --no-cache-dir Flask==3.0.0 flask-cors==4.0.0 gunicorn==21.2.0
pip install --no-cache-dir praw==7.7.1
pip install --no-cache-dir "pandas<2.1" "numpy<1.25"
pip install --no-cache-dir nltk==3.8.1
pip install --no-cache-dir openai==1.12.0

echo "=== Verifying installations ==="
python -c "import flask; print('Flask OK')"
python -c "import nltk; print('NLTK OK')"
python -c "import openai; print('OpenAI OK')"

echo "=== Build complete ==="
