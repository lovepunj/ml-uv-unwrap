#!/bin/bash
cd "$(dirname "$0")"
echo "Starting ML UV Unwrap server on http://localhost:8000"
echo "Press Ctrl+C to stop"
python3 -m uvicorn web.server:app --host 0.0.0.0 --port 8000 --reload
