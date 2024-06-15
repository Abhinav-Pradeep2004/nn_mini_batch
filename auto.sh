#!/bin/bash

python3 /app/src/train_pipeline.py

touch /var/log/model_logs.log

chmod +x /var/log/model_logs.log

tail -f /var/log/model_logs.log
