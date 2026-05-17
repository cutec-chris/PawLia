FROM python:3.12-slim

RUN pip install --no-cache-dir radicale icalendar vobject PyYAML bcrypt passlib

# Install our custom storage backend where Python can find it
COPY picoclaw_storage.py /usr/local/lib/python3.12/dist-packages/picoclaw_storage.py

COPY config.cfg /radicale/config.cfg
# users file is mounted at /radicale/users at runtime

EXPOSE 5232

CMD ["python", "-m", "radicale", "--config", "/radicale/config.cfg"]
