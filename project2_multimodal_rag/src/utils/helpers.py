import os
import hashlib
import uuid
from datetime import datetime


def generate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def generate_unique_id():
    return str(uuid.uuid4())


def format_timestamp(dt=None):
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def get_file_extension(filename):
    return filename.split(".")[-1].lower() if "." in filename else ""


def ensure_dir_exists(path):
    os.makedirs(path, exist_ok=True)


def truncate_text(text, max_length=500):
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."