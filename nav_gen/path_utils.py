import hashlib
import re


def make_safe_task_dir_name(task_instruction, max_length=120):
    cleaned = re.sub(r"\s+", " ", str(task_instruction).strip())
    cleaned = cleaned.replace("/", "_").replace("\\", "_")
    cleaned = cleaned.rstrip(". ")
    if not cleaned:
        cleaned = "task"

    if len(cleaned) <= max_length:
        return cleaned

    digest = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
    prefix_length = max_length - len("__") - len(digest)
    prefix = cleaned[:prefix_length].rstrip(" ._")
    if not prefix:
        prefix = "task"
    return f"{prefix}__{digest}"
