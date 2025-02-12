import logging
import sys

log = logging.getLogger("hammer")
log.setLevel(logging.INFO)
log_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setFormatter(log_format)
log.addHandler(console_handler)
