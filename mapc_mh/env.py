"""
CPU-only JAX environment setup. Must be imported before any JAX import.
All mapc_mh modules import this first.
"""
import os

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
