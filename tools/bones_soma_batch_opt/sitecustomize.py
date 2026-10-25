"""Explicit per-attempt opt-in; a requested bootstrap failure closes the worker."""
import os
import sys

if os.environ.get('UMR_EXECUTION_OPT_MANIFEST'):
    try:
        import execution_bootstrap
        execution_bootstrap.activate()
    except BaseException as error:
        print('[EXECUTION_OPT] bootstrap failed: '+repr(error), file=sys.stderr, flush=True)
        os._exit(79)
