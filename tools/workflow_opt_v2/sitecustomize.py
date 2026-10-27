"""Explicit, fail-closed startup for the optional second execution profile."""
import os
import sys

if os.environ.get('UMR_WORKFLOW_OPT_MANIFEST'):
    try:
        import workflow_bootstrap
        workflow_bootstrap.activate()
    except BaseException as error:
        print('[WORKFLOW_OPT_V2] bootstrap failed: '+repr(error), file=sys.stderr, flush=True)
        os._exit(80)
elif os.environ.get('UMR_EXECUTION_OPT_MANIFEST'):
    # Preserve the existing profile if this directory was left on PYTHONPATH.
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'bones_soma_batch_opt'))
    try:
        import execution_bootstrap
        execution_bootstrap.activate()
    except BaseException as error:
        print('[EXECUTION_OPT] bootstrap failed: '+repr(error), file=sys.stderr, flush=True)
        os._exit(79)
