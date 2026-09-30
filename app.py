"""Launch live dictation. Original record/stop mode is in record_app.py."""
import faulthandler

# Emit Python stacks for SIGBUS/SIGSEGV instead of leaving the browser with
# only a disconnected WebSocket. This is especially important for CUDA-native
# failures, which cannot be caught as regular Python exceptions.
faulthandler.enable(all_threads=True)

from live_app import main

if __name__ == '__main__':
    main()
