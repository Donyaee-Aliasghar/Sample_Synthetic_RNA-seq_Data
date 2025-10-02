"""Main module for running the program and measuring its execution time."""

import time

from .runner import runner


def main():
    """Main function with time package."""
    start = time.time()
    runner()
    end = time.time()

    print("-" * 40)

    total = end - start
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = int(total % 60)

    # Show time process with leading zeros.
    print(f"[⏱️ ] Total execution time >>> {h:02d}:{m:02d}:{s:02d} ({total:.10f} s) <<<")


if __name__ == "__main__":
    main()
