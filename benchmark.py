"""
benchmark.py

Measures the average prediction latency of the complete prediction pipeline.

Usage:
    python benchmark.py image.jpg
"""

import sys
import time
import statistics

from predict import predict

NUM_RUNS = 100


def main():
    if len(sys.argv) != 2:
        print("Usage: python benchmark.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Warm-up (ignore first run)
    predict(image_path)

    times = []

    for _ in range(NUM_RUNS):
        start = time.perf_counter()

        predict(image_path)

        end = time.perf_counter()

        times.append((end - start) * 1000)

    print("=" * 40)
    print("Prediction Benchmark")
    print("=" * 40)
    print(f"Runs               : {NUM_RUNS}")
    print(f"Average Latency    : {statistics.mean(times):.2f} ms")
    print(f"Minimum Latency    : {min(times):.2f} ms")
    print(f"Maximum Latency    : {max(times):.2f} ms")
    print(f"Std Deviation      : {statistics.stdev(times):.2f} ms")


if __name__ == "__main__":
    main()