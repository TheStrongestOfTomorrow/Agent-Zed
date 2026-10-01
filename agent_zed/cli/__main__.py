"""Entrypoint for python3 -m agent_zed.cli"""

import sys
import asyncio
from agent_zed.cli.interface import ZedCLI
from agent_zed.benchmark.runner import run_benchmark_suite

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        print("Starting Agent-Zed HARD Coding Benchmark Suite...")
        pass_rate, total = run_benchmark_suite()
        if pass_rate == 100.0:
            print(f"BENCHMARK PASSED 100% ({total}/{total} tests)")
            sys.exit(0)
        else:
            print(f"BENCHMARK FAILED ({pass_rate}% pass rate)")
            sys.exit(1)
    else:
        cli = ZedCLI()
        asyncio.run(cli.ceo_chat_loop())

if __name__ == "__main__":
    main()
