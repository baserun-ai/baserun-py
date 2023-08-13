import argparse
import runpy
from baserun import Baserun


def main():
    parser = argparse.ArgumentParser(description='Baserun CLI tool')
    parser.add_argument('--api-url', default='https://baserun.ai/api/v1', help='Baserun API URL')
    parser.add_argument('module', help='Python module to run')
    args = parser.parse_args()

    Baserun.init(api_url=args.api_url)

    try:
        runpy.run_module(args.module, run_name="__main__")
    finally:
        Baserun.flush()


if __name__ == "__main__":
    main()
