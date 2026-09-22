from __future__ import annotations

import argparse
import json

from dr_detection.gradcam import make_gradcam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/efficientnet_b0.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="artifacts/gradcam.png")
    parser.add_argument("--class-id", type=int, default=None)
    args = parser.parse_args()

    result = make_gradcam(args.config, args.checkpoint, args.image, args.output, args.class_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
