#!/usr/bin/env python3
"""Find visually similar/publicly indexed pages for an image using Google Vision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from google.cloud import vision


def detect_web(image_path: Path) -> dict[str, object]:
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_path.read_bytes())
    response = client.web_detection(image=image)
    if response.error.message:
        raise RuntimeError(response.error.message)

    result = response.web_detection
    return {
        "full_matches": [
            {"url": item.url, "score": item.score} for item in result.full_matching_images
        ],
        "partial_matches": [
            {"url": item.url, "score": item.score} for item in result.partial_matching_images
        ],
        "similar_images": [
            {"url": item.url, "score": item.score} for item in result.visually_similar_images
        ],
        "pages": [
            {"url": item.url, "title": item.page_title} for item in result.pages_with_matching_images
        ],
        "entities": [
            {"description": item.description, "score": item.score}
            for item in result.web_entities
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Google-indexed web content for matches to an image."
    )
    parser.add_argument("image", type=Path, help="Path to a local image file")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"image file does not exist: {args.image}")
    print(json.dumps(detect_web(args.image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
