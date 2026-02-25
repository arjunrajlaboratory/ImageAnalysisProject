#!/usr/bin/env python3
"""
Test connection to NimbusImage API.

Validates image loading, annotation retrieval, and multi-channel merging
against a live Nimbus instance. Run after setup_env.sh creates the venv.

Usage:
    # With env vars:
    export NIMBUS_API_URL=http://localhost:8080/api/v1
    export NIMBUS_TOKEN=your_token_here
    python test_connection.py

    # With interactive login (will prompt for username/password):
    python test_connection.py

    # With a specific dataset:
    python test_connection.py --dataset 69988c84b48d8121b565aba4
"""

import argparse
import getpass
import os
import sys
from collections import Counter

import numpy as np

import annotation_client.annotations as annotations_client
import annotation_client.tiles as tiles
import annotation_utilities.annotation_tools as annotation_tools


def ensure_rgb(image):
    """Ensure image is (H, W, 3) uint8 RGB. Copied from entrypoint.py to avoid sam2 import."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0 and image.min() >= 0.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    elif image.dtype == np.uint16:
        image = (image / 256).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    return image


def get_auth(args):
    """Get API URL and token from env vars, CLI args, or interactive login."""
    api_url = args.api_url or os.environ.get('NIMBUS_API_URL', 'http://localhost:8080/api/v1')
    token = args.token or os.environ.get('NIMBUS_TOKEN')

    if token:
        print(f"API URL: {api_url}")
        print(f"Token:   {token[:8]}...")
        return api_url, token

    # Login via girder_client (CLI args or interactive)
    import girder_client
    gc = girder_client.GirderClient(apiUrl=api_url)

    username = args.username or os.environ.get('NIMBUS_USERNAME') or input("Username: ")
    password = args.password or os.environ.get('NIMBUS_PASSWORD') or getpass.getpass("Password: ")

    print(f"Logging in as '{username}' to {api_url}...")
    gc.authenticate(username, password)
    token = gc.token

    print(f"API URL: {api_url}")
    print(f"Token:   {token[:8]}...")
    return api_url, token


def test_image_loading(api_url, token, dataset_id):
    """Test image metadata and loading."""
    print("\n" + "=" * 60)
    print("IMAGE LOADING TEST")
    print("=" * 60)

    tileClient = tiles.UPennContrastDataset(
        apiUrl=api_url, token=token, datasetId=dataset_id
    )

    # Report metadata
    idx_range = tileClient.tiles.get('IndexRange', {})
    num_channels = idx_range.get('IndexC', 1)
    num_z = idx_range.get('IndexZ', 1)
    num_time = idx_range.get('IndexT', 1)
    num_xy = idx_range.get('IndexXY', 1)

    print(f"  Channels:   {num_channels}")
    print(f"  Z-planes:   {num_z}")
    print(f"  Timepoints: {num_time}")
    print(f"  XY pos:     {num_xy}")

    if 'channels' in tileClient.tiles:
        print(f"  Channel names: {tileClient.tiles['channels']}")

    size_x = tileClient.tiles.get('sizeX', 'unknown')
    size_y = tileClient.tiles.get('sizeY', 'unknown')
    print(f"  Image size: {size_x} x {size_y}")
    print(f"  mm_x: {tileClient.tiles.get('mm_x', 'N/A')}")
    print(f"  mm_y: {tileClient.tiles.get('mm_y', 'N/A')}")
    print(f"  Total frames: {len(tileClient.tiles.get('frames', []))}")

    # Load a single frame
    print("\n  Loading single frame (XY=0, Z=0, T=0, C=0)...")
    frame = tileClient.coordinatesToFrameIndex(0, Z=0, T=0, channel=0)
    image = tileClient.getRegion(dataset_id, frame=frame).squeeze()
    print(f"  Single frame: shape={image.shape}, dtype={image.dtype}")

    # Load merged RGB via the same pipeline as the SAM2 worker
    print("\n  Loading merged RGB image (same as SAM2 worker)...")
    images = annotation_tools.get_images_for_all_channels(tileClient, dataset_id, 0, 0, 0)
    print(f"  Loaded {len(images)} channel images")
    for i, img in enumerate(images):
        print(f"    Channel {i}: shape={img.shape}, dtype={img.dtype}, "
              f"range=[{img.min():.1f}, {img.max():.1f}]")

    layers = annotation_tools.get_layers(tileClient.client, dataset_id)
    print(f"  Found {len(layers)} layers")

    merged = annotation_tools.process_and_merge_channels(images, layers)
    print(f"  Merged image: shape={merged.shape}, dtype={merged.dtype}")

    rgb = ensure_rgb(merged)
    print(f"  RGB output:   shape={rgb.shape}, dtype={rgb.dtype}, "
          f"range=[{rgb.min()}, {rgb.max()}]")

    return tileClient


def test_annotation_loading(api_url, token, dataset_id):
    """Test annotation retrieval and analysis."""
    print("\n" + "=" * 60)
    print("ANNOTATION LOADING TEST")
    print("=" * 60)

    annotationClient = annotations_client.UPennContrastAnnotationClient(
        apiUrl=api_url, token=token
    )

    # Get all polygon annotations
    print("  Fetching polygon annotations...")
    polygons = annotationClient.getAnnotationsByDatasetId(dataset_id, shape='polygon')
    print(f"  Total polygons: {len(polygons)}")

    if len(polygons) == 0:
        print("  No polygon annotations found. Skipping detailed analysis.")
        return

    # Tag breakdown
    tag_counter = Counter()
    for ann in polygons:
        ann_tags = ann.get('tags', [])
        if ann_tags:
            for t in ann_tags:
                tag_counter[t] += 1
        else:
            tag_counter['(untagged)'] += 1

    print(f"\n  Tag breakdown ({len(tag_counter)} unique tags):")
    for tag, count in tag_counter.most_common():
        print(f"    {tag}: {count}")

    # Location distribution
    xy_counter = Counter()
    z_counter = Counter()
    time_counter = Counter()
    for ann in polygons:
        loc = ann.get('location', {})
        xy_counter[loc.get('XY', 0)] += 1
        z_counter[loc.get('Z', 0)] += 1
        time_counter[loc.get('Time', 0)] += 1

    print(f"\n  Location distribution:")
    print(f"    XY positions: {dict(xy_counter.most_common())}")
    print(f"    Z planes:     {dict(z_counter.most_common())}")
    print(f"    Timepoints:   {dict(time_counter.most_common())}")

    # Coordinate stats
    all_x = []
    all_y = []
    for ann in polygons:
        for coord in ann.get('coordinates', []):
            all_x.append(coord['x'])
            all_y.append(coord['y'])

    if all_x:
        print(f"\n  Coordinate ranges:")
        print(f"    X: [{min(all_x):.1f}, {max(all_x):.1f}]")
        print(f"    Y: [{min(all_y):.1f}, {max(all_y):.1f}]")

    # Also check for points and lines
    points = annotationClient.getAnnotationsByDatasetId(dataset_id, shape='point')
    lines = annotationClient.getAnnotationsByDatasetId(dataset_id, shape='line')
    print(f"\n  Other shapes: {len(points)} points, {len(lines)} lines")


def main():
    parser = argparse.ArgumentParser(description='Test NimbusImage API connection')
    parser.add_argument('--dataset', type=str,
                        default='69988c84b48d8121b565aba4',
                        help='Dataset ID to test with')
    parser.add_argument('--api-url', type=str, default=None,
                        help='API URL (default: $NIMBUS_API_URL or http://localhost:8080/api/v1)')
    parser.add_argument('--token', type=str, default=None,
                        help='Auth token (default: $NIMBUS_TOKEN or interactive login)')
    parser.add_argument('--username', type=str, default=None,
                        help='Username for login (default: $NIMBUS_USERNAME or interactive)')
    parser.add_argument('--password', type=str, default=None,
                        help='Password for login (default: $NIMBUS_PASSWORD or interactive)')
    args = parser.parse_args()

    print("=" * 60)
    print("NimbusImage API Connection Test")
    print("=" * 60)

    api_url, token = get_auth(args)
    dataset_id = args.dataset
    print(f"Dataset: {dataset_id}")

    try:
        test_image_loading(api_url, token, dataset_id)
    except Exception as e:
        print(f"\n  ERROR in image loading: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_annotation_loading(api_url, token, dataset_id)
    except Exception as e:
        print(f"\n  ERROR in annotation loading: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
