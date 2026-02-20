"""
Mock Girder server for local GPU worker testing.

Impersonates the Girder/large_image REST API so that worker Docker containers
can run locally without a real Girder instance.  Serves test images from a
local directory and captures all annotations/property values workers produce.

Usage:
    python server.py --images-dir ../fixtures/images --port 5555

The server expects the images directory to contain at least one TIFF file.
It treats the first TIFF found as the dataset image.
"""

import argparse
import io
import json
import os
import pickle
import sys
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request

try:
    import tifffile
except ImportError:
    print("tifffile is required: pip install tifffile")
    sys.exit(1)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state populated at startup
# ---------------------------------------------------------------------------
IMAGE_DIR = None        # Path to fixtures/images/
IMAGES = {}             # {filename: numpy array}
TILE_METADATA = {}      # {dataset_id: tiles metadata dict}
FRAMES = {}             # {dataset_id: [frame numpy arrays]}
DATASET_ITEM_ID = None  # The "large image" item id
FOLDER_ID = None        # The dataset folder id (what workers pass as datasetId)

# Captured outputs
CREATED_ANNOTATIONS = []
CREATED_CONNECTIONS = []
PROPERTY_VALUES = []
WORKER_INTERFACES = {}
OUTPUT_DIR = None


def load_images(images_dir):
    """Load all TIFF files from the images directory."""
    global IMAGE_DIR, IMAGES, TILE_METADATA, FRAMES, DATASET_ITEM_ID, FOLDER_ID

    IMAGE_DIR = Path(images_dir)
    tiff_files = sorted(IMAGE_DIR.glob("*.tif")) + sorted(IMAGE_DIR.glob("*.tiff"))

    if not tiff_files:
        print(f"WARNING: No TIFF files found in {images_dir}")
        print("Generating a synthetic test image...")
        generate_synthetic_image(images_dir)
        tiff_files = sorted(IMAGE_DIR.glob("*.tif")) + sorted(IMAGE_DIR.glob("*.tiff"))

    for tiff_path in tiff_files:
        print(f"Loading: {tiff_path.name}")
        img = tifffile.imread(str(tiff_path))
        IMAGES[tiff_path.name] = img

    # Use the first image as the dataset
    first_name = list(IMAGES.keys())[0]
    first_img = IMAGES[first_name]

    # Generate IDs
    FOLDER_ID = "test_dataset"
    DATASET_ITEM_ID = "test_item_" + uuid.uuid4().hex[:8]

    # Build tile metadata from image shape
    # Supported shapes: (H, W), (C, H, W), (T, C, H, W), (Z, T, C, H, W), etc.
    if first_img.ndim == 2:
        height, width = first_img.shape
        num_channels = 1
        num_z = 1
        num_t = 1
        num_xy = 1
    elif first_img.ndim == 3:
        num_channels, height, width = first_img.shape
        num_z = 1
        num_t = 1
        num_xy = 1
    elif first_img.ndim == 4:
        num_z, num_channels, height, width = first_img.shape
        num_t = 1
        num_xy = 1
    elif first_img.ndim == 5:
        num_t, num_z, num_channels, height, width = first_img.shape
        num_xy = 1
    else:
        print(f"WARNING: Unexpected image dimensions: {first_img.ndim}")
        height, width = first_img.shape[-2], first_img.shape[-1]
        num_channels = 1
        num_z = 1
        num_t = 1
        num_xy = 1

    # Build frames list
    frames = []
    frame_arrays = []
    frame_idx = 0
    for t in range(num_t):
        for z in range(num_z):
            for c in range(num_channels):
                for xy in range(num_xy):
                    frame_info = {
                        "Frame": frame_idx,
                        "IndexC": c,
                        "IndexZ": z,
                        "IndexT": t,
                        "IndexXY": xy,
                    }
                    frames.append(frame_info)

                    # Extract the frame data
                    if first_img.ndim == 2:
                        frame_data = first_img
                    elif first_img.ndim == 3:
                        frame_data = first_img[c]
                    elif first_img.ndim == 4:
                        frame_data = first_img[z, c]
                    elif first_img.ndim == 5:
                        frame_data = first_img[t, z, c]
                    else:
                        frame_data = first_img

                    frame_arrays.append(frame_data)
                    frame_idx += 1

    channel_names = [f"Channel_{i}" for i in range(num_channels)]

    tiles_meta = {
        "tileWidth": width,
        "tileHeight": height,
        "sizeX": width,
        "sizeY": height,
        "levels": 1,
        "magnification": 20,
        "mm_x": 0.000325,
        "mm_y": 0.000325,
        "dtype": str(first_img.dtype),
        "frames": frames,
        "channels": channel_names,
        "IndexRange": {
            "IndexC": num_channels,
            "IndexZ": num_z,
            "IndexT": num_t,
            "IndexXY": num_xy,
        },
        "IndexStride": {
            "IndexC": 1,
            "IndexZ": num_channels,
            "IndexT": num_channels * num_z,
            "IndexXY": num_channels * num_z * num_t,
        },
    }

    TILE_METADATA[DATASET_ITEM_ID] = tiles_meta
    FRAMES[DATASET_ITEM_ID] = frame_arrays

    print(f"Loaded image: {first_name} shape={first_img.shape} dtype={first_img.dtype}")
    print(f"  Channels={num_channels}, Z={num_z}, T={num_t}, XY={num_xy}")
    print(f"  Width={width}, Height={height}")
    print(f"  Folder ID (datasetId for workers): {FOLDER_ID}")
    print(f"  Item ID (large image): {DATASET_ITEM_ID}")
    print(f"  Total frames: {len(frames)}")


def generate_synthetic_image(images_dir):
    """Generate a small synthetic multi-channel test image."""
    path = Path(images_dir)
    path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    # 3-channel, 512x512 image with some blobs
    height, width = 512, 512
    channels = []
    for c in range(3):
        img = np.zeros((height, width), dtype=np.uint16)
        # Add some random circular blobs
        for _ in range(20):
            cy, cx = rng.integers(50, height - 50), rng.integers(50, width - 50)
            radius = rng.integers(10, 40)
            yy, xx = np.ogrid[-cy:height - cy, -cx:width - cx]
            mask = xx**2 + yy**2 <= radius**2
            intensity = rng.integers(500, 4000)
            img[mask] = np.maximum(img[mask], intensity)
        # Add background noise
        img = img + rng.integers(0, 200, (height, width), dtype=np.uint16)
        channels.append(img)

    multi_channel = np.stack(channels, axis=0)  # (3, 512, 512)
    out_path = path / "synthetic_cells.tiff"
    tifffile.imwrite(str(out_path), multi_channel)
    print(f"Generated synthetic image: {out_path} shape={multi_channel.shape}")


# ---------------------------------------------------------------------------
# Girder API: Dataset/folder endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/folder/<dataset_id>", methods=["GET"])
def get_dataset_folder(dataset_id):
    """GET /folder/{datasetId} - Return folder metadata."""
    return jsonify({
        "_id": dataset_id,
        "name": "Test Dataset",
        "meta": {
            "selectedLargeImageId": DATASET_ITEM_ID,
        },
    })


@app.route("/api/v1/item", methods=["GET"])
def list_items():
    """GET /item?folderId={datasetId}&limit=0 - List items in folder."""
    return jsonify([
        {
            "_id": DATASET_ITEM_ID,
            "name": "test_image.tiff",
            "folderId": FOLDER_ID,
            "largeImage": {"fileId": "fake_file_id"},
        }
    ])


# ---------------------------------------------------------------------------
# Girder API: Tiles/image endpoints (large_image)
# ---------------------------------------------------------------------------

@app.route("/api/v1/item/<item_id>/tiles", methods=["GET"])
def get_tiles(item_id):
    """GET /item/{id}/tiles - Tile metadata."""
    meta = TILE_METADATA.get(item_id)
    if meta is None:
        return jsonify({"error": f"Unknown item: {item_id}"}), 404
    return jsonify(meta)


@app.route("/api/v1/item/<item_id>/tiles/internal_metadata", methods=["GET"])
def get_tiles_internal(item_id):
    """GET /item/{id}/tiles/internal_metadata - Internal tile metadata."""
    return jsonify({"internal": True, "itemId": item_id})


@app.route("/api/v1/item/<item_id>/tiles/region", methods=["GET"])
def get_tiles_region(item_id):
    """GET /item/{id}/tiles/region - Return image data.

    The real large_image endpoint supports many parameters.
    We support: frame, encoding, left, top, right, bottom.
    """
    frame_idx = int(request.args.get("frame", 0))
    encoding = request.args.get("encoding", "pickle:5")

    frame_list = FRAMES.get(item_id, [])
    if frame_idx >= len(frame_list):
        return jsonify({"error": f"Frame {frame_idx} out of range"}), 404

    frame_data = frame_list[frame_idx]

    # Handle subregion requests
    left = request.args.get("left")
    top = request.args.get("top")
    right = request.args.get("right")
    bottom = request.args.get("bottom")

    if left is not None and top is not None and right is not None and bottom is not None:
        left, top, right, bottom = int(float(left)), int(float(top)), int(float(right)), int(float(bottom))
        frame_data = frame_data[top:bottom, left:right]

    # Return in requested encoding
    if encoding.startswith("pickle"):
        buf = pickle.dumps(frame_data)
        return Response(buf, mimetype="application/octet-stream")
    elif encoding == "TIFF":
        buf = io.BytesIO()
        tifffile.imwrite(buf, frame_data)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/tiff")
    else:
        # Default to pickle
        buf = pickle.dumps(frame_data)
        return Response(buf, mimetype="application/octet-stream")


@app.route("/api/v1/item/<item_id>/tiles/fzxy/<int:frame_idx>/0/0/0", methods=["GET"])
def get_raw_image(item_id, frame_idx):
    """GET /item/{id}/tiles/fzxy/{frame}/0/0/0 - Raw image as PNG."""
    frame_list = FRAMES.get(item_id, [])
    if frame_idx >= len(frame_list):
        return jsonify({"error": f"Frame {frame_idx} out of range"}), 404

    frame_data = frame_list[frame_idx]
    buf = io.BytesIO()
    # Return as TIFF since PNG can't handle 16-bit well
    tifffile.imwrite(buf, frame_data)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/tiff")


# ---------------------------------------------------------------------------
# Girder API: Annotation endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/upenn_annotation", methods=["GET"])
def get_annotations():
    """GET /upenn_annotation?datasetId=... - List annotations."""
    dataset_id = request.args.get("datasetId")
    shape = request.args.get("shape")

    result = CREATED_ANNOTATIONS
    if dataset_id:
        result = [a for a in result if a.get("datasetId") == dataset_id]
    if shape:
        result = [a for a in result if a.get("shape") == shape]

    return jsonify(result)


@app.route("/api/v1/upenn_annotation/", methods=["POST"])
def create_annotation():
    """POST /upenn_annotation/ - Create a single annotation."""
    data = request.get_json()
    data["_id"] = "ann_" + uuid.uuid4().hex[:12]
    CREATED_ANNOTATIONS.append(data)
    return jsonify(data)


@app.route("/api/v1/upenn_annotation/multiple", methods=["POST"])
def create_multiple_annotations():
    """POST /upenn_annotation/multiple - Create multiple annotations."""
    data = request.get_json()
    results = []
    for ann in data:
        ann["_id"] = "ann_" + uuid.uuid4().hex[:12]
        CREATED_ANNOTATIONS.append(ann)
        results.append(ann)
    print(f"Created {len(results)} annotations (total: {len(CREATED_ANNOTATIONS)})")
    return jsonify(results)


@app.route("/api/v1/upenn_annotation/multiple", methods=["DELETE"])
def delete_multiple_annotations():
    """DELETE /upenn_annotation/multiple - Delete multiple annotations."""
    data = request.get_json()
    global CREATED_ANNOTATIONS
    CREATED_ANNOTATIONS = [a for a in CREATED_ANNOTATIONS if a["_id"] not in data]
    return jsonify({"deleted": len(data)})


@app.route("/api/v1/upenn_annotation/count", methods=["GET"])
def count_annotations():
    """GET /upenn_annotation/count?datasetId=... - Count annotations."""
    return jsonify({"count": len(CREATED_ANNOTATIONS)})


@app.route("/api/v1/upenn_annotation/<annotation_id>", methods=["GET"])
def get_annotation_by_id(annotation_id):
    """GET /upenn_annotation/{id} - Get annotation by id."""
    for ann in CREATED_ANNOTATIONS:
        if ann["_id"] == annotation_id:
            return jsonify(ann)
    return jsonify({"error": "Not found"}), 404


# ---------------------------------------------------------------------------
# Girder API: Connection endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/annotation_connection/", methods=["GET", "POST"])
def connections():
    """GET/POST /annotation_connection/ - List or create connections."""
    if request.method == "POST":
        data = request.get_json()
        data["_id"] = "conn_" + uuid.uuid4().hex[:12]
        CREATED_CONNECTIONS.append(data)
        return jsonify(data)
    return jsonify(CREATED_CONNECTIONS)


@app.route("/api/v1/annotation_connection/multiple", methods=["POST"])
def create_multiple_connections():
    """POST /annotation_connection/multiple - Create multiple connections."""
    data = request.get_json()
    results = []
    for conn in data:
        conn["_id"] = "conn_" + uuid.uuid4().hex[:12]
        CREATED_CONNECTIONS.append(conn)
        results.append(conn)
    return jsonify(results)


@app.route("/api/v1/annotation_connection/connectTo/", methods=["POST"])
def connect_to_nearest():
    """POST /annotation_connection/connectTo/ - Connect to nearest."""
    data = request.get_json()
    print(f"connectToNearest called: {json.dumps(data, indent=2)}")
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Girder API: Property endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/annotation_property_values", methods=["GET", "POST"])
def property_values():
    """GET/POST /annotation_property_values - Property values."""
    if request.method == "POST":
        data = request.get_json()
        PROPERTY_VALUES.append(data)
        print(f"Received property values for annotation")
        return jsonify({"status": "ok"})
    return jsonify(PROPERTY_VALUES)


@app.route("/api/v1/annotation_property_values/multiple", methods=["POST"])
def add_multiple_property_values():
    """POST /annotation_property_values/multiple - Batch property values."""
    data = request.get_json()
    PROPERTY_VALUES.extend(data)
    print(f"Received {len(data)} property value entries (total: {len(PROPERTY_VALUES)})")
    return jsonify({"status": "ok"})


@app.route("/api/v1/annotation_property", methods=["GET", "POST"])
def properties():
    """GET/POST /annotation_property - Property definitions."""
    if request.method == "POST":
        data = request.get_json()
        data["_id"] = "prop_" + uuid.uuid4().hex[:12]
        return jsonify(data)
    return jsonify([])


# ---------------------------------------------------------------------------
# Girder API: Worker interface endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/worker_interface", methods=["POST"])
def set_worker_interface():
    """POST /worker_interface?image=... - Set worker interface."""
    image = request.args.get("image", "unknown")
    data = request.get_json()
    WORKER_INTERFACES[image] = data
    print(f"Worker interface set for image: {image}")
    return jsonify({"_id": "iface_" + uuid.uuid4().hex[:8]})


@app.route("/api/v1/worker_preview", methods=["POST"])
def set_worker_preview():
    """POST /worker_preview?image=... - Set worker preview."""
    image = request.args.get("image", "unknown")
    data = request.get_json()
    print(f"Worker preview set for image: {image}")
    return jsonify({"_id": "preview_" + uuid.uuid4().hex[:8]})


# ---------------------------------------------------------------------------
# Girder API: File upload (for image processing workers)
# ---------------------------------------------------------------------------

@app.route("/api/v1/file", methods=["POST"])
def upload_file_init():
    """POST /file - Initialize file upload."""
    return jsonify({
        "_id": "file_" + uuid.uuid4().hex[:8],
        "size": 0,
    })


@app.route("/api/v1/file/chunk", methods=["POST"])
def upload_file_chunk():
    """POST /file/chunk - Upload file chunk."""
    return jsonify({"_id": "file_" + uuid.uuid4().hex[:8]})


@app.route("/api/v1/item/<item_id>", methods=["GET", "PUT"])
def item_by_id(item_id):
    """GET/PUT /item/{id} - Get or update item."""
    if request.method == "PUT":
        metadata = request.args.get("metadata")
        if metadata:
            print(f"Item {item_id} metadata updated: {metadata}")
        return jsonify({"_id": item_id, "itemId": item_id})
    return jsonify({"_id": item_id, "name": "test_item"})


# ---------------------------------------------------------------------------
# Dataset view endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/dataset_view", methods=["GET"])
def dataset_views():
    """GET /dataset_view?datasetId=... - Dataset views."""
    return jsonify([])


# ---------------------------------------------------------------------------
# Status & output endpoints (for inspection)
# ---------------------------------------------------------------------------

@app.route("/api/v1/_test/status", methods=["GET"])
def test_status():
    """Custom endpoint: see what the mock has captured."""
    return jsonify({
        "annotations_count": len(CREATED_ANNOTATIONS),
        "connections_count": len(CREATED_CONNECTIONS),
        "property_values_count": len(PROPERTY_VALUES),
        "worker_interfaces": list(WORKER_INTERFACES.keys()),
    })


@app.route("/api/v1/_test/annotations", methods=["GET"])
def test_annotations():
    """Custom endpoint: get all captured annotations."""
    return jsonify(CREATED_ANNOTATIONS)


@app.route("/api/v1/_test/property_values", methods=["GET"])
def test_property_values():
    """Custom endpoint: get all captured property values."""
    return jsonify(PROPERTY_VALUES)


@app.route("/api/v1/_test/reset", methods=["POST"])
def test_reset():
    """Custom endpoint: clear all captured data."""
    global CREATED_ANNOTATIONS, CREATED_CONNECTIONS, PROPERTY_VALUES
    CREATED_ANNOTATIONS = []
    CREATED_CONNECTIONS = []
    PROPERTY_VALUES = []
    return jsonify({"status": "reset"})


@app.route("/api/v1/_test/save", methods=["POST"])
def test_save():
    """Custom endpoint: save all captured data to JSON files."""
    if OUTPUT_DIR:
        out = Path(OUTPUT_DIR)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "annotations.json", "w") as f:
            json.dump(CREATED_ANNOTATIONS, f, indent=2)
        with open(out / "property_values.json", "w") as f:
            json.dump(PROPERTY_VALUES, f, indent=2)
        with open(out / "connections.json", "w") as f:
            json.dump(CREATED_CONNECTIONS, f, indent=2)
        print(f"Saved outputs to {out}/")
    return jsonify({"status": "saved"})


# ---------------------------------------------------------------------------
# Girder API: Catch-all for unimplemented endpoints
# ---------------------------------------------------------------------------

@app.route("/api/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def catch_all(path):
    """Catch-all for unimplemented endpoints — logs and returns empty."""
    print(f"UNHANDLED: {request.method} /api/v1/{path}")
    print(f"  Args: {dict(request.args)}")
    if request.is_json:
        print(f"  Body: {json.dumps(request.get_json(), indent=2)[:500]}")
    return jsonify({})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mock Girder server for local worker testing")
    parser.add_argument("--images-dir", type=str, default="tests/fixtures/images",
                        help="Directory containing test TIFF images")
    parser.add_argument("--port", type=int, default=5555,
                        help="Port to run the server on")
    parser.add_argument("--output-dir", type=str, default="tests/fixtures/output",
                        help="Directory to save captured annotations/properties")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir

    load_images(args.images_dir)

    print(f"\n{'='*60}")
    print(f"Mock Girder server running on http://localhost:{args.port}")
    print(f"  Images from: {args.images_dir}")
    print(f"  Output to:   {args.output_dir}")
    print(f"  Dataset ID (use as --datasetId): {FOLDER_ID}")
    print(f"{'='*60}\n")

    app.run(host="0.0.0.0", port=args.port, debug=False)


if __name__ == "__main__":
    main()
