# SPDX-License-Identifier: GPL-3.0-or-later
# This Blender API component is covered by this directory's LICENSE.
"""Query original Atrium geometry on CPU without rendering.

blender --background --factory-startup --disable-autoexec --python query_rays.py \
    -- CAPTURE_DIR --output rays.json

Use the original saved scene from the capture. The output is a new file, and
includes hashes tying each query to the scene, manifest and this script.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Vector


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+")
    parser.add_argument("--arms", choices=["closed", "open"], nargs="+")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not bpy.app.background:
        parser.error("Run this query in a separate background Blender process")
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output must be a new file")
    if args.frames and (len(set(args.frames)) != len(args.frames) or any(not 0 <= f <= 65 for f in args.frames)):
        parser.error("Frame indices must be unique integers from 0 through 65")
    root = args.capture_dir.resolve()
    manifest_path, blend_path = root / "manifest.json", root / "original-scene.blend"
    if not blend_path.resolve().is_relative_to(root) or not manifest_path.resolve().is_relative_to(root):
        parser.error("Scene and manifest must be inside the capture directory")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "worldline-atrium-pilot-v1" or manifest.get("status") != "complete":
        parser.error("Expected a completed Atrium v1 capture")
    if sha256(blend_path) != manifest.get("blend_sha256"):
        parser.error("Saved scene hash does not match manifest")
    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=False, use_scripts=False)
    scene = bpy.context.scene
    camera = scene.camera
    if camera is None or camera.data.type != "PERSP":
        parser.error("Expected the original perspective camera")
    width, height = manifest["resolution"]
    actual_resolution = [scene.render.resolution_x * scene.render.resolution_percentage // 100,
                         scene.render.resolution_y * scene.render.resolution_percentage // 100]
    if actual_resolution != [width, height]:
        parser.error("Native scene resolution disagrees with manifest")
    corners = camera.data.view_frame(scene=scene)
    left, right = min(v.x for v in corners), max(v.x for v in corners)
    bottom, top = min(v.y for v in corners), max(v.y for v in corners)
    plane_z = corners[0].z
    # Derive K from Blender's own projection bounds, independently of manifest K.
    native_k = [[-plane_z * width / (right - left), 0., -left * width / (right - left)],
                [0., -plane_z * height / (top - bottom), top * height / (top - bottom)], [0., 0., 1.]]
    conversion = Matrix.Diagonal((1, -1, -1, 1))
    result = {"schema": "worldline-atrium-raycast-v1", "blend_sha256": sha256(blend_path),
              "manifest_sha256": sha256(manifest_path), "query_sha256": sha256(__file__),
              "query_license": "GPL-3.0-or-later", "blender": bpy.app.version_string,
              "method": "CPU scene.ray_cast through actual camera.view_frame at pixel centers; no rendering",
              "native_camera_K": native_k,
              "native_camera_model": {"type": "PERSPECTIVE", "lens_mm": camera.data.lens,
                  "sensor_width_mm": camera.data.sensor_width, "sensor_fit": camera.data.sensor_fit,
                  "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y]},
              "camera_view_frame_local": [list(v) for v in corners], "queried_frames": [], "samples": []}
    for arm, records in manifest["arms"].items():
        if args.arms and arm not in args.arms:
            continue
        for record in records:
            if "exr" not in record or (args.frames and record["frame"] not in args.frames):
                continue
            camera.matrix_world = Matrix(record["world_from_camera"]) @ conversion
            bpy.data.objects["Door hinge controlled by action"].rotation_euler.z = math.radians(102) if record["door_open"] else 0
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            origin = camera.matrix_world.translation.copy()
            queried = {"arm": arm, "frame": record["frame"], "ray_count": 0, "hit_count": 0}
            for fy in (.1, .25, .4, .6, .75, .9):
                for fx in (.05, .15, .25, .35, .65, .75, .85, .95):
                    x, y = int(fx * width), int(fy * height)
                    local = Vector((left + (right - left) * (x + .5) / width,
                                    top - (top - bottom) * (y + .5) / height, plane_z)).normalized()
                    direction = (camera.matrix_world.to_3x3() @ local).normalized()
                    hit, position, normal, face, obj, matrix = scene.ray_cast(depsgraph, origin, direction, distance=camera.data.clip_end)
                    queried["ray_count"] += 1
                    if not hit:
                        continue
                    queried["hit_count"] += 1
                    delta = position - origin
                    axial = (Matrix(record["world_from_camera"]).to_3x3().transposed() @ delta).z
                    result["samples"].append({"arm": arm, "frame": record["frame"], "pixel_xy": [x, y],
                        "euclidean_range": delta.length, "axial_z": axial, "object_id": obj.pass_index,
                        "object_name": obj.name, "world_hit": list(position)})
            result["queried_frames"].append(queried)
    if not result["samples"]:
        parser.error("Selection produced no foreground ray hits")
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"queried_frames": len(result["queried_frames"]), "foreground_ray_samples": len(result["samples"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
