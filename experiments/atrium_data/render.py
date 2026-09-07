# SPDX-License-Identifier: GPL-3.0-or-later
# Original Worldline data-generation script. See LICENSE and DATA-LICENSE.
"""Render paired interaction data in a separate background Blender process.

blender --background --factory-startup --python render.py -- --output NEW_DIR
The scene and all materials are created here. No external assets are loaded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import bpy
from mathutils import Matrix, Vector


def material(name, color, roughness=.5, metallic=0, grain=0):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes, links = result.node_tree.nodes, result.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    if grain:
        texture = nodes.new("ShaderNodeTexNoise")
        texture.inputs["Scale"].default_value = grain
        texture.inputs["Detail"].default_value = 3
        ramp = nodes.new("ShaderNodeValToRGB")
        for item, factor in zip(ramp.color_ramp.elements, (.72, 1.13)):
            item.color = (*(min(v*factor, 1) for v in color), 1)
        links.new(texture.outputs["Fac"], ramp.inputs["Fac"])
        links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
        bump = nodes.new("ShaderNodeBump")
        bump.inputs["Strength"].default_value = .18
        bump.inputs["Distance"].default_value = .016
        links.new(texture.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], shader.inputs["Normal"])
    return result


def box(name, center, size, mat, bevel=.025, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    if bevel:
        modifier = obj.modifiers.new("Soft physical edges", "BEVEL")
        modifier.width, modifier.segments = bevel, 3
        obj.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
    if parent:
        matrix = obj.matrix_world.copy()
        obj.parent = parent
        obj.matrix_world = matrix
    return obj


def cylinder(name, center, radius, depth, mat):
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=radius, depth=depth, location=center)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    bevel = obj.modifiers.new("Rounded rim", "BEVEL")
    bevel.width, bevel.segments = .025, 3
    for poly in obj.data.polygons:
        poly.use_smooth = True
    return obj


def aim(obj, target):
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()


def plant(name, center, rng, leaf, ceramic, wood):
    x, y, z = center
    cylinder(name+" pot", (x, y, z+.26), .24, .52, ceramic)
    cylinder(name+" stem", (x, y, z+.95), .027, 1.4, wood)
    for index in range(34):
        angle = rng.uniform(0, math.tau)
        height = rng.uniform(.55, 1.7)
        radius = rng.uniform(.12, .44)*(1.2-height/2.3)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=12, ring_count=6,
            location=(x+radius*math.cos(angle), y+radius*math.sin(angle), z+height))
        obj = bpy.context.object
        obj.name = f"{name} leaf {index}"
        obj.scale = (.055, .22, .018)
        obj.rotation_euler = (rng.uniform(-.45, .45), rng.uniform(-.3, .3), angle)
        obj.data.materials.append(leaf)
        for poly in obj.data.polygons:
            poly.use_smooth = True


def build_scene(seed):
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    rng = random.Random(seed)
    plaster = material("Original warm plaster", (.72, .66, .55), .82, grain=42)
    stone = material("Original honed limestone", (.47, .43, .34), .34, grain=48)
    walnut = material("Original walnut", (.17, .067, .025), .36, grain=9)
    brass = material("Original brushed brass", (.53, .32, .10), .27, metallic=.86, grain=85)
    dark = material("Original dark window frames", (.029, .04, .035), .3, metallic=.65)
    leaf = material("Original leaves", (.057, .16, .045), .4)
    ceramic = material("Original terracotta", (.36, .12, .055), .48, grain=35)
    cushion = material("Original linen", (.49, .47, .36), .92, grain=180)
    accent = material("Original seeded accent", tuple(rng.uniform(.08, .38) for _ in range(3)), .26)
    # All dimensions are in meters. Room one y<0; room two y>0.
    for ix in range(9):
        for iy in range(14):
            box(f"Floor slab {ix} {iy}", (-4+ix, -4.5+iy, -.055), (.994, .994, .10), stone, .008)
    box("Right wall", (4.55, 2, 1.75), (.18, 14, 3.5), plaster)
    box("Back wall", (0, -5.05, 1.75), (9.2, .18, 3.5), plaster)
    box("Far wall", (0, 9.05, 1.75), (9.2, .18, 3.5), plaster)
    box("Partition left", (-2.85, 0, 1.75), (3.3, .18, 3.5), plaster)
    box("Partition right", (2.85, 0, 1.75), (3.3, .18, 3.5), plaster)
    box("Partition lintel", (0, 0, 3.17), (2.4, .18, .66), plaster)
    for x in (-1.24, 1.24):
        box("Door jamb", (x, -.035, 1.4), (.075, .28, 2.8), walnut, .01)
    box("Door top frame", (0, -.035, 2.82), (2.55, .28, .09), walnut, .01)
    box("Ceiling front", (0, -2.5, 3.6), (9.2, 5.1, .18), plaster)
    for y in range(-4, 10, 2):
        box("Window upright", (-4.52, y, 1.75), (.09, .08, 3.5), dark, .008)
    for z in (.08, 1.3, 3.47):
        box("Window rail", (-4.52, 2, z), (.09, 14, .07), dark, .006)
    box("Outside ground", (-9, 2, -.17), (10, 20, .15), stone)
    for y in (-3, 2, 6):
        plant(f"Window plant {y}", (-3.9, y, 0), rng, leaf, ceramic, walnut)
    for y in (2, 5, 8):
        box("Rear ceiling beam", (0, y, 3.55), (9.2, .12, .18), walnut)
    box("Bench seat", (3.55, -2.4, .48), (1.1, 3.2, .14), walnut)
    box("Bench cushion", (3.55, -2.4, .62), (.99, 2.9, .19), cushion, .075)
    for y in (-3.5, -1.3):
        for x in (3.17, 3.93):
            box("Bench brass leg", (x, y, .22), (.055, .055, .44), brass, .008)
    cylinder("Rear circular table", (.35, 3.4, .76), 1.1, .075, walnut)
    cylinder("Table pedestal", (.35, 3.4, .36), .16, .72, dark)
    cylinder("Ceramic vase", (.6, 3.3, 1.02), .14, .45, accent)
    box("Rear display plinth", (3.1, 5.7, .45), (1.1, 1.1, .9), plaster)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=.42, location=(3.1, 5.7, 1.4))
    bpy.context.object.name = "Original brass sculpture"
    bpy.context.object.data.materials.append(brass)
    for poly in bpy.context.object.data.polygons:
        poly.use_smooth = True
    hinge = bpy.data.objects.new("Door hinge controlled by action", None)
    bpy.context.collection.objects.link(hinge)
    hinge.location = (-1.18, 0, 0)
    bpy.context.view_layer.update()
    box("Door leaf", (0, 0, 1.37), (2.36, .075, 2.74), walnut, .015, hinge)
    for x in (-.8, -.4, 0, .4, .8):
        box("Door inlay", (x, -.044, 1.37), (.012, .009, 2.55), brass, .002, hinge)
    box("Door handle", (.95, -.12, 1.25), (.038, .11, .45), brass, .012, hinge)
    # Procedural sky and lights; no panorama or other external file.
    world = bpy.data.worlds.new("Original procedural daylight")
    bpy.context.scene.world = world
    world.use_nodes = True
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING" if bpy.app.version >= (5, 0, 0) else "NISHITA"
    sky.sun_elevation, sky.sun_rotation = math.radians(32), math.radians(135)
    sky.sun_intensity = .8
    world.node_tree.links.new(sky.outputs["Color"], world.node_tree.nodes["Background"].inputs["Color"])
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = .32
    light_data = bpy.data.lights.new("Soft window light", "AREA")
    light_data.energy, light_data.shape, light_data.size, light_data.size_y = 1600, "RECTANGLE", 5, 4
    light = bpy.data.objects.new("Soft window light", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (-4.8, -1, 3.8)
    aim(light, (0, 0, 1))
    bpy.ops.object.camera_add(location=(-.75, -4.4, 1.6))
    camera = bpy.context.object
    camera.name = "Recorded camera"
    camera.data.lens, camera.data.sensor_width, camera.data.sensor_fit = 18, 36, "HORIZONTAL"
    camera.data.clip_start, camera.data.clip_end = .05, 100
    bpy.context.scene.camera = camera
    return camera, hinge


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_format(scene, kind):
    settings = scene.render.image_settings
    if hasattr(settings, "media_type"):
        settings.media_type = "MULTI_LAYER_IMAGE" if kind == "exr" else "IMAGE"
    settings.file_format = "OPEN_EXR_MULTILAYER" if kind == "exr" else "PNG"
    settings.color_depth = "32" if kind == "exr" else "8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-seed", type=int, default=51000)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 1, 25, 41, 53, 65])
    parser.add_argument("--arms", nargs="+", choices=["closed", "open"], default=["closed", "open"])
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--device", choices=["METAL", "CPU"], default="METAL")
    parser.add_argument("--max-seconds", type=float, default=600)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if min(args.width, args.height, args.samples) < 1 or max(args.width, args.height) > 1280:
        parser.error("Positive dimensions up to1280 and positive samples are required")
    if any(not 0 <= f <= 65 for f in args.frames) or len(set(args.frames)) != len(args.frames):
        parser.error("Frame ids must be unique integers from0 through65")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    camera, hinge = build_scene(args.scene_seed)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.seed = args.scene_seed
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = args.width, args.height, 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    if args.device == "METAL":
        pref = bpy.context.preferences.addons["cycles"].preferences
        pref.compute_device_type = "METAL"
        pref.refresh_devices()
        for device in pref.devices:
            device.use = device.type == "METAL"
        if not any(d.type == "METAL" for d in pref.devices):
            raise RuntimeError("No Metal device found; use --device CPU explicitly")
        scene.cycles.device = "GPU"
    else:
        scene.cycles.device = "CPU"
    layer = scene.view_layers[0]
    layer.use_pass_z, layer.use_pass_normal, layer.use_pass_object_index = True, True, True
    for index, obj in enumerate(sorted(bpy.data.objects, key=lambda obj: obj.name), start=1):
        obj.pass_index = index
    output_format(scene, "exr")
    scene.render.image_settings.exr_codec = "ZIP"
    fx = args.width*camera.data.lens/camera.data.sensor_width
    manifest = {"schema": "worldline-atrium-pilot-v1", "blender": bpy.app.version_string,
                "script_sha256": digest(__file__), "scene_seed": args.scene_seed,
                "scene_family": "single_atrium_layout_v1", "seed_scope": "Accent color and plant leaf placement only; not independent architectural layouts",
                "dataset_license": "CC0-1.0", "assets": "Original procedural geometry and materials only",
                "render_engine": "Cycles", "device": args.device, "samples": args.samples,
                "rgb_transform": {"view": scene.view_settings.view_transform, "look": scene.view_settings.look,
                                  "exposure": scene.view_settings.exposure, "gamma": scene.view_settings.gamma,
                                  "display": scene.display_settings.display_device, "exr": "scene-linear"},
                "resolution": [args.width, args.height], "K": [[fx, 0, args.width/2], [0, fx, args.height/2], [0, 0, 1]],
                "camera_model": {"type": "PERSPECTIVE", "lens_mm": camera.data.lens,
                                 "sensor_width_mm": camera.data.sensor_width, "sensor_fit": "HORIZONTAL",
                                 "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y]},
                "camera_convention": "world_from_camera uses OpenCV x right, y down, z forward; world z up, meters",
                "depth_semantics": "Native Cycles axial camera-Z depth in meters; independently verified on Blender5.1.2 first-frame ray casts",
                "protocol": "interact or wait, then24left turns,16waits,24right turns; each turn7.5degrees",
                "interaction_semantics": "Programmed remote door toggle; no reach/contact/collision simulation",
                "objects": {obj.pass_index: obj.name for obj in bpy.data.objects}, "arms": {}}
    conversion = Matrix.Diagonal((1, -1, -1, 1))
    for arm in args.arms:
        output = args.output/arm
        output.mkdir(exist_ok=True)
        controls = ["interact" if arm == "open" else "wait"] + ["left"]*24 + ["wait"]*16 + ["right"]*24
        records, yaw = [], 0.
        for index in range(66):
            action = None if index == 0 else controls[index-1]
            if action == "left": yaw += math.pi/24
            if action == "right": yaw -= math.pi/24
            hinge.rotation_euler.z = math.radians(102) if arm == "open" and index > 0 else 0
            aim(camera, camera.location+Vector((-math.sin(yaw), math.cos(yaw), -.07)))
            bpy.context.view_layer.update()
            record = {"frame": index, "action_from_previous": action, "door_open": arm == "open" and index > 0,
                      "yaw_radians": yaw, "world_from_camera": [list(row) for row in camera.matrix_world@conversion]}
            if index in args.frames:
                if time.perf_counter()-started >= args.max_seconds:
                    raise TimeoutError("Render budget reached between frames; existing outputs retained")
                tick = time.perf_counter()
                exr = output/f"{index:04d}.exr"
                output_format(scene, "exr")
                scene.render.filepath = str(exr)
                bpy.ops.render.render(write_still=True)
                output_format(scene, "png")
                png = output/f"{index:04d}.png"
                bpy.data.images["Render Result"].save_render(str(png), scene=scene)
                record.update({"png": str(png.relative_to(args.output)), "png_sha256": digest(png),
                               "exr": str(exr.relative_to(args.output)), "exr_sha256": digest(exr),
                               "render_and_save_seconds": time.perf_counter()-tick})
                print("WORLDLINE_FRAME "+json.dumps({"arm": arm, "frame": index, "seconds": record["render_and_save_seconds"]}), flush=True)
            records.append(record)
            manifest["arms"][arm] = records
            manifest["elapsed_seconds"] = time.perf_counter()-started
            (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/"original-scene.blend"))
    manifest["blend_sha256"] = digest(args.output/"original-scene.blend")
    manifest["dense_sequence"] = set(args.frames) == set(range(66))
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.perf_counter()-started
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


if __name__ == "__main__":
    main()
