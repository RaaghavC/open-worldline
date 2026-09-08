# SPDX-License-Identifier: GPL-3.0-or-later
"""Capture independent camera and door interventions using the original scene.

These are programmed training targets, never model-generated predictions.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
import time

import bpy
from mathutils import Matrix, Vector

SCENE_PATH = Path(__file__).resolve().parents[1]/'atrium_data/render.py'
spec = importlib.util.spec_from_file_location('worldline_original_atrium_scene', SCENE_PATH)
scene_code = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene_code)
ARMS = [(motion, door) for motion in ('stationary', 'left', 'right') for door in ('closed', 'interact')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--max-seconds', type=float, default=1200)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    if args.output.exists() or args.samples < 1 or not 0 < args.max_seconds <= 1800:
        parser.error('Require a new output, positive sample count and at most 1800 seconds')
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    camera, hinge = scene_code.build_scene(51000)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.seed = 51000
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 1248, 704, 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = 'AgX'
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'METAL'
    preferences.refresh_devices()
    for device in preferences.devices:
        device.use = device.type == 'METAL'
    if not any(device.use for device in preferences.devices):
        raise RuntimeError('Metal GPU required for this local capture')
    scene.cycles.device = 'GPU'
    scene_code.output_format(scene, 'png')
    conversion = Matrix.Diagonal((1, -1, -1, 1))
    manifest = {
        'schema':'worldline-atrium-factorial-v1', 'status':'running', 'split':'development',
        'scene_family':'single_atrium_layout_v1', 'scene_seed':51000, 'independent_layouts':1,
        'dataset_license':'CC0-1.0', 'scene_source_sha256':scene_code.digest(SCENE_PATH),
        'capture_source_sha256':scene_code.digest(__file__), 'blender':bpy.app.version_string,
        'render_engine':'Cycles', 'device':'METAL', 'samples':args.samples, 'resolution':[1248,704],
        'K':[[624.,0.,624.],[0.,624.,352.],[0.,0.,1.]],
        'camera_convention':'world_from_camera uses OpenCV x right, y down, z forward; world z up; meters',
        'action_channels':['local_right_m','local_up_m','local_forward_m','yaw_left_rad','pitch_up_rad','interact_pulse'],
        'command_alignment':'record t contains the command applied from observation t-1 to t; frame0 has no incoming command',
        'frames_per_arm':17, 'yaw_step_radians':math.pi/120,
        'rgb_transform':{'view':scene.view_settings.view_transform,'look':scene.view_settings.look,'exposure':scene.view_settings.exposure,'gamma':scene.view_settings.gamma,'display':scene.display_settings.display_device},
        'interaction_semantics':'Programmed instantaneous remote door toggle to102degrees; no reach/contact/collision simulation',
        'repeated_states':'Identical camera and door states reuse the first saved PNG for that state; every reuse is identified',
        'model_inputs':['initial_rgb','requested_command_deltas'],
        'training_targets':['future_rgb'],
        'extra_metadata_not_model_inputs':['door_open','yaw_radians','world_from_camera'],
        'neural_model_execution':False, 'arms':{},
    }
    state_images = {}
    rendered = reused = 0
    for motion, door in ARMS:
        name = motion+'_'+door
        folder = args.output/name
        folder.mkdir()
        records = []
        for frame in range(17):
            if time.perf_counter()-start > args.max_seconds:
                raise TimeoutError('Capture time limit reached between renders; incomplete output retained')
            direction = {'stationary':0,'left':1,'right':-1}[motion]
            yaw = direction*frame*math.pi/120
            opened = door == 'interact' and frame > 0
            incoming = None if frame == 0 else [0.,0.,0.,direction*math.pi/120,0.,float(door == 'interact' and frame == 1)]
            hinge.rotation_euler.z = math.radians(102) if opened else 0.
            scene_code.aim(camera, camera.location+Vector((-math.sin(yaw),math.cos(yaw),-.07)))
            bpy.context.view_layer.update()
            png = folder/f'{frame:04d}.png'
            state_key = (direction*frame, opened) if direction else (0, opened)
            tick = time.perf_counter()
            reuse = state_images.get(state_key)
            if reuse:
                shutil.copyfile(args.output/reuse, png)
                reused += 1
            else:
                scene.render.filepath = str(png)
                bpy.ops.render.render(write_still=True)
                state_images[state_key] = str(png.relative_to(args.output))
                rendered += 1
            record = {'frame':frame,'command_from_previous':incoming,'door_open':opened,'yaw_radians':yaw,
                'world_from_camera':[list(row) for row in camera.matrix_world@conversion],
                'png':str(png.relative_to(args.output)),'png_sha256':scene_code.digest(png),
                'reused_from':reuse,'render_or_copy_seconds':time.perf_counter()-tick}
            records.append(record)
            manifest['arms'][name] = records
            manifest['elapsed_seconds'] = time.perf_counter()-start
            manifest['rendered_unique_states'], manifest['copied_identical_states'] = rendered,reused
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print('WORLDLINE_FACTORIAL '+json.dumps({'arm':name,'frame':frame,'rendered':reuse is None,'elapsed_seconds':manifest['elapsed_seconds']}),flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output/'original-scene.blend'))
    manifest['blend_sha256'] = scene_code.digest(args.output/'original-scene.blend')
    manifest['status'] = 'complete'
    manifest['elapsed_seconds'] = time.perf_counter()-start
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
