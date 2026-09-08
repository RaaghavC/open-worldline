"""Tiny analytic output fixture plus corrupted scalar and binary hash checks."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from arrays import inventory, tensor_sha
from audit import ARMS, check_output_hashes, numeric_outputs


def save(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    header = {}
    payload = []
    for name, value in sorted(values.items()):
        value = np.ascontiguousarray(value, dtype='<f4')
        b = value.tobytes()
        header[name] = dict(shape=list(value.shape), dtype='F32', data_offsets=[offset, offset + len(b)])
        offset += len(b)
        payload.append(b)
    data = json.dumps(header, separators=(',', ':')).encode()
    data += b' ' * (-len(data) % 8)
    path.write_bytes(struct.pack('<Q', len(data)) + data + b''.join(payload))


def fixture(root):
    shape = (1, 1, 5, 2, 2)
    initial = {'projections.24_q.a.weight': np.ones((2, 2), np.float32),
               'projections.24_q.b.weight': np.zeros((2, 2), np.float32)}
    final = {k: v + np.float32(.125) for k, v in initial.items()}
    save(root/'controller-initial.safetensors', initial)
    save(root/'controller-final.safetensors', final)
    windows = {}
    for i, arm in enumerate(ARMS):
        x = np.full(shape, i, np.float32)
        x[:, :, :1] = 0
        windows[arm] = {'target': x, 'observation': x[:, :, :1].copy(), 'commands': np.zeros((1, 16, 6), np.float32)}
    rows = [dict(update=1, noise_key='noise_0000', branches=list(ARMS[:2]), auxiliary_edge=list(ARMS[:2])),
            dict(update=2, noise_key='noise_0001', branches=list(ARMS[2:4]), auxiliary_edge=None)]
    data = dict(plan={'schedule': rows}, windows=windows, initial=initial,
                draws={'noise_0000': np.full(shape, 3, np.float32), 'noise_0001': np.full(shape, 4, np.float32)})
    report = {'parity': [], 'updates': [], 'completed_updates': 2}
    for c, value in (('positive', 6), ('negative', 7)):
        x = np.full(shape, value, np.float32)
        save(root/('parity/native-'+c+'.safetensors'), {'velocity': x})
        for mode in ('full', 'cached'):
            for arm in ARMS:
                report['parity'].append(dict(context=c, arm=arm, path=mode, exact_equal=True, max_abs=0.,
                    native_sha256=tensor_sha(x), actual_sha256=tensor_sha(x), seconds=.1, feature_extract_seconds=.2))
    for i, row in enumerate(rows):
        prefix = root/f'update-{i+1}'
        noise = data['draws'][row['noise_key']]
        # Exactly representable integer velocities plus 1/4 or 1/2 residual.
        loss = .0625 if i == 0 else .25
        for arm in row['branches']:
            prediction = noise - windows[arm]['target'] + np.float32(.25 if i == 0 else .5)
            save(prefix/('main-'+arm+'.safetensors'), {'velocity': prediction})
        aux = dict(enabled=False, edge=None, weight=1., feature_extracts=0, predictions=0, loss=0.)
        if i == 0:
            for number, label, value in ((0, 'positive', 1), (0, 'negative', .5), (1, 'positive', 2), (1, 'negative', 1)):
                save(prefix/f'aux-{number}-{label}.safetensors', {'velocity': np.full(shape, value, np.float32)})
            # G0=3, G1=6, clean effect=-3; target effect=+1, squared error=16.
            aux = dict(enabled=True, edge=list(ARMS[:2]), weight=1., feature_extracts=2, predictions=4, loss=16.)
        gradients = {n: np.full_like(v, (.25 if n.endswith('.b.weight') else 0.) if i == 0 else .125)
                     for n, v in initial.items()}
        save(prefix/'gradients.safetensors', gradients)
        norms = {n: float(np.linalg.norm(v.astype(np.float64))) for n, v in gradients.items()}
        total_norm = float(np.sqrt(sum(n*n for n in norms.values())))
        report['updates'].append(dict(update=i+1, optimizer_updates=1, main_predictions=2,
            all_controller_gradients_present_finite=True, foundation_gradients_absent=True,
            main=[dict(arm=a, future_flow_mse=loss) for a in row['branches']], auxiliary=aux,
            total_objective=loss+aux['loss'], gradient_l2_before_clip=total_norm,
            gradient_l2_after_clip=total_norm, profile_seconds=.3, main_seconds=.2,
            gradients=dict(all_present_finite=True, parameter_tensors=len(initial), l2_after_clip=total_norm,
                parameters={n:dict(elements=v.size, l2=norms[n], max_abs=float(np.abs(v).max())) for n, v in gradients.items()})))
    report['output_sha256'] = {n: r['sha256'] for n, r in inventory(root).items()}
    return data, report


class AuditTests(unittest.TestCase):
    def test_valid_tiny_analytic_losses_gradient_and_parity_records(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            data, report = fixture(root)
            check_output_hashes(root, report)
            value = numeric_outputs(root, report, data)
            self.assertEqual(value['updates'][0]['auxiliary_mse'], 16.)
            self.assertEqual(value['updates'][0]['main_mse'], [.0625, .0625])
            self.assertEqual(value['updates'][1]['main_mse'], [.25, .25])
            self.assertEqual(value['updates'][0]['non_B_gradient_l2'], 0.)
            self.assertGreater(value['updates'][1]['non_B_gradient_l2'], 0.)
            self.assertEqual(value['raw_bridged_parity_tensors_retained'], 0)

    def test_corrupted_scalar_parity_record_and_retained_file_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            data, report = fixture(root)
            for change in ('loss', 'parity'):
                bad = copy.deepcopy(report)
                if change == 'loss':
                    bad['updates'][0]['main'][0]['future_flow_mse'] += 1
                else:
                    bad['parity'][0]['actual_sha256'] = '0' * 64
                with self.assertRaises(ValueError):
                    numeric_outputs(root, bad, data)
            p = root/'update-1/gradients.safetensors'
            p.write_bytes(p.read_bytes()[:-1] + b'X')
            with self.assertRaisesRegex(ValueError, 'output inventory'):
                check_output_hashes(root, report)


if __name__ == '__main__':
    unittest.main()
