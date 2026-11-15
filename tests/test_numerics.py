import tempfile
import unittest
from pathlib import Path
import hashlib
import io
from importlib.resources import files

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "CPU torch is required for numerical tests")
class Numerics(unittest.TestCase):
    def test_calibration_references(self):
        from roshift.catalog import MODELS, asset
        for model in MODELS:
            meta = asset("calibration", model, "metadata.json")
            for name, shape in [("input_ids.pt", [32, 4096]), ("selection_input_ids.pt", [32, 512])]:
                reference = asset('external_data.json')['files'][f'assets/calibration/{model}/{name}']
                self.assertEqual(reference['file_sha256'], meta['files'][name]['sha256'])
                self.assertEqual(reference['content']['shape'], shape)
                self.assertEqual(reference['content']['dtype'], 'torch.int64')
                self.assertFalse(files('roshift').joinpath('assets','calibration',model,name).is_file())

    def test_quota(self):
        from roshift.runtime.selectors import quota
        self.assertEqual(quota(4096, 40).count(77), 26)
        self.assertEqual(quota(4096, 60).count(52), 6)
        self.assertEqual(quota(4096, 50), [64]*32)

    def test_server_equal_quota_cpu(self):
        from roshift.runtime.selectors import server_blockwise_topk
        x = torch.arange(1, 257).float().reshape(1, 256)
        y = server_blockwise_topk(x, None, 50., 128, torch.arange(256))
        self.assertEqual(int(y.ne(0).sum()), 128)
        self.assertTrue(torch.equal(y[:, 64:128], x[:, 64:128]))
        self.assertTrue(torch.equal(y[:, 192:], x[:, 192:]))

    def test_rotation_and_permutation(self):
        from roshift.prepare.rotation import mixture_rotation, normalized_metric, fixed_permutation
        w = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        m = normalized_metric(w, torch.tensor([1., 2., 1.]))
        self.assertAlmostEqual(float(torch.trace(m)), 1., places=6)
        q = mixture_rotation([m]*6, [.25, .25, .125, .125, .125, .125], torch.float32)
        self.assertTrue(torch.allclose(q.T@q, torch.eye(3), atol=1e-5))
        p = fixed_permutation(128, 11, "layer0:gate")
        self.assertTrue(torch.equal(p, fixed_permutation(128, 11, "layer0:gate")))
        self.assertFalse(torch.equal(p, fixed_permutation(128, 11, "layer1:gate")))

    def test_sidecar(self):
        try: import safetensors
        except ImportError: self.skipTest("safetensors not installed")
        from roshift.prepare.sidecar import write_sidecar, verify_sidecar
        with tempfile.TemporaryDirectory() as temp:
            write_sidecar(temp, [("layer0.q", torch.eye(3))], {"test": True})
            self.assertTrue(verify_sidecar(temp))
            with self.assertRaises(FileExistsError): write_sidecar(temp, [("x", torch.eye(3))], {})
            (Path(temp)/"weights/layer0.q.safetensors").write_bytes(b"bad")
            self.assertFalse(verify_sidecar(temp))


if __name__ == "__main__":
    unittest.main()
