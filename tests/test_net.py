"""Unit tests for the from-scratch network (numpy only). Run: python3 -m unittest discover -s tests -v"""
import os, sys, unittest, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gn_net as N

F = np.float32


def tiny(seed=1):
    net = N.Net(d=32, L=2, H=2, ctx=32, vcap=64, seed=seed); net.set_vocab(40); return net


def batch(rng, B=3, T=16):
    return rng.integers(1, 40, (B, T)), rng.integers(1, 40, (B, T)), np.ones((B, T), F)


class StateConditioning(unittest.TestCase):
    def test_zero_projector_is_function_preserving(self):
        net, rng = tiny(), np.random.default_rng(0); ids, tg, w = batch(rng)
        S = rng.random((3, 16, N.NS)).astype(F)
        a, _ = net.loss(ids, tg, w); b, _ = net.loss(ids, tg, w, S)
        self.assertAlmostEqual(float(a.d), float(b.d), places=6)

    def test_projector_gradient_matches_finite_differences(self):
        net, rng = tiny(), np.random.default_rng(0); ids, tg, w = batch(rng)
        S = rng.random((3, 16, N.NS)).astype(F)
        net.P["Ws"].d[:] = rng.normal(0, 0.1, net.P["Ws"].d.shape).astype(F)
        for p in net.P.values(): p.g = None
        l, _ = net.loss(ids, tg, w, S); N.backward(l); g = net.P["Ws"].g[5, 7]
        W, eps = net.P["Ws"].d, 1e-2
        W[5, 7] += eps; lp = float(net.loss(ids, tg, w, S)[0].d); W[5, 7] -= 2 * eps; lm = float(net.loss(ids, tg, w, S)[0].d); W[5, 7] += eps
        self.assertAlmostEqual((lp - lm) / (2 * eps), float(g), delta=2e-4 + 0.05 * abs(float(g)))

    def test_cached_generation_matches_full_forward(self):
        net, rng = tiny(), np.random.default_rng(0)
        net.P["Ws"].d[:] = rng.normal(0, 0.1, net.P["Ws"].d.shape).astype(F)
        sv = rng.random(N.NS).astype(F); prompt = [1, 2, 3, 4]
        out = net.generate(prompt, max_new=8, temp=1e-3, top_k=1, top_p=1.0, rep=1.0, sv=sv, rng=np.random.default_rng(0))
        cur, ref = list(prompt), []
        with N.no_grad():
            for _ in range(8):
                x = net.forward(np.array([cur]), np.broadcast_to(sv, (1, len(cur), N.NS)))
                ref.append(int(np.argmax(x.d[0, -1] @ net.P["E"].d[:net.V].T))); cur.append(ref[-1])
        self.assertEqual(out, ref)

    def test_old_checkpoint_without_projector_loads(self):
        net = tiny()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "n.npz"); net.save(p); z = dict(np.load(p)); z.pop("P_Ws"); np.savez(os.path.join(d, "old.npz"), **z)
            old = N.Net.load(os.path.join(d, "old.npz"))
        self.assertEqual(float(abs(old.P["Ws"].d).max()), 0.0)


class Growth(unittest.TestCase):
    def test_grow_layer_preserves_function(self):
        net, rng = tiny(), np.random.default_rng(0); ids, tg, w = batch(rng)
        a, _ = net.loss(ids, tg, w); net.grow_layer(); b, _ = net.loss(ids, tg, w)
        self.assertAlmostEqual(float(a.d), float(b.d), places=5)

    def test_widen_preserves_function(self):
        net, rng = tiny(), np.random.default_rng(0); ids, tg, w = batch(rng)
        a, _ = net.loss(ids, tg, w); net.widen_mlp(net.layers[0], 16); b, _ = net.loss(ids, tg, w)
        self.assertAlmostEqual(float(a.d), float(b.d), places=5)

    def test_training_step_reduces_loss_on_a_fixed_batch(self):
        net, rng = tiny(), np.random.default_rng(0); ids, tg, w = batch(rng)
        first = net.step(ids, tg, w, lr=2e-2)[0]
        for _ in range(30): last = net.step(ids, tg, w, lr=2e-2)[0]
        self.assertLess(last, first)


if __name__ == "__main__":
    unittest.main()
