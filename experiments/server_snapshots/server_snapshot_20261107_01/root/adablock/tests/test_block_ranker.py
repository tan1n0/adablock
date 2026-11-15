import tempfile
import unittest
from pathlib import Path

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.block_ranker import BlockRanker, BlockRankerConfig, cobs_terms, load_ranker, save_ranker
from models.layer_head_sparse_qwen import LayerHeadQKSparseController
from scripts.build_ranking_dataset import collect_document, ranking_metadata
from utils.ranking_data import RankingDataset, RankingWriter, assert_disjoint, document_identity
from utils.ranking_loss import PairSamplingConfig, coverage_positive_ids, sample_pairs, weighted_rank_loss
from utils.qk_block_oracle import build_qk_block_digests


class RankerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                            max_position_embeddings=512, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        cls.base = Qwen2ForCausalLM(config).eval()

    def make_dataset(self, path, ids, doc="one"):
        meta = ranking_metadata(self.base, "tiny-random-qwen", [0, 1], 4, 4, 2, 2)
        writer = RankingWriter(path, meta)
        identity = document_identity({"source": "unit", "doc_id": doc}, str(ids.tolist()))
        collect_document(self.base, ids, writer, identity, sample_stride=4, query_chunk_size=2)
        writer.finish()
        return RankingDataset(path)

    def test_coverage_and_sampling(self):
        mass = torch.tensor([.20, .08, .02])
        self.assertEqual(coverage_positive_ids(mass, .70, .95).tolist(), [0, 1])
        scores = torch.tensor([.1, .2, .9])
        cfg = PairSamplingConfig(max_positives=16, positive_pairs=32)
        pairs = sample_pairs(scores, mass, .70, cfg, torch.Generator().manual_seed(42))
        self.assertEqual(set(map(tuple, pairs.tolist())), {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(sample_pairs(scores, mass, .99, cfg).numel(), 0)
        self.assertEqual(sample_pairs(torch.ones(4), torch.ones(4)/4, 0, cfg).numel(), 0)
        self.assertEqual(sample_pairs(torch.empty(0), torch.empty(0), 1, cfg).numel(), 0)

    def test_loss_direction_and_empty(self):
        mass = torch.tensor([.2, .02])
        pairs = torch.tensor([[0, 1]])
        wrong = torch.tensor([0., 1.], requires_grad=True)
        right = torch.tensor([1., 0.])
        loss = weighted_rank_loss(wrong, mass, pairs)
        self.assertGreater(float(loss), float(weighted_rank_loss(right, mass, pairs)))
        loss.backward()
        self.assertLess(float(wrong.grad[0]), 0)
        self.assertGreater(float(wrong.grad[1]), 0)
        self.assertEqual(float(weighted_rank_loss(wrong, mass, pairs[:0])), 0)

    def test_small_rank_and_sign_invariance(self):
        d = build_qk_block_digests(torch.randn(2, 5, 8), block_size=4, cobs_rank=4)
        self.assertEqual(d.covariance_basis.shape, (2, 2, 3, 8))
        q = torch.randn(2, 8)
        a, v = cobs_terms(q, d.mean[0], d.covariance_basis[0], d.covariance_eigenvalues[0])
        _, flipped = cobs_terms(q, d.mean[0], -d.covariance_basis[0], d.covariance_eigenvalues[0])
        torch.testing.assert_close(v, flipped)
        self.assertTrue(bool(v.isfinite().all()))
        zero = build_qk_block_digests(torch.ones(1, 1, 8), 1, cobs_rank=4)
        self.assertEqual(zero.covariance_basis.shape[-2], 0)

    def test_causal_collection_mass_and_components(self):
        ids = torch.randint(0, 64, (1, 64))
        changed = ids.clone()
        changed[:, 40:] = torch.randint(0, 64, (1, 24))
        with tempfile.TemporaryDirectory() as tmp:
            a = self.make_dataset(Path(tmp)/"a", ids)
            b = self.make_dataset(Path(tmp)/"b", changed)
            compared = 0
            for i in range(len(a)):
                x, y = a[i], b[i]
                self.assertAlmostEqual(float(x["teacher_mass"].sum()+x["fixed_mass"]), 1, places=5)
                self.assertTrue(bool(((x["block_ids"]+1)*4 <= x["token_position"]).all()))
                if x["token_position"] < 40:
                    compared += 1
                    for name in ("query", "mean", "teacher_mass", "fixed_mass"):
                        torch.testing.assert_close(x[name], y[name], atol=1e-6, rtol=1e-5)
            self.assertGreater(compared, 0)
            with self.assertRaisesRegex(ValueError, "overlap"):
                assert_disjoint(a, b)
            a.manifest["documents"][0]["components"] = ["shared-component"]
            b.manifest["documents"][0]["components"] = ["shared-component"]
            with self.assertRaisesRegex(ValueError, "overlap"):
                assert_disjoint(a, b)

    def test_gqa_teacher_against_eager(self):
        ids = torch.randint(0, 64, (1, 40))
        with tempfile.TemporaryDirectory() as tmp:
            data = self.make_dataset(Path(tmp)/"data", ids)
            self.base.config._attn_implementation = "eager"
            try:
                with torch.no_grad():
                    out = self.base(ids, output_attentions=True, use_cache=False)
                for sample in (data[0], data[1], data[len(data) - 1]):
                    layer, head, t = sample["layer_id"], sample["kv_head_id"], sample["token_position"]
                    weights = out.attentions[layer][0, head*2:(head+1)*2, t].mean(0)
                    expected = torch.stack([weights[int(b)*4:int(b)*4+4].sum() for b in sample["block_ids"]])
                    torch.testing.assert_close(sample["teacher_mass"], expected, atol=1e-6, rtol=1e-5)
            finally:
                self.base.config._attn_implementation = "sdpa"

    def test_checkpoint_runtime_and_dense_equivalence(self):
        ids = torch.randint(0, 64, (1, 40))
        with tempfile.TemporaryDirectory() as tmp:
            data = self.make_dataset(Path(tmp)/"data", ids)
            ranker = BlockRanker(BlockRankerConfig(8, 2, 2, projection_size=8, hidden_size=16, block_chunk_size=3)).eval()
            sample = data[0]
            original = ranker(sample)
            save_ranker(Path(tmp)/"ranker.pt", ranker, data.metadata, {})
            loaded, payload = load_ranker(Path(tmp)/"ranker.pt")
            torch.testing.assert_close(original, loaded(sample))
            changed = dict(sample, basis=-sample["basis"])
            torch.testing.assert_close(original, loaded(changed))
            # Capture the exact scaled query/sample used by the online scorer.
            prefix = ids[:, :sample["token_position"]]
            current = ids[:, sample["token_position"]:sample["token_position"]+1]
            with torch.no_grad():
                prefill = self.base.model(prefix, use_cache=True)
                dense = self.base.model(torch.cat((prefix, current), 1), use_cache=False).last_hidden_state[:, -1]
            cache = prefill.past_key_values.to_legacy_cache()
            controller = LayerHeadQKSparseController(self.base, cache, 4, 1, 2, cobs_rank=2,
                         ranker=loaded, ranker_metadata=payload["data_metadata"], profile=True)
            controller.begin_step(prefix.shape[1], 100, 100, False)
            controller.install()
            try:
                t = prefix.shape[1]
                with torch.no_grad():
                    sparse = self.base.model(current, position_ids=torch.tensor([[t]]),
                              cache_position=torch.tensor([t]), use_cache=False).last_hidden_state[:, -1]
                stats = controller.finish_step()
                torch.testing.assert_close(sparse, dense, atol=1e-5, rtol=1e-4)
                self.assertAlmostEqual(stats.avg_selected_tokens, prefix.shape[1])
                controller.begin_step(t+1, 100, 0, False)
                with torch.no_grad():
                    zero = self.base.model(current, position_ids=torch.tensor([[t+1]]),
                               cache_position=torch.tensor([t+1]), use_cache=False).last_hidden_state
                controller.finish_step()
                self.assertTrue(bool(zero.isfinite().all()))
            finally:
                controller.uninstall()
            # Shared feature path: offline sample vs controller scorer.
            d = build_qk_block_digests(cache[sample["layer_id"]][0][0], 4, cobs_rank=2)
            controller._history_len = prefix.shape[1]
            queries = sample["query"].unsqueeze(0).expand(2, -1, -1)
            scores = controller._learned_block_scores(sample["layer_id"], queries, d, sample["block_ids"])
            torch.testing.assert_close(scores[sample["kv_head_id"], sample["block_ids"]], original,
                                       atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
