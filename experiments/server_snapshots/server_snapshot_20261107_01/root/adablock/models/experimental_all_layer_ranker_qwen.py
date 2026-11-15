"""Explicit experimental cross-layer inference with an unchanged ranker checkpoint."""
from models.layer_head_sparse_qwen import LayerHeadQKSparseController

class ExperimentalAllLayerRankerController(LayerHeadQKSparseController):
    def __init__(self, *args, all_layer_ranker=False, **kwargs):
        if not all_layer_ranker:
            # Preserve the original default behavior when not explicitly opted in.
            super().__init__(*args, **kwargs)
            return
        ranker=kwargs.get("ranker")
        metadata=kwargs.get("ranker_metadata")
        if ranker is None or metadata is None:
            raise ValueError("All-layer experiment requires the original ranker and metadata.")
        original_layers=tuple(metadata["layers"])
        # The base initializer validates original metadata and builds all-layer
        # digests. No attention/scoring is executed in that initializer.
        kwargs["ranker_untrained_layers"]="cobs"
        super().__init__(*args, **kwargs)
        if ranker.layer_embedding.num_embeddings != len(self.layers):
            raise ValueError("Original layer embedding must cover every real layer ID.")
        self.training_layers=original_layers
        self.ranker_layers=set(range(len(self.layers)))
        self.experimental_all_layer_ranker=True
        assert tuple(metadata["layers"])==original_layers  # never relabel training metadata

    def _select_token_indices(self, layer_idx, query_states):
        if getattr(self,"experimental_all_layer_ranker",False):
            if self.ranker is None or layer_idx not in self.ranker_layers:
                raise RuntimeError("All-layer experiment must never fall back to another selector.")
        return super()._select_token_indices(layer_idx,query_states)
