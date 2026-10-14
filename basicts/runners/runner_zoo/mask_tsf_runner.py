from typing import Dict

import torch

from .simple_tsf_runner import SimpleTimeSeriesForecastingRunner


class MaskedTimeSeriesForecastingRunner(SimpleTimeSeriesForecastingRunner):
    """
    Runner that randomly masks a fraction of history time steps to simulate
    missing data and test model robustness.

    Config parameters:
        MASK_RATIO (float, 0.0~1.0): Fraction of history time steps to randomly zero out
            during training (default 0.0 = no masking).
            Applied independently per node; only the flow channel (channel 0) is masked,
            time-of-day / day-of-week channels are kept intact.
        TEST_MASK_RATIO (float or None): Missing ratio applied during the test pipeline.
            None (default) = no masking at evaluation time (backward compatible).
            When set, a deterministic mask (seed = TEST_MASK_SEED) is applied during
            test only; validation stays clean. The generator is re-seeded at the start
            of every test pass, so all models tested at the same ratio see identical
            masked inputs (fair comparison).
        TEST_MASK_SEED (int): seed for the deterministic evaluation mask (default 2024).
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.mask_ratio = cfg.get('MASK_RATIO', 0.0)
        self.mask_ratio_eval = cfg.get('TEST_MASK_RATIO', None)
        self.test_mask_seed = cfg.get('TEST_MASK_SEED', 2024)
        # suffix appended to test_metrics/test_results file names when evaluating
        # with a missing ratio (e.g. '_mr20'), so repeated evaluations with
        # different ratios do not overwrite each other.
        self.test_suffix = '' if self.mask_ratio_eval is None \
            else '_mr{}'.format(int(round(self.mask_ratio_eval * 100)))
        self._in_test = False
        self._test_mask_gen = None

    def on_test_start(self) -> None:
        """Seed the deterministic evaluation-mask generator at the start of every
        test pass, so the masked test inputs do not depend on how many epochs the
        model was trained (or whether it was resumed)."""
        self._in_test = True
        if self.mask_ratio_eval is not None:
            # CPU generator keeps the mask sequence identical across devices/runs.
            self._test_mask_gen = torch.Generator()
            self._test_mask_gen.manual_seed(self.test_mask_seed)

    def on_test_end(self) -> None:
        self._in_test = False
        self._test_mask_gen = None

    def _apply_flow_mask(self, history_data: torch.Tensor, ratio: float,
                         gen: torch.Generator = None) -> torch.Tensor:
        """Randomly zero out `ratio` of history time steps on the flow channel only.

        history_data: [B, L, N, C] (after feature selection).
        """
        B, L, N, C = history_data.shape
        if gen is not None:
            keep_mask = (torch.rand(B, L, N, 1, generator=gen) > ratio).float()
            keep_mask = keep_mask.to(history_data.device)
        else:
            keep_mask = (torch.rand(B, L, N, 1, device=history_data.device) > ratio).float()
        history_data[..., 0:1] = history_data[..., 0:1] * keep_mask
        return history_data

    def _maybe_mask_history(self, history_data: torch.Tensor,
                            train: bool) -> torch.Tensor:
        """Apply the training random mask or the deterministic evaluation mask
        (whichever applies to the current phase) to the flow channel of history_data.
        Custom runners (e.g. DGCRN/MTGNN) can reuse this in their own forward."""
        if train and self.mask_ratio > 0:
            history_data = self._apply_flow_mask(history_data, self.mask_ratio)
        elif (not train) and self._in_test and self.mask_ratio_eval is not None:
            history_data = self._apply_flow_mask(history_data, self.mask_ratio_eval,
                                                 gen=self._test_mask_gen)
        return history_data

    def forward(self, data: Dict, epoch: int = None, iter_num: int = None,
                train: bool = True, **kwargs) -> Dict:
        """Forward pass with optional random history masking."""

        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # ---- Random missing mask on the flow channel only ----
        history_data = self._maybe_mask_history(history_data, train)

        model_return = self.model(history_data=history_data,
                                  future_data=future_data_4_dec,
                                  batch_seen=iter_num, epoch=epoch, train=train)

        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return
