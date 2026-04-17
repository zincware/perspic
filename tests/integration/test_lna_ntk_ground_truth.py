"""End-to-end LNA verification against the analytic empirical NTK.

Builds a tiny sequence model (Embedding -> Linear) producing (B, S, K) logits,
computes the full eNTK by row-wise Jacobian extraction, and checks that the
LNA components produced by the perspic calculators match the analytic values.

The full-NTK oracle depends on nothing but ``model.parameters()`` and
``torch.autograd``, so a passing test pins down the entire LNA pipeline
(SamplewiseCalculatorOpacus + Linearizer + CouplingCalculator) on a
sequence-shaped output without relying on cross-backend agreement.

This is a port of the methodology used in
``studies/2026_experiments_cva/llm_tasks/full_ntk_evaluation`` shrunk to
CI-friendly size (D_out = 30, P ~ 100).
"""

import pytest
import torch

# Opacus emits a benign UserWarning on the embedding path because the input
# is a long-tensor that does not require grad; the full backward hook still
# fires and complains. The numerics are correct — silence it for clean output.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Full backward hook is firing:UserWarning"
)
import torch.nn as nn
import torch.nn.functional as F

from perspic.calculator.coupling import CouplingCalculator
from perspic.calculator.linearizer import Linearizer
from perspic.calculator.samplewise_opacus import SamplewiseCalculatorOpacus
from perspic.utils import BatchStatSnapshot

VOCAB_SIZE = 8
EMBED_DIM = 4
N_CLASSES = 5
BATCH_SIZE = 2
SEQ_LEN = 3


class TinySequenceModel(nn.Module):
    """Embedding -> Linear producing (B, S, K) logits."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, N_CLASSES)

    def forward(self, x):
        return self.head(self.embed(x))


def cross_entropy_flat(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


@pytest.fixture
def model_and_batch():
    torch.manual_seed(0)
    # Model is left at the default float32. float64 would give a tighter
    # oracle comparison but opacus's embedding_norm_sample (see
    # opacus/grad_sample/embedding_norm_sample.py) allocates a float32
    # accumulator and crashes with a dtype mismatch on a float64 model.
    model = TinySequenceModel()
    x = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
    y = torch.randint(0, N_CLASSES, (BATCH_SIZE, SEQ_LEN))
    return model, x, y


def _full_ntk_lna(model, x, y):
    """Compute LNA components from the full Jacobian (analytic ground truth).

    Returns dict with chi_net, chi_loss, delta_loss, chi_align matching the
    convention used by the perspic calculators with normalize=False.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    P = sum(p.numel() for p in params)

    logits = model(x)
    flat = logits.reshape(-1)
    D_out = flat.numel()

    # Row-wise Jacobian: J[d] = grad of flat[d] w.r.t. all params
    J = torch.zeros(D_out, P, dtype=logits.dtype)
    for d in range(D_out):
        model.zero_grad()
        flat[d].backward(retain_graph=(d < D_out - 1))
        J[d] = torch.cat([p.grad.detach().flatten() for p in params])

    # Loss-space gradient: dL/df at the same outputs
    logits_detached = logits.detach().requires_grad_(True)
    loss = cross_entropy_flat(logits_detached, y)
    (grad_f,) = torch.autograd.grad(loss, logits_detached)
    grad_f_flat = grad_f.reshape(-1)

    chi_net = (J**2).sum().item()  # trace(J J^T)
    chi_loss = (grad_f_flat**2).sum().item()  # ||grad_f L||^2
    delta_loss = -((J.T @ grad_f_flat) ** 2).sum().item()  # -||grad_theta L||^2
    chi_align = (-delta_loss) / (chi_net * chi_loss)

    model.zero_grad()
    return {
        "chi_net": chi_net,
        "chi_loss": chi_loss,
        "delta_loss": delta_loss,
        "chi_align": chi_align,
    }


def _perspic_lna(model, x, y, approximate_with_n=None):
    """Run the perspic LNA pipeline (mirrors the study script's compute_lna).

    Uses ``normalize=False`` so the returned scalars are directly comparable
    to the analytic values produced by ``_full_ntk_lna``.
    """
    sample_calc = SamplewiseCalculatorOpacus(
        strict=False, approximate_with_n=approximate_with_n
    )
    linearizer = Linearizer()
    coupling = CouplingCalculator()

    with BatchStatSnapshot(model, x):
        sample_results = sample_calc.compute(
            model, cross_entropy_flat, x, y, normalize=False
        )
        chi_net = sample_results["batch_grad_norms_network"].item()
        chi_loss = sample_results["batch_grad_norms_loss"].item()

        probe = linearizer.compute(
            model=model, criterion=cross_entropy_flat, x1=x, y1=y
        )
        _, _, delta_loss = probe["self"]

        chi_align = coupling.calculate(
            delta_loss=delta_loss, chi_loss=chi_loss, chi_net=chi_net
        )

    return {
        "chi_net": chi_net,
        "chi_loss": chi_loss,
        "delta_loss": delta_loss,
        "chi_align": float(chi_align),
    }


class TestLNAvsFullNTK:
    """Verify perspic's LNA pipeline against the analytic NTK on a sequence model.

    The new (B, S, K) reshape branch in SamplewiseCalculatorOpacus has no other
    test coverage; the full-NTK oracle is the strongest available check because
    it bypasses every other perspic backend.
    """

    def test_lna_matches_full_ntk_exact(self, model_and_batch):
        model, x, y = model_and_batch
        truth = _full_ntk_lna(model, x, y)
        got = _perspic_lna(model, x, y, approximate_with_n=None)

        for key in ("chi_net", "chi_loss", "delta_loss", "chi_align"):
            assert got[key] == pytest.approx(
                truth[key], rel=1e-4, abs=1e-6
            ), f"{key}: perspic={got[key]:.10g}, ntk={truth[key]:.10g}"

    def test_lna_matches_full_ntk_hutchinson(self, model_and_batch):
        # Only chi_net is affected by the Rademacher projection; the other
        # three components do not depend on approximate_with_n.
        model, x, y = model_and_batch
        truth = _full_ntk_lna(model, x, y)

        torch.manual_seed(123)
        got = _perspic_lna(model, x, y, approximate_with_n=2048)

        # rel=0.02 is an honest envelope, not just a fixed-seed pin: across
        # five seeds at n=2048 the worst observed relative error on this
        # model was 0.6%, so 2% leaves ~3x headroom and would catch a small
        # systematic bias in the Rademacher path.
        assert got["chi_net"] == pytest.approx(truth["chi_net"], rel=0.02), (
            f"chi_net (Hutchinson): perspic={got['chi_net']:.6g}, "
            f"ntk={truth['chi_net']:.6g}"
        )
        # The other components share the deterministic code path and should
        # still match the ground truth tightly.
        for key in ("chi_loss", "delta_loss"):
            assert got[key] == pytest.approx(
                truth[key], rel=1e-4, abs=1e-6
            ), f"{key}: perspic={got[key]:.10g}, ntk={truth[key]:.10g}"
