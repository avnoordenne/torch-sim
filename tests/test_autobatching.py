from typing import Any

import pytest
import torch

import torch_sim as ts
from tests.conftest import DEVICE, DTYPE
from torch_sim.autobatching import (
    BinningAutoBatcher,
    InFlightAutoBatcher,
    MemoryModel,
    _n_edges_scalers,
    _nonnegative_least_squares,
    _per_copy_cost_mib,
    _reference_systems,
    calculate_memory_scalers,
    determine_max_batch_size,
    fit_memory_model,
    to_constant_volume_bins,
)
from torch_sim.models.lennard_jones import LennardJonesModel
from torch_sim.state import detach_state_graph


def test_exact_fit():
    values = [1, 2, 1]
    bins = to_constant_volume_bins(values, 2)
    assert len(bins) == 2


def test_weight_pos():
    values = [[1, "x"], [2, "y"], [1, "z"]]
    bins = to_constant_volume_bins(values, 2, weight_pos=0)
    for vol_bin in bins:
        for item in vol_bin:
            assert isinstance(item, list)
            assert isinstance(item[0], int)
            assert isinstance(item[1], str)


def test_key_func():
    values = [{"x": "a", "y": 1}, {"x": "b", "y": 5}, {"x": "b", "y": 3}]
    bins = to_constant_volume_bins(values, 2, key=lambda x: x["y"])

    for vol_bin in bins:
        for item in vol_bin:
            assert isinstance(item, dict)
            assert "x" in item
            assert "y" in item


def test_no_fit():
    values = [42, 24]
    bins = to_constant_volume_bins(values, 20)
    assert bins == [[42], [24]]


def test_bounds_and_tuples():
    c = [
        ("a", 10, "foo"),
        ("b", 10, "log"),
        ("c", 11),
        ("d", 1, "bar"),
        ("e", 2, "bommel"),
        ("f", 7, "floggo"),
    ]
    V_max = 11

    bins = to_constant_volume_bins(c, V_max, weight_pos=1, upper_bound=11)
    bins = [sorted(_bin, key=lambda x: x[0]) for _bin in bins]
    assert bins == [
        [("a", 10, "foo"), ("d", 1, "bar")],
        [("b", 10, "log")],
        [
            ("e", 2, "bommel"),
            ("f", 7, "floggo"),
        ],
    ]

    bins = to_constant_volume_bins(c, V_max, weight_pos=1, lower_bound=1)
    bins = [sorted(_bin, key=lambda x: x[0]) for _bin in bins]
    assert bins == [
        [("c", 11)],
        [("a", 10, "foo")],
        [("b", 10, "log")],
        [
            ("e", 2, "bommel"),
            ("f", 7, "floggo"),
        ],
    ]

    bins = to_constant_volume_bins(c, V_max, weight_pos=1, lower_bound=1, upper_bound=11)
    bins = [sorted(_bin, key=lambda x: x[0]) for _bin in bins]
    assert bins == [
        [("a", 10, "foo")],
        [("b", 10, "log")],
        [("e", 2, "bommel"), ("f", 7, "floggo")],
    ]


def test_calculate_scaling_metric(si_sim_state: ts.SimState) -> None:
    """Test calculation of scaling metrics for a state."""
    # Test n_atoms metric
    n_atoms_metric = calculate_memory_scalers(si_sim_state, "n_atoms")
    assert n_atoms_metric == [si_sim_state.n_atoms]

    # Test n_atoms_x_density metric
    density_metric = calculate_memory_scalers(si_sim_state, "n_atoms_x_density")
    volume = torch.abs(torch.linalg.det(si_sim_state.cell[0])) / 1000
    expected = si_sim_state.n_atoms * (si_sim_state.n_atoms / volume.item())
    assert pytest.approx(density_metric[0], rel=1e-5) == expected

    # Test invalid metric (intentionally pass invalid value to test error handling)
    with pytest.raises(ValueError, match="Invalid metric"):
        calculate_memory_scalers(si_sim_state, "invalid_metric")


def test_calculate_scaling_metric_non_periodic(benzene_sim_state: ts.SimState) -> None:
    """Test calculation of scaling metrics for a non-periodic state."""
    n_atoms_metric = calculate_memory_scalers(benzene_sim_state, "n_atoms")
    assert n_atoms_metric == [benzene_sim_state.n_atoms]

    n_atoms_x_density_metric = calculate_memory_scalers(
        benzene_sim_state, "n_atoms_x_density"
    )
    assert n_atoms_x_density_metric[0] > 0
    bbox = (
        benzene_sim_state.positions.max(dim=0).values
        - benzene_sim_state.positions.min(dim=0).values
    ).clone()
    pbc_tensor = torch.as_tensor(
        benzene_sim_state.pbc, device=benzene_sim_state.device, dtype=torch.bool
    )
    if pbc_tensor.ndim == 0:
        pbc_tensor = pbc_tensor.repeat(3)
    for idx, p in enumerate(pbc_tensor):
        if not p:
            bbox[idx] += 2.0
    assert pytest.approx(n_atoms_x_density_metric[0], rel=1e-5) == (
        benzene_sim_state.n_atoms**2 / (bbox.prod().item() / 1000)
    )


def test_calculate_scaling_metric_mixed_pbc_uses_per_system_path(
    si_double_sim_state: ts.SimState,
) -> None:
    """Mixed PBC in list form should not use vectorized periodic volume path."""
    mixed_pbc_state = ts.SimState.from_state(si_double_sim_state, pbc=[True, False, True])
    metric_values = calculate_memory_scalers(mixed_pbc_state, "n_atoms_x_density")
    expected_values: list[float] = []
    for split_state in mixed_pbc_state.split():
        bbox = (
            split_state.positions.max(dim=0).values
            - split_state.positions.min(dim=0).values
        ).clone()
        split_state_pbc = torch.as_tensor(split_state.pbc, dtype=torch.bool).tolist()
        for axis_idx, is_periodic in enumerate(split_state_pbc):
            if not is_periodic:
                bbox[axis_idx] += 2.0
        volume = bbox.prod() / 1000
        expected_values.append(
            split_state.n_atoms * (split_state.n_atoms / volume.item())
        )
    assert metric_values == pytest.approx(expected_values, rel=1e-5)


def test_n_edges_scalers_periodic(si_sim_state: ts.SimState) -> None:
    """n_edges scalers for a single periodic system have correct shape and type."""
    result = _n_edges_scalers(si_sim_state, cutoff=5.0)
    assert isinstance(result, list)
    assert len(result) == si_sim_state.n_systems
    assert all(isinstance(v, float) for v in result)
    assert all(v >= 0 for v in result)


def test_n_edges_scalers_non_periodic(benzene_sim_state: ts.SimState) -> None:
    """n_edges scalers for a non-periodic (molecular) system have correct shape/type."""
    result = _n_edges_scalers(benzene_sim_state, cutoff=5.0)
    assert isinstance(result, list)
    assert len(result) == benzene_sim_state.n_systems
    assert all(isinstance(v, float) for v in result)
    assert all(v >= 0 for v in result)


def test_n_edges_scalers_batched(ar_double_sim_state: ts.SimState) -> None:
    """n_edges scalers for a batched state return one value per system."""
    result = _n_edges_scalers(ar_double_sim_state, cutoff=5.0)
    assert isinstance(result, list)
    assert len(result) == ar_double_sim_state.n_systems
    assert all(isinstance(v, float) for v in result)
    assert all(v >= 0 for v in result)


@pytest.mark.parametrize("items", [[], {}])
def test_to_constant_volume_bins_empty_input(
    items: list[Any] | dict[int, float],
) -> None:
    """to_constant_volume_bins returns empty bins for empty list/dict input."""
    # Dict input is part of the public API and used by BinningAutoBatcher.
    bins = to_constant_volume_bins(items, max_volume=1.0)
    assert bins == []


def test_split_state(si_double_sim_state: ts.SimState) -> None:
    """Test splitting a batched state into individual states."""
    split_states = si_double_sim_state.split()

    # Check we get the right number of states
    assert len(split_states) == 2

    # Check each state has the correct properties
    for split_state in split_states:
        assert split_state.n_systems == 1
        assert split_state.system_idx is not None
        assert torch.all(
            split_state.system_idx == 0
        )  # Each split state should have system indices reset to 0
        assert split_state.n_atoms == si_double_sim_state.n_atoms // 2
        assert split_state.positions.shape[0] == si_double_sim_state.n_atoms // 2
        assert split_state.cell.shape[0] == 1


def test_binning_auto_batcher(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with different states."""
    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,  # Set a small value to force multiple batches
    )
    batcher.load_states(states)

    # Check that the batcher correctly identified the metrics
    assert len(batcher.memory_scalers) == 2
    assert batcher.memory_scalers[0] == si_sim_state.n_atoms
    assert batcher.memory_scalers[1] == fe_supercell_sim_state.n_atoms

    # Get batches until None is returned
    batches = [batch for batch, _ in batcher]

    # Check we got the expected number of systems
    assert len(batches) == len(batcher.batched_states)

    # Test restore_original_order
    restored_states = batcher.restore_original_order(batches)
    assert len(restored_states) == len(states)

    # Check that the restored states match the original states in order
    assert restored_states[0].n_atoms == states[0].n_atoms
    assert restored_states[1].n_atoms == states[1].n_atoms

    # Check atomic numbers to verify the correct order
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)


def test_binning_auto_batcher_n_edges(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with n_edges memory metric."""
    states = [si_sim_state, fe_supercell_sim_state]
    cutoff = 5.0

    # Pre-compute scalers to set a meaningful max_memory_scaler
    scalers = [_n_edges_scalers(s, cutoff)[0] for s in states]

    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_edges",
        cutoff=cutoff,
        max_memory_scaler=sum(scalers) + 1,
    )
    batcher.load_states(states)

    assert len(batcher.memory_scalers) == len(states)
    assert all(isinstance(v, float) for v in batcher.memory_scalers)
    assert batcher.memory_scalers == scalers

    batches = [batch for batch, _ in batcher]
    restored_states = batcher.restore_original_order(batches)

    assert len(restored_states) == len(states)
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)


def test_binning_auto_batcher_auto_metric(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test BinningAutoBatcher with different states."""
    # monkeypatch determine max memory scaler
    monkeypatch.setattr(
        "torch_sim.autobatching.determine_max_batch_size",
        lambda *args, **kwargs: 50,  # noqa: ARG005
    )

    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
    )
    batcher.load_states(states)

    # Check that the batcher correctly identified the metrics
    assert len(batcher.memory_scalers) == 2
    assert batcher.memory_scalers[0] == si_sim_state.n_atoms
    assert batcher.memory_scalers[1] == fe_supercell_sim_state.n_atoms

    # Get batches until None is returned
    batches = [batch for batch, _ in batcher]

    # Check we got the expected number of batches
    assert len(batches) == len(batcher.batched_states)

    # Test restore_original_order
    restored_states = batcher.restore_original_order(batches)
    assert len(restored_states) == len(states)

    # Check that the restored states match the original states in order
    assert restored_states[0].n_atoms == states[0].n_atoms
    assert restored_states[1].n_atoms == states[1].n_atoms

    # Check atomic numbers to verify the correct order
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)


def test_binning_auto_batcher_with_indices(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with indices tracking."""
    states = [si_sim_state, fe_supercell_sim_state]

    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,
    )
    batcher.load_states(states)

    # Get batches and track indices manually
    batches_with_indices = []
    for batch, indices in batcher:
        batches_with_indices.append((batch, indices))

    # Check we got the expected number of batches
    assert len(batches_with_indices) == len(batcher.batched_states)

    # Check that the indices match the expected bin indices
    for idx, (_, indices) in enumerate(batches_with_indices):
        assert indices == batcher.index_bins[idx]


def test_binning_auto_batcher_restore_order_with_split_states(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher's restore_original_order method with split states."""
    # Create a list of states with different sizes
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric to avoid GPU memory testing
    batcher = BinningAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260.0,  # Set a small value to force multiple batches
    )
    batcher.load_states(states)

    # loop through all batches to test we're restore order correctly
    batches = []
    for batch, _indices in batcher:
        batches.append(batch)

    # Test restore_original_order with split states
    # This tests the chain.from_iterable functionality
    restored_states = batcher.restore_original_order(batches)

    # Check we got the right number of states back
    assert len(restored_states) == len(states)

    # Check that the restored states match the original states in order
    assert restored_states[0].n_atoms == states[0].n_atoms
    assert restored_states[1].n_atoms == states[1].n_atoms

    # Check atomic numbers to verify the correct order
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)


def test_in_flight_max_metric_too_small(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher with different states."""
    # Create a list of states
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric
    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=1.0,  # Set a small value to force multiple batches
    )
    # Get the first batch
    with pytest.raises(ValueError, match="is greater than max_metric"):
        batcher.load_states(states)


def test_in_flight_auto_batcher(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher with different states."""
    # Create a list of states
    states = [si_sim_state, fe_supercell_sim_state]

    # Initialize the batcher with a fixed max_metric
    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=260,  # Set a small value to force multiple batches
    )
    batcher.load_states(states)

    # Get the first batch
    first_batch, [] = batcher.next_batch(states, None)
    assert isinstance(first_batch, ts.SimState)

    # Create a convergence tensor where the first state has converged
    convergence = torch.tensor([True])

    # Get the next batch
    next_batch, popped_batch = batcher.next_batch(first_batch, convergence)
    assert isinstance(next_batch, ts.SimState)
    assert isinstance(popped_batch, list)
    assert isinstance(popped_batch[0], ts.SimState)

    # Check that the converged state was removed
    assert len(batcher.current_scalers) == 1
    assert len(batcher.current_idx) == 1
    assert len(batcher.completed_idx_og_order) == 1

    # Create a convergence tensor where the remaining state has converged
    convergence = torch.tensor([True])

    # Get the next batch, which should be None since all states have converged
    final_batch, popped_batch = batcher.next_batch(next_batch, convergence)
    assert final_batch is None

    # Check that all states are marked as completed
    assert len(batcher.completed_idx_og_order) == 2


def test_determine_max_batch_size_fibonacci(
    si_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that determine_max_batch_size uses Fibonacci sequence correctly."""

    # Mock measure_model_memory_forward to avoid actual GPU memory testing
    def mock_measure(*_args: Any, **_kwargs: Any) -> float:
        return 0.1  # Return a small constant memory usage

    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", mock_measure
    )

    # Test with a small max_atoms value to limit the sequence
    max_size = determine_max_batch_size(si_sim_state, lj_model, max_atoms=16)
    # The Fibonacci sequence up to 10 is [1, 2, 3, 5, 8, 13]
    # Since we're not triggering OOM errors with our mock, it should return the
    # largest value that fits within max_atoms (simstate has 8 atoms, so 2 batches)
    assert max_size == 2


def test_detach_state_graph_drops_grad_but_keeps_values(
    si_sim_state: ts.SimState,
) -> None:
    """`detach_state_graph` strips grad graphs (the UMA leak) but preserves data.

    Models such as UMA return a graph-carrying ``energy`` (``requires_grad=True``)
    while their forces are detached; accumulating those graph-carrying states for
    the whole run is the memory leak. The helper must detach grad-carrying tensors
    in place, leave non-grad tensors untouched, and not change any values.
    """
    # Give one tensor attribute an autograd graph, as UMA's energy would carry.
    grad_positions = (si_sim_state.positions.detach().clone().requires_grad_()) * 2
    values_before = grad_positions.detach().clone()
    si_sim_state.positions = grad_positions
    masses_before = si_sim_state.masses  # a plain, non-grad tensor
    assert si_sim_state.positions.requires_grad

    returned = detach_state_graph(si_sim_state)

    assert returned is si_sim_state  # detaches in place
    assert not si_sim_state.positions.requires_grad
    assert si_sim_state.positions.grad_fn is None
    assert torch.allclose(si_sim_state.positions, values_before)  # values unchanged
    assert si_sim_state.masses is masses_before  # non-grad tensors left as-is


@pytest.mark.parametrize(
    "oom_message",
    [
        "CUDA out of memory. Tried to allocate 20.00 MiB",
        # Warp / nvalchemiops allocator (used by ORB v3 neighbor lists) phrases
        # OOM differently and must still be recognised by the default matcher.
        "Failed to allocate 2556 bytes on device 'cuda:0'",
    ],
)
def test_determine_max_batch_size_recognises_oom_variants(
    si_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
    oom_message: str,
) -> None:
    """OOM is detected for both PyTorch and Warp-style allocator messages.

    Regression test: the default ``oom_error_message`` must cover the Warp
    allocator wording, and a non-matching first entry in the message list must
    not short-circuit the check before later entries are compared.
    """
    call_count = {"n": 0}

    def mock_measure(*_args: Any, **_kwargs: Any) -> float:
        call_count["n"] += 1
        if call_count["n"] >= 3:  # OOM once the batch grows past a couple probes
            raise RuntimeError(oom_message)
        return 0.1

    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", mock_measure
    )

    # Uses the (broadened) default oom_error_message. Should degrade to a safe
    # batch size instead of propagating the OOM RuntimeError.
    max_size = determine_max_batch_size(si_sim_state, lj_model, max_atoms=10_000)
    assert max_size >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("probe_ooms", [True, False])
def test_determine_max_batch_size_releases_cached_memory(
    *,
    si_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
    probe_ooms: bool,
) -> None:
    """The probe must hand the device back on every exit path.

    Regression test: probing deliberately grows batches until the GPU runs out
    of memory. If PyTorch's caching allocator keeps holding that memory after
    the probe returns, a separate allocator - such as the Warp/cudaMallocAsync
    pool behind the alchemiops neighbor lists - can fail to allocate even a few
    bytes on the very next call.
    """

    def mock_measure(*_args: Any, **_kwargs: Any) -> float:
        # Reserve a chunk and release it, so it lands in PyTorch's cache the
        # way a real probe's activations do.
        buffer = torch.empty(256 * 1024**2, dtype=torch.uint8, device="cuda")
        del buffer
        if probe_ooms:
            raise RuntimeError("CUDA out of memory")
        return 0.1

    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", mock_measure
    )

    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_reserved()

    determine_max_batch_size(si_sim_state, lj_model, max_atoms=10_000)

    assert torch.cuda.memory_reserved() <= baseline


def test_determine_max_batch_size_reraises_non_oom_error(
    si_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine (non-OOM) error is still propagated, not swallowed."""

    def mock_measure(*_args: Any, **_kwargs: Any) -> float:
        raise RuntimeError("shape mismatch in einsum")

    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", mock_measure
    )

    with pytest.raises(RuntimeError, match="shape mismatch"):
        determine_max_batch_size(si_sim_state, lj_model, max_atoms=10_000)


@pytest.mark.parametrize("scale_factor", [1.1, 1.4])
def test_determine_max_batch_size_small_scale_factor_no_infinite_loop(
    si_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    monkeypatch: pytest.MonkeyPatch,
    scale_factor: float,
) -> None:
    """Test determine_max_batch_size doesn't infinite loop with small scale factors."""
    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", lambda *_: 0.1
    )

    max_size = determine_max_batch_size(
        si_sim_state, lj_model, max_atoms=20, scale_factor=scale_factor
    )
    assert 0 < max_size <= 20

    # Verify sequence is strictly increasing (prevents infinite loop)
    sizes = [1]
    while (
        next_size := max(round(sizes[-1] * scale_factor), sizes[-1] + 1)
    ) * si_sim_state.n_atoms <= 20:
        sizes.append(next_size)

    assert all(sizes[idx] > sizes[idx - 1] for idx in range(1, len(sizes)))
    assert max_size == sizes[-1]


def test_in_flight_auto_batcher_restore_order(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher's restore_original_order method."""
    states = [si_sim_state, fe_supercell_sim_state]

    batcher = InFlightAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=260.0
    )
    batcher.load_states(states)

    # Get the first batch
    first_batch, [] = batcher.next_batch(states, None)

    # Simulate convergence of all states
    completed_states_list = []
    convergence = torch.tensor([True])
    next_batch, completed_states = batcher.next_batch(first_batch, convergence)
    completed_states_list.extend(completed_states)

    # sample batch a second time
    # sample batch a second time
    next_batch, completed_states = batcher.next_batch(next_batch, convergence)
    completed_states_list.extend(completed_states)

    # Test restore_original_order
    restored_states = batcher.restore_original_order(completed_states_list)
    assert len(restored_states) == 2

    # Check that the restored states match the original states in order
    assert restored_states[0].n_atoms == states[0].n_atoms
    assert restored_states[1].n_atoms == states[1].n_atoms

    # Check atomic numbers to verify the correct order
    assert torch.all(restored_states[0].atomic_numbers == states[0].atomic_numbers)
    assert torch.all(restored_states[1].atomic_numbers == states[1].atomic_numbers)

    # # Test error when number of states doesn't match
    # with pytest.raises(
    #     ValueError, match="Number of completed states .* does not match"
    # ):
    #     batcher.restore_original_order([si_sim_state])


@pytest.mark.parametrize(
    "num_steps_per_batch",
    [
        5,  # At 5 steps, not every state will converge before the next batch.
        # This tests the merging of partially converged states with new states
        # which has been a bug in the past.
        10,  # At 10 steps, all states will converge before the next batch
    ],
)
def test_in_flight_with_fire(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    num_steps_per_batch: int,
) -> None:
    si_fire_state = ts.fire_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_fire_state = ts.fire_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    fire_states = [si_fire_state, fe_fire_state] * 5
    fire_states = [state.clone() for state in fire_states]
    for state in fire_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        # max_metric=400_000,
        max_memory_scaler=600,
    )
    batcher.load_states(fire_states)

    def convergence_fn(state: ts.FireState) -> torch.Tensor:
        assert state.system_idx is not None
        system_wise_max_force = torch.zeros(
            state.n_systems, device=state.device, dtype=torch.float64
        )
        max_forces = state.forces.norm(dim=1)
        system_wise_max_force = system_wise_max_force.scatter_reduce(
            dim=0, index=state.system_idx, src=max_forces, reduce="amax"
        )
        return system_wise_max_force < 5e-1

    all_completed_states, convergence_tensor = [], None
    while True:
        state, completed_states = batcher.next_batch(state, convergence_tensor)

        all_completed_states.extend(completed_states)
        if state is None:
            break

        for _ in range(num_steps_per_batch):
            state = ts.fire_step(state=state, model=lj_model)
        convergence_tensor = convergence_fn(state)

    assert len(all_completed_states) == len(fire_states)


def test_binning_auto_batcher_with_fire(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    si_fire_state = ts.fire_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_fire_state = ts.fire_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    fire_states = [si_fire_state, fe_fire_state] * 5
    fire_states = [state.clone() for state in fire_states]
    for state in fire_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batch_lengths = [state.n_atoms for state in fire_states]
    optimal_batches = to_constant_volume_bins(batch_lengths, 400)
    optimal_n_systems = len(optimal_batches)

    batcher = BinningAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=400
    )
    batcher.load_states(fire_states)

    finished_states = []
    n_systems = 0
    for batch, _ in batcher:
        n_systems += 1
        for _ in range(5):
            batch = ts.fire_step(state=batch, model=lj_model)

        finished_states.extend(batch.split())

    restored_states = batcher.restore_original_order(finished_states)
    assert len(restored_states) == len(fire_states)
    for restored, original in zip(restored_states, fire_states, strict=True):
        assert torch.all(restored.atomic_numbers == original.atomic_numbers)
    # analytically determined to be optimal
    assert n_systems == optimal_n_systems


def test_in_flight_max_iterations(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test InFlightAutoBatcher with max_iterations limit."""
    # Create states that won't naturally converge
    states = [si_sim_state.clone(), fe_supercell_sim_state.clone()]

    # Set max_iterations to a small value to ensure quick termination
    max_iterations = 3
    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=800.0,
        max_iterations=max_iterations,
    )
    batcher.load_states(states)

    # Get the first batch
    state, [] = batcher.next_batch(None, None)
    assert state is not None

    # Create a convergence tensor that never converges
    convergence_tensor = torch.zeros(state.n_systems, dtype=torch.bool)

    all_completed_states = []
    iteration_count = 0

    # Process batches until complete
    while state is not None:
        iteration_count += 1
        state, completed_states = batcher.next_batch(state, convergence_tensor)
        all_completed_states.extend(completed_states)

        # Update convergence tensor for next iteration (still all False)
        if state is not None:
            convergence_tensor = torch.zeros(state.n_systems, dtype=torch.bool)

        if iteration_count > max_iterations + 4:
            raise ValueError("Should have terminated by now")

    # Verify all states were processed
    assert len(all_completed_states) == len(states)

    # Verify we didn't exceed max_iterations + 1 iterations (first call doesn't count)
    assert iteration_count == 3

    # Verify iteration_count tracking
    for idx in range(len(states)):
        assert batcher.iteration_count[idx] == max_iterations


@pytest.mark.parametrize(
    "num_steps_per_batch",
    [
        5,  # At 5 steps, not every state will converge before the next batch.
        10,  # At 10 steps, all states will converge before the next batch
    ],
)
def test_in_flight_with_bfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    num_steps_per_batch: int,
) -> None:
    """Test InFlightAutoBatcher with BFGS optimizer."""
    si_bfgs_state = ts.bfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_bfgs_state = ts.bfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    bfgs_states = [si_bfgs_state, fe_bfgs_state] * 5
    bfgs_states = [state.clone() for state in bfgs_states]
    for state in bfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=6000,
    )
    batcher.load_states(bfgs_states)

    def convergence_fn(state: ts.BFGSState) -> torch.Tensor:
        assert state.system_idx is not None
        system_wise_max_force = torch.zeros(
            state.n_systems, device=state.device, dtype=torch.float64
        )
        max_forces = state.forces.norm(dim=1)
        system_wise_max_force = system_wise_max_force.scatter_reduce(
            dim=0, index=state.system_idx, src=max_forces, reduce="amax"
        )
        return system_wise_max_force < 5e-1

    all_completed_states, convergence_tensor = [], None
    while True:
        state, completed_states = batcher.next_batch(state, convergence_tensor)

        all_completed_states.extend(completed_states)
        if state is None:
            break

        for _ in range(num_steps_per_batch):
            state = ts.bfgs_step(state=state, model=lj_model)
        convergence_tensor = convergence_fn(state)

    assert len(all_completed_states) == len(bfgs_states)


def test_binning_auto_batcher_with_bfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with BFGS optimizer."""
    si_bfgs_state = ts.bfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_bfgs_state = ts.bfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    bfgs_states = [si_bfgs_state, fe_bfgs_state] * 5
    bfgs_states = [state.clone() for state in bfgs_states]
    for state in bfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = BinningAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=6000
    )
    batcher.load_states(bfgs_states)

    all_finished_states: list[ts.SimState] = []
    total_batches = 0
    for batch, _ in batcher:
        total_batches += 1  # noqa: SIM113
        for _ in range(5):
            batch = ts.bfgs_step(state=batch, model=lj_model)
        all_finished_states.extend(batch.split())

    assert len(all_finished_states) == len(bfgs_states)


@pytest.mark.parametrize(
    "num_steps_per_batch",
    [
        5,  # At 5 steps, not every state will converge before the next batch.
        10,  # At 10 steps, all states will converge before the next batch
    ],
)
def test_in_flight_with_lbfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
    num_steps_per_batch: int,
) -> None:
    """Test InFlightAutoBatcher with L-BFGS optimizer."""
    si_lbfgs_state = ts.lbfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_lbfgs_state = ts.lbfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    lbfgs_states = [si_lbfgs_state, fe_lbfgs_state] * 5
    lbfgs_states = [state.clone() for state in lbfgs_states]
    for state in lbfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = InFlightAutoBatcher(
        model=lj_model,
        memory_scales_with="n_atoms",
        max_memory_scaler=6000,
    )
    batcher.load_states(lbfgs_states)

    def convergence_fn(state: ts.LBFGSState) -> torch.Tensor:
        assert state.system_idx is not None
        system_wise_max_force = torch.zeros(
            state.n_systems, device=state.device, dtype=torch.float64
        )
        max_forces = state.forces.norm(dim=1)
        system_wise_max_force = system_wise_max_force.scatter_reduce(
            dim=0, index=state.system_idx, src=max_forces, reduce="amax"
        )
        return system_wise_max_force < 5e-1

    all_completed_states, convergence_tensor = [], None
    while True:
        state, completed_states = batcher.next_batch(state, convergence_tensor)

        all_completed_states.extend(completed_states)
        if state is None:
            break

        for _ in range(num_steps_per_batch):
            state = ts.lbfgs_step(state=state, model=lj_model)
        convergence_tensor = convergence_fn(state)

    assert len(all_completed_states) == len(lbfgs_states)


def test_binning_auto_batcher_with_lbfgs(
    si_sim_state: ts.SimState,
    fe_supercell_sim_state: ts.SimState,
    lj_model: LennardJonesModel,
) -> None:
    """Test BinningAutoBatcher with L-BFGS optimizer."""
    si_lbfgs_state = ts.lbfgs_init(si_sim_state, lj_model, cell_filter=ts.CellFilter.unit)
    fe_lbfgs_state = ts.lbfgs_init(
        fe_supercell_sim_state, lj_model, cell_filter=ts.CellFilter.unit
    )

    lbfgs_states = [si_lbfgs_state, fe_lbfgs_state] * 5
    lbfgs_states = [state.clone() for state in lbfgs_states]
    for state in lbfgs_states:
        state.positions += torch.randn_like(state.positions) * 0.01

    batcher = BinningAutoBatcher(
        model=lj_model, memory_scales_with="n_atoms", max_memory_scaler=6000
    )
    batcher.load_states(lbfgs_states)

    all_finished_states: list[ts.SimState] = []
    total_batches = 0
    for batch, _ in batcher:
        total_batches += 1  # noqa: SIM113
        for _ in range(5):
            batch = ts.lbfgs_step(state=batch, model=lj_model)
        all_finished_states.extend(batch.split())

    assert len(all_finished_states) == len(lbfgs_states)


def _molecules_state(device: torch.device = DEVICE) -> ts.SimState:
    """Isolated molecules from 3 to 60 atoms, like a mixed molecular library."""
    from ase.build import molecule

    names = ["H2O", "CH4", "C6H6", "CH3CH2OH", "C60", "NH3"]
    return ts.io.atoms_to_state([molecule(n) for n in names], device=device, dtype=DTYPE)


def _mock_measurements(
    monkeypatch: pytest.MonkeyPatch,
    const: float,
    per_atom: float,
    per_edge: float,
    *,
    batch_fixed: float = 0.0,
    free_mib: float = 1000.0,
) -> None:
    """Make per-copy measurements follow an exact affine memory model."""

    def per_copy(state: ts.SimState, *_args: Any) -> tuple[float, float]:
        n_edges = _n_edges_scalers(state, 6.0)[0]
        return const + per_atom * state.n_atoms + per_edge * n_edges, batch_fixed

    monkeypatch.setattr("torch_sim.autobatching._per_copy_cost_mib", per_copy)
    monkeypatch.setattr("torch_sim.autobatching._free_device_mib", lambda _: free_mib)


def test_nonnegative_least_squares_recovers_and_clamps() -> None:
    design = torch.tensor(
        [[1.0, 3, 6], [1, 12, 120], [1, 9, 72], [1, 60, 1800]], dtype=torch.float64
    )
    exact = torch.tensor([30.0, 1.0, 0.2], dtype=torch.float64)
    torch.testing.assert_close(_nonnegative_least_squares(design, design @ exact), exact)

    negative = torch.tensor([30.0, -1.0, 0.2], dtype=torch.float64)
    assert (_nonnegative_least_squares(design, design @ negative) >= 0).all()


def test_fit_memory_model_recovers_affine_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_measurements(monkeypatch, 26.0, 1.0, 0.2, batch_fixed=50.0)
    memory_model, budget = fit_memory_model(
        _molecules_state(), model=None, max_memory_padding=0.8
    )
    assert memory_model.const == pytest.approx(26.0, rel=1e-6)
    assert memory_model.per_atom == pytest.approx(1.0, rel=1e-6)
    assert memory_model.per_edge == pytest.approx(0.2, rel=1e-6)
    assert budget == pytest.approx((1000.0 - 50.0) * 0.8)


def test_fit_memory_model_never_undershoots_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Costs no affine model fits exactly must still be covered at every reference."""
    state = _molecules_state()

    def per_copy(ref: ts.SimState, *_args: Any) -> tuple[float, float]:
        return 5.0 * ref.n_atoms**1.5, 0.0  # superlinear, not affine

    monkeypatch.setattr("torch_sim.autobatching._per_copy_cost_mib", per_copy)
    monkeypatch.setattr("torch_sim.autobatching._free_device_mib", lambda _: 1e6)
    memory_model, _ = fit_memory_model(state, model=None)

    predicted = memory_model.predict(state)
    for idx in _reference_systems(
        state.n_atoms_per_system.double(),
        torch.tensor(_n_edges_scalers(state, 6.0), dtype=torch.float64),
    ):
        assert predicted[idx] >= 5.0 * int(state.n_atoms_per_system[idx]) ** 1.5 - 1e-6


def test_binning_measured_fits_small_and_large_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a large per-system constant must not cap the budget below a system.

    With a proportional metric, the smallest system's probe made every edge look
    expensive and set max_memory_scaler below the largest system's metric.
    """
    _mock_measurements(monkeypatch, 30.0, 1.0, 0.2, free_mib=2000.0)
    state = _molecules_state()
    batcher = BinningAutoBatcher(model=None, memory_scales_with="measured")
    budget = batcher.load_states(state)

    assert isinstance(batcher.memory_scales_with, MemoryModel)
    assert max(batcher.memory_scalers) <= budget
    for index_bin in batcher.index_bins:
        assert sum(batcher.memory_scalers[i] for i in index_bin) <= budget


def test_measured_keeps_user_max_memory_scaler(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_measurements(monkeypatch, 30.0, 1.0, 0.2, free_mib=2000.0)
    batcher = BinningAutoBatcher(
        model=None, memory_scales_with="measured", max_memory_scaler=1500.0
    )
    assert batcher.load_states(_molecules_state()) == 1500.0


def test_in_flight_measured_rejects_iterator() -> None:
    batcher = InFlightAutoBatcher(model=None, memory_scales_with="measured")
    with pytest.raises(ValueError, match="up front"):
        batcher.load_states(iter(_molecules_state().split()))


def test_unfitted_measured_metric_raises(benzene_sim_state: ts.SimState) -> None:
    with pytest.raises(ValueError, match="fitted first"):
        calculate_memory_scalers(benzene_sim_state, memory_scales_with="measured")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_fit_memory_model_on_device() -> None:
    """Smoke test of the real measurement path."""
    state = _molecules_state(torch.device("cuda"))
    lj_model = LennardJonesModel(
        sigma=3.405, epsilon=0.0104, device=torch.device("cuda"), dtype=DTYPE
    )
    memory_model, budget = fit_memory_model(state, lj_model, max_memory_padding=0.5)
    assert min(memory_model.const, memory_model.per_atom, memory_model.per_edge) >= 0
    assert 0 < budget < torch.cuda.mem_get_info()[1] / 1024**2
    assert all(cost > 0 for cost in memory_model.predict(state))


@pytest.mark.parametrize("oom_above_mib", [None, 3000.0])
def test_per_copy_cost_resolves_small_slope_under_large_fixed_cost(
    monkeypatch: pytest.MonkeyPatch, oom_above_mib: float | None
) -> None:
    """Regression: a large fixed batch cost hid a tiny system's per-copy cost.

    Sizing the second batch from one copy's total picked ~3 copies, so a 25 MiB
    per-copy cost under ~1 GiB of fixed cost was measured as noise.
    """
    fixed, per_copy = 1000.0, 25.0
    calls = []

    def fake_measure(state: ts.SimState, _model: Any) -> float:
        calls.append(state.n_systems)
        jitter = 10.0 if len(calls) % 2 else -10.0  # allocator noise
        mib = fixed + per_copy * state.n_systems + jitter
        if oom_above_mib is not None and mib > oom_above_mib:
            raise RuntimeError("CUDA out of memory")
        return mib / 1024

    monkeypatch.setattr(
        "torch_sim.autobatching.measure_model_memory_forward", fake_measure
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    state = _molecules_state()[0]
    measured, batch_fixed = _per_copy_cost_mib(
        state, None, 14_000.0, 100_000, ["CUDA out of memory"]
    )
    assert measured == pytest.approx(per_copy, rel=0.05)
    assert batch_fixed == pytest.approx(fixed, rel=0.05)
