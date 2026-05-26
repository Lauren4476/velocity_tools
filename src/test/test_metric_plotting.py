import numpy as np
import jax.numpy as jnp

from velocity_tools import extract_streamline


def test_metric_partitions_match_distance_metric_percentiles() -> None:
    ra = jnp.array([3.0, 2.0, 1.0, 0.5, 0.25], dtype=jnp.float64)
    dec = jnp.zeros_like(ra)
    pc_coords = np.vstack([np.asarray(ra), np.asarray(dec)])

    partitions = extract_streamline.get_metric_partitions(pc_coords, 4)
    distance_metric = np.asarray(extract_streamline.get_distance_metric(ra, dec))
    expected = np.asarray([np.percentile(distance_metric, per) for per in np.linspace(0.0, 100.0, 5)])

    assert partitions.shape == (5,)
    assert np.allclose(partitions, expected)


def test_metric_boundary_sampling_varies_with_angle() -> None:
    ra = jnp.array([3.0, 2.0, 1.0, 0.5, 0.25], dtype=jnp.float64)
    dec = jnp.zeros_like(ra)
    pc_coords = np.vstack([np.asarray(ra), np.asarray(dec)])
    partitions = extract_streamline.get_metric_partitions(pc_coords, 4)

    curves, trace = extract_streamline.sample_metric_boundaries(pc_coords, partitions[1:2], n_samples=256)
    ra_curve, dec_curve = curves[0]
    radii = np.sqrt(ra_curve**2 + dec_curve**2)

    assert ra_curve.shape == (256,)
    assert dec_curve.shape == (256,)
    assert radii.max() > radii.min()
    assert trace["theta_weight"] == 1.0
