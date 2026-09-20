import math

import pytest
import torch

from torch_pointcloud.metrics import nuscenes_detection_metrics
from torch_pointcloud.metrics.nuscenes import filter_boxes_by_range, nuscenes_velocity_attributes


def _box(x: float, y: float, yaw: float = 0.0) -> list[float]:
    return [x, y, 0.0, 4.0, 2.0, 1.5, yaw]


def test_nuscenes_detection_metrics_ap_distinct_per_threshold() -> None:
    r"""Offsets 0.7 / 1.5 / 3.0 m pass 1, 2 and 3 of the four matching thresholds (strict `<`).

    Car (3 GT): at $d{=}0.5$ nothing matches (AP 0). At $d{=}1$ the cumulative recall is $[1/3, 1/3, 1/3]$,
    so the interpolated precision is 1 up to grid index 33 and 0 after: AP $= 23 \cdot 0.9 / (90 \cdot 0.9)
    = 23/90$. At $d{=}2$ recall reaches $2/3$ (ones up to index 66, AP $56/90$); at $d{=}4$ all three match
    (AP 1). Pedestrian: one exact prediction, AP 1 at every threshold.
    """
    pred_boxes = torch.tensor([_box(0.7, 0.0), _box(10.0, 1.5), _box(23.0, 0.0), _box(0.0, 30.0)])
    pred_scores = torch.tensor([0.9, 0.8, 0.7, 0.95])
    pred_labels = torch.tensor([0, 0, 0, 1])
    gt_boxes = torch.tensor([_box(0.0, 0.0), _box(10.0, 0.0), _box(20.0, 0.0), _box(0.0, 30.0)])
    gt_labels = torch.tensor([0, 0, 0, 1])
    batch = torch.zeros(4, dtype=torch.long)
    args = (pred_boxes, pred_scores, pred_labels, batch, gt_boxes, gt_labels, batch)
    names = ("car", "pedestrian")
    for threshold, ap in {0.5: 0.0, 1.0: 23.0 / 90.0, 2.0: 56.0 / 90.0, 4.0: 1.0}.items():
        out = nuscenes_detection_metrics(*args, class_names=names, dist_thresholds=(threshold,))
        assert out["AP/car"] == pytest.approx(ap)
        assert out["AP/pedestrian"] == pytest.approx(1.0)
    out = nuscenes_detection_metrics(*args, class_names=names)
    assert out["AP/car"] == pytest.approx(169.0 / 360.0)
    assert out["mAP"] == pytest.approx(529.0 / 720.0)


def test_nuscenes_detection_metrics_101_point_interpolation_hand_derived() -> None:
    r"""AP from the 101-point clipping formula on a curve with an interpolated precision ramp.

    Matches in score order are TP, FP, TP, TP over 3 GT: recall $[1/3, 1/3, 2/3, 1]$, precision
    $[1, 1/2, 2/3, 3/4]$. Interpolated at $r_i = i/100$: 1 for $i \le 33$, $1/2 + (r_i - 1/3)/2$ for
    $34 \le i \le 66$ and $2/3 + (r_i - 2/3)/4$ for $67 \le i \le 100$. Dropping $i \le 10$, subtracting
    $0.1$ and clamping at 0 sums to $23 \cdot 0.9 + \sum_{34}^{66} (7/30 + i/200) + \sum_{67}^{100}
    (2/5 + i/400) = 20.7 + 15.95 + 20.6975$, so AP $= 57.3475 / (90 \cdot 0.9) = 22939/32400$.
    """
    pred_boxes = torch.tensor([_box(0.0, 0.0), _box(35.0, 0.0), _box(10.5, 0.0), _box(20.0, 1.0)])
    pred_scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    gt_boxes = torch.tensor([_box(0.0, 0.0), _box(10.0, 0.0), _box(20.0, 0.0)])
    out = nuscenes_detection_metrics(
        pred_boxes,
        pred_scores,
        torch.zeros(4, dtype=torch.long),
        torch.zeros(4, dtype=torch.long),
        gt_boxes,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        class_names=("car",),
        dist_thresholds=(2.0,),
    )
    assert out["AP/car"] == pytest.approx(22939.0 / 32400.0)


def test_nuscenes_detection_metrics_greedy_closest_consumes_gt() -> None:
    """The top-scoring prediction takes its closest GT even when a globally better assignment exists.

    The 0.9 prediction lies 1.2 m from GT A and 1.8 m from GT B, both under the 2 m threshold; the 0.8
    prediction is 0.5 m from A but 3.5 m from B. Pairing the first with B would match both predictions,
    but the official greedy rule matches it to the closest GT (A), leaving the second unmatched: one TP
    out of three GT (AP 23/90) with ATE 1.2 from the consumed closest match, not 1.8.
    """
    pred_boxes = torch.tensor([_box(1.2, 0.0), _box(-0.5, 0.0)])
    gt_boxes = torch.tensor([_box(0.0, 0.0), _box(3.0, 0.0), _box(40.0, 0.0)])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9, 0.8]),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        gt_boxes,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
        class_names=("car",),
        dist_thresholds=(2.0,),
    )
    assert out["AP/car"] == pytest.approx(23.0 / 90.0)
    assert out["mATE"] == pytest.approx(1.2)


def test_nuscenes_detection_metrics_tp_errors_hand_values() -> None:
    """ATE is the BEV distance (z ignored), ASE the size-aligned 1 - IoU, AOE the absolute yaw difference.

    The prediction sits exactly 1.0 m from the GT center: it fails the 0.5 m and (strictly) the 1.0 m
    thresholds and matches at 2 m and 4 m, so AP/car = 0.5. On the 2 m match: ATE = 1.0; the size-aligned
    IoU of (4, 2, 1.5) vs (2, 2, 1.5) is 6/12, so ASE = 0.5; AOE = |0.3 - (-0.2)| = 0.5. Velocity and
    attributes are absent, so AVE and AAE take the full 1.0 penalty, and
    NDS = (5 * 0.5 + 0 + 0.5 + 0.5 + 0 + 0) / 10 = 0.35.
    """
    gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.3]])
    pred_boxes = torch.tensor([[1.0, 0.0, 0.5, 2.0, 2.0, 1.5, -0.2]])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        pred_boxes, torch.tensor([0.9]), zero, zero, gt_boxes, zero, zero, class_names=("car",)
    )
    assert out["AP/car"] == pytest.approx(0.5)
    assert out["mATE"] == pytest.approx(1.0)
    assert out["mASE"] == pytest.approx(0.5)
    assert out["mAOE"] == pytest.approx(0.5)
    assert out["mAVE"] == 1.0
    assert out["mAAE"] == 1.0
    assert out["NDS"] == pytest.approx(0.35)


def test_nuscenes_detection_metrics_barrier_orientation_modulo_pi() -> None:
    """A barrier rotated by pi - 0.3 scores AOE 0.3 (period pi); any other class scores pi - 0.3.

    The same matched pair carries equal velocities and attributes: the car measures AVE and AAE of 0,
    while the barrier excludes both, and with no contributing class they report the full 1.0 penalty.
    """
    gt_boxes = torch.tensor([_box(0.0, 0.0, yaw=0.0) + [1.0, 1.0]])
    pred_boxes = torch.tensor([_box(0.0, 0.0, yaw=math.pi - 0.3) + [1.0, 1.0]])
    zero = torch.tensor([0])
    attributes = torch.tensor([2])
    for name, aoe in (("barrier", 0.3), ("car", math.pi - 0.3)):
        out = nuscenes_detection_metrics(
            pred_boxes,
            torch.tensor([0.9]),
            zero,
            zero,
            gt_boxes,
            zero,
            zero,
            class_names=(name,),
            pred_attributes=attributes,
            gt_attributes=attributes,
        )
        assert out["mAOE"] == pytest.approx(aoe, abs=1e-5)
        expected_penalty = 1.0 if name == "barrier" else 0.0
        assert out["mAVE"] == pytest.approx(expected_penalty)
        assert out["mAAE"] == pytest.approx(expected_penalty)


def test_filter_boxes_by_range_strict_bev_distance() -> None:
    """The class-range filter uses the BEV distance from the origin with a strict inequality."""
    boxes = torch.tensor([_box(30.0, 40.0), _box(3.0, 4.0), _box(0.0, -39.0), _box(0.0, 41.0)])
    labels = torch.tensor([0, 0, 1, 1])
    mask = filter_boxes_by_range(boxes, labels, ranges=[50.0, 40.0])
    assert torch.equal(mask, torch.tensor([False, True, True, False]))


def test_nuscenes_detection_metrics_range_and_num_points_filters() -> None:
    """Out-of-range boxes (both sides) and zero-point GT boxes are dropped before matching.

    After filtering, a single in-range TP remains against a single GT (AP 1). Without `gt_num_points`
    the zero-point GT stays: recall stops at 1/2, so the interpolated precision is 1 up to grid index 50
    and AP drops to 40/90 = 4/9.
    """
    pred_boxes = torch.tensor([_box(10.0, 0.0), _box(60.0, 0.0)])
    pred_scores = torch.tensor([0.9, 0.95])
    gt_boxes = torch.tensor([_box(10.0, 0.0), _box(60.0, 0.0), _box(20.0, 0.0)])
    args = (
        pred_boxes,
        pred_scores,
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        gt_boxes,
        torch.zeros(3, dtype=torch.long),
        torch.zeros(3, dtype=torch.long),
    )
    out = nuscenes_detection_metrics(*args, class_names=("car",), gt_num_points=torch.tensor([5, 7, 0]))
    assert out["AP/car"] == pytest.approx(1.0)
    assert out["mATE"] == pytest.approx(0.0, abs=1e-7)
    out = nuscenes_detection_metrics(*args, class_names=("car",))
    assert out["AP/car"] == pytest.approx(4.0 / 9.0)


def test_nuscenes_detection_metrics_max_boxes_per_sample_cap() -> None:
    """The prediction cap keeps the highest-scoring boxes and applies per sample, not globally."""
    # Sample 0: an FP outscores the TP; a cap of 1 leaves only the FP, so nothing matches.
    pred_boxes = torch.tensor([_box(30.0, 0.0), _box(0.0, 0.0)])
    gt_boxes = torch.tensor([_box(0.0, 0.0)])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9, 0.8]),
        torch.zeros(2, dtype=torch.long),
        torch.zeros(2, dtype=torch.long),
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        max_boxes_per_sample=1,
    )
    assert out["AP/car"] == pytest.approx(0.0)
    # Two samples with one exact TP each survive a per-sample cap of 1 untouched.
    pred_boxes = torch.tensor([_box(0.0, 0.0), _box(5.0, 5.0)])
    gt_boxes = pred_boxes.clone()
    batch = torch.tensor([0, 1])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9, 0.8]),
        torch.zeros(2, dtype=torch.long),
        batch,
        gt_boxes,
        torch.zeros(2, dtype=torch.long),
        batch,
        class_names=("car",),
        max_boxes_per_sample=1,
    )
    assert out["AP/car"] == pytest.approx(1.0)


def test_nuscenes_detection_metrics_nds_identity() -> None:
    """NDS = (5 * mAP + sum of clipped TP scores) / 10; a perfect prediction reaches exactly 1.0.

    With an exact match carrying equal velocity and attribute, every error is 0 and NDS = (5 + 5) / 10.
    A 2 m/s velocity gap and a wrong attribute keep mAP = 1 but zero the clipped AVE and AAE scores:
    NDS = (5 + 3) / 10 = 0.8.
    """
    gt_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.5, 1.0, 0.0]])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        gt_boxes.clone(),
        torch.tensor([0.9]),
        zero,
        zero,
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        pred_attributes=torch.tensor([1]),
        gt_attributes=torch.tensor([1]),
    )
    assert out["mAVE"] == pytest.approx(0.0)
    assert out["mAAE"] == pytest.approx(0.0)
    assert out["NDS"] == pytest.approx(1.0)
    pred_boxes = torch.tensor([[0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.5, 1.0, 2.0]])
    out = nuscenes_detection_metrics(
        pred_boxes,
        torch.tensor([0.9]),
        zero,
        zero,
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        pred_attributes=torch.tensor([2]),
        gt_attributes=torch.tensor([1]),
    )
    assert out["mAVE"] == pytest.approx(2.0)
    assert out["mAAE"] == pytest.approx(1.0)
    assert out["NDS"] == pytest.approx(0.8)


def test_nuscenes_detection_metrics_void_attribute_and_missing_velocity_penalty() -> None:
    """A negative GT attribute id and 7-column boxes leave nothing measured: AVE and AAE fall back to 1."""
    gt_boxes = torch.tensor([_box(0.0, 0.0)])
    zero = torch.tensor([0])
    out = nuscenes_detection_metrics(
        gt_boxes.clone(),
        torch.tensor([0.9]),
        zero,
        zero,
        gt_boxes,
        zero,
        zero,
        class_names=("car",),
        pred_attributes=torch.tensor([3]),
        gt_attributes=torch.tensor([-1]),
    )
    assert out["mAVE"] == 1.0
    assert out["mAAE"] == 1.0
    assert out["NDS"] == pytest.approx(0.8)


def test_nuscenes_velocity_attributes_hand_derived_case() -> None:
    """Hand-derived against the official tables: moving above 1 m/s BEV speed (strictly), the parked /
    stopped / standing default at or below it, and no attribute (-1) for `barrier` / `traffic_cone`."""
    class_names = (
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    )
    labels = torch.tensor([0, 0, 3, 8, 8, 7, 5, 9])
    velocity = torch.tensor(
        [
            [3.0, 4.0],
            [1.0, 0.0],
            [0.3, 0.0],
            [0.8, 0.9],
            [0.5, 0.5],
            [0.0, 0.0],
            [9.0, 0.0],
            [9.0, 0.0],
        ],
    )
    out = nuscenes_velocity_attributes(labels, velocity, class_names=class_names)
    # car@5 -> vehicle.moving (0); car@1.0 is not strictly moving -> vehicle.parked (2); bus@0.3 ->
    # vehicle.stopped (1); pedestrian@1.2 -> pedestrian.moving (7); pedestrian@0.7 -> pedestrian.standing
    # (6); bicycle@0 -> cycle.without_rider (4); barrier / traffic_cone -> -1.
    assert out.dtype == torch.long
    assert out.tolist() == [0, 2, 1, 7, 6, 4, -1, -1]


def test_nuscenes_velocity_attributes_speed_threshold() -> None:
    """Lowering `speed_threshold` flips a borderline box to its class's moving attribute."""
    labels = torch.tensor([0])
    velocity = torch.tensor([[1.0, 0.0]])
    assert nuscenes_velocity_attributes(labels, velocity, class_names=("car",)).tolist() == [2]
    assert nuscenes_velocity_attributes(labels, velocity, class_names=("car",), speed_threshold=0.5).tolist() == [0]
