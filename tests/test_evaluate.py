import numpy as np

from gtauav_loc.evaluate import build_ground_truth_name_set, parse_tile_coordinates, score_sift_ransac


def test_parse_tile_coordinates_uses_last_two_tokens() -> None:
    assert parse_tile_coordinates("4_0_10_9") == (10.0, 9.0)


def test_build_ground_truth_name_set_uses_positive_names() -> None:
    meta = {"positive_names": ["a/b/4_0_1_1.png", "c/d/4_0_2_2.png"]}
    assert build_ground_truth_name_set(meta) == {"4_0_1_1", "4_0_2_2"}


def test_score_sift_ransac_finds_inliers_for_translated_points() -> None:
    query_keypoints = np.array(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 10.0],
            [5.0, 2.0],
            [2.0, 5.0],
        ],
        dtype=np.float32,
    )
    tile_keypoints = query_keypoints + np.array([3.0, 7.0], dtype=np.float32)
    query_desc = np.eye(len(query_keypoints), dtype=np.float32)
    tile_desc = query_desc.copy()

    inliers, confidence = score_sift_ransac(query_keypoints, query_desc, tile_keypoints, tile_desc)

    assert inliers >= 4
    assert confidence > 0.5
