import numpy as np
import pandas as pd
import pytest

from ovarian import stage3, stats

SIZE = 64


def gradient_image(size: int = SIZE) -> np.ndarray:
    column = np.linspace(0, 255, size, dtype=np.uint8)
    return np.repeat(np.tile(column, (size, 1))[:, :, None], 3, axis=2)


def predictions(y_pred, p_malignant, patients, y_true, split="test") -> pd.DataFrame:
    return pd.DataFrame({"image": [f"{i}.png" for i in range(len(y_pred))], "class": "benign",
                         "patient": patients, "split": split,
                         "evaluation_order": np.arange(1, len(y_pred) + 1),
                         "y_true": y_true, "y_pred": y_pred, "p_malignant": p_malignant})


def test_crop_uses_the_box_and_falls_back_to_the_full_frame():
    image = gradient_image()
    whole = stage3.crop(image, None, size=16)
    boxed = stage3.crop(image, (0, 0, SIZE // 2, SIZE), size=16)
    assert whole.shape == boxed.shape == (16, 16, 3)
    assert boxed.mean() < whole.mean()
    assert np.array_equal(stage3.crop(image, (10, 10, 10, 10), size=16), whole)


def test_tied_vote_is_malignant_at_stage_three_and_first_image_elsewhere():
    images = pd.DataFrame({"patient": ["a", "a"], "y_pred": [0, 1]})
    assert stats.patient_vote(images).loc[0, "y_pred_majority"] == 0
    assert stats.patient_vote(images, tie_to_positive=True).loc[0, "y_pred_majority"] == 1

    frame = predictions(y_pred=[0, 1], p_malignant=[0.2, 0.8], patients=["a", "a"], y_true=[1, 1])
    votes = stage3.patient_majority(frame)
    assert bool(votes.loc[0, "vote_tied"]) and votes.loc[0, "y_pred_majority"] == 1
    assert votes.loc[0, "p_malignant_mean"] == pytest.approx(0.5)
    assert list(votes.columns) == stage3.PATIENT_COLUMNS


def test_descriptor_normalization_uses_the_released_statistics(tmp_path):
    names = ["area", "perimeter"]
    (tmp_path / "norm.json").write_text(
        '{"feature_names": ["area", "perimeter"], "means": {"area": 10, "perimeter": 4},'
        ' "stds": {"area": 2, "perimeter": 0.5}}')
    loaded, mean, std = stage3.load_normalization(tmp_path / "norm.json")
    assert loaded == names and list(mean) == [10, 4] and list(std) == [2, 0.5]

    frame = pd.DataFrame({"image": ["a.png"], "class": ["benign"], "patient": ["a"], "split": ["test"],
                          "label": [0]})
    features = pd.DataFrame({"area": [14.0], "perimeter": [3.0]}, index=["a.png"])
    dataset = stage3.Stage3Dataset(frame, boxes={}, features=features, mean=mean, std=std)
    assert dataset.tabular[0] == pytest.approx([2.0, -2.0])

    with pytest.raises(KeyError):
        stage3.Stage3Dataset(frame, boxes={}, features=features.rename(index={"a.png": "b.png"}),
                             mean=mean, std=std)


def test_released_boxes_map_a_fallback_to_no_box(tmp_path):
    (tmp_path / "b.json").write_text(
        '{"a.png": {"box": [1, 2, 3, 4], "confidence": 0.9, "full_frame_fallback": false},'
        ' "b.png": {"box": [0, 0, 8, 8], "confidence": null, "full_frame_fallback": true}}')
    boxes = stage3.load_boxes(tmp_path / "b.json")
    assert boxes["a.png"] == (1, 2, 3, 4) and boxes["b.png"] is None


def test_aggregation_rules_cover_every_rule_and_fall_back_to_majority():
    frame = predictions(y_pred=[1, 1, 0, 0], p_malignant=[0.90, 0.80, 0.10, 0.20],
                        patients=["a", "a", "b", "b"], y_true=[1, 1, 0, 0])
    rules = stage3.aggregation_rules(frame)["test"]
    assert list(rules) == ["Majority vote", "Mean probability", "Max probability",
                           "Weighted vote >=0.50", "Weighted vote >=0.75", "Weighted vote >=0.85",
                           "Weighted vote >=0.95"]
    assert rules["Majority vote"]["accuracy"] == pytest.approx(1.0)
    assert rules["Max probability"]["accuracy"] == pytest.approx(1.0)
    # No image of either patient reaches 0.95 confidence, so the rule falls back to the majority vote.
    assert rules["Weighted vote >=0.95"] == rules["Majority vote"]


def test_training_config_carries_the_reported_settings():
    six = stage3.TrainConfig.for_configuration(6)
    two = stage3.TrainConfig.for_configuration(2)
    assert (six.learning_rate, six.batch_size) == (1e-5, 4)
    assert (two.learning_rate, two.batch_size) == (1e-4, 8)
    assert six.epochs == two.epochs == 250
    assert six.weight_decay == 0.0
    assert stage3.TrainConfig.for_configuration(6, epochs=3, batch_size=1).epochs == 3


def test_descriptor_files_are_required_unless_a_fallback_is_asked_for(tmp_path, monkeypatch):
    names = ["area", "perimeter"]
    present = pd.DataFrame({"image": ["a.png"], "area": [1.0], "perimeter": [2.0]})
    present.to_csv(tmp_path / "automatic_morphological_features_val.csv", index=False)
    paths_by_split = {"val": tmp_path / "automatic_morphological_features_val.csv",
                      "train": tmp_path / "automatic_morphological_features_train.csv"}

    with pytest.raises(FileNotFoundError):
        stage3.load_features(paths_by_split, names)

    released = pd.DataFrame({"image": ["a.png", "b.png"], "area": [9.0, 3.0], "perimeter": [9.0, 4.0]})
    monkeypatch.setattr(stage3.data, "descriptors", lambda: released)
    table = stage3.load_features(paths_by_split, names, fallback=True)
    assert sorted(table.index) == ["a.png", "b.png"]
    # The automatic file wins where it covers an image; the released table only fills the rest.
    assert table.loc["a.png", "area"] == 1.0 and table.loc["b.png", "area"] == 3.0


def test_split_metrics_reports_both_levels():
    frame = predictions(y_pred=[1, 1, 0, 0], p_malignant=[0.9, 0.8, 0.1, 0.2],
                        patients=["a", "a", "b", "b"], y_true=[1, 1, 0, 0])
    m = stage3.split_metrics(frame)["test"]
    assert m["n_images"] == 4 and m["n_patients"] == 2
    assert m["image_accuracy"] == pytest.approx(1.0) and m["patient_accuracy"] == pytest.approx(1.0)
    assert m["image_auc"] == pytest.approx(1.0) and m["patient_auc"] == pytest.approx(1.0)
    assert m["patient_confusion_matrix"] == [[1, 0], [0, 1]]
