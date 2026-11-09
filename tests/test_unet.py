"""U-Net: форма ответа, сохранение и загрузка, честная оценка и выбор активной модели."""
import numpy as np

from cloudai.net import CloudNet, load_active_model
from cloudai.training import LABEL_CLEAR, LABEL_CLOUD
from cloudai.unet import CloudUNet, UNet, evaluate_maps


def test_analyze_returns_same_keys_as_cloud_net_for_odd_sizes():
    model = CloudUNet(UNet(base=4))
    rgb = np.random.default_rng(0).random((37, 53, 3)).astype(np.float32)
    result = model.analyze(rgb)
    assert result["probability"].shape == (37, 53)
    assert result["mask"].shape == (37, 53) and result["mask"].dtype == bool
    assert set(result) == {"probability", "mask", "valid", "cloud_percent", "valid_fraction"}


def test_save_and_load_give_the_same_probabilities(tmp_path):
    model = CloudUNet(UNet(base=4), info={"model": "U-Net", "promoted": True})
    rgb = np.random.default_rng(1).random((40, 48, 3)).astype(np.float32)
    model.save(tmp_path / "unet.pt")
    loaded = CloudUNet.load(tmp_path / "unet.pt")
    assert np.allclose(model.predict_map(rgb), loaded.predict_map(rgb))
    assert loaded.info["model"] == "U-Net"


def test_evaluate_maps_is_perfect_for_correct_answers_and_zero_for_inverted():
    labels = np.array([[LABEL_CLOUD, LABEL_CLEAR], [0, LABEL_CLOUD]], dtype=np.uint8)
    truth_map = (labels == LABEL_CLOUD).astype(np.float32)
    right = evaluate_maps(lambda rgb: truth_map, [(np.zeros((2, 2, 3)), labels)])
    wrong = evaluate_maps(lambda rgb: 1.0 - truth_map, [(np.zeros((2, 2, 3)), labels)])
    assert right["cloud_iou"] == 1.0 and right["pixels"] == 3
    assert wrong["cloud_iou"] == 0.0


def test_active_model_is_unet_only_when_its_file_exists(tmp_path):
    npz = tmp_path / "cloud_net.npz"
    CloudNet.load().save(npz)
    assert isinstance(load_active_model(npz, tmp_path / "cloud_unet.pt"), CloudNet)
    CloudUNet(UNet(base=4)).save(tmp_path / "cloud_unet.pt")
    assert isinstance(load_active_model(npz, tmp_path / "cloud_unet.pt"), CloudUNet)
