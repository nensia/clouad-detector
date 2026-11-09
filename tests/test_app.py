"""Веб-интерфейс: запросы к серверу. Все папки подменяются на временные."""
import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as webapp
from cloudai.net import CloudNet
from cloudai.synth import make_scene


def scene_bytes(coverage: float, seed: int) -> bytes:
    scene = make_scene(seed, size=256, coverage=coverage, kind="побережье", nodata=False, cloud_style="слоистые")
    buf = io.BytesIO()
    Image.fromarray((scene["rgb"] * 255).astype(np.uint8)).save(buf, "JPEG", quality=92)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch, model):
    folders = webapp.Folders(tmp_path / "data", tmp_path / "reports", tmp_path / "models")
    model.save(folders.model)                       # копия модели во временной папке
    monkeypatch.setattr(webapp, "folders", folders)
    monkeypatch.setitem(webapp.state, "model", None)
    with TestClient(webapp.app) as c:
        c.folders = folders
        yield c


def test_page_and_status(client):
    page = client.get("/")
    assert page.status_code == 200 and "Облачный контроль" in page.text
    status = client.get("/api/status").json()
    assert status["model_ready"] is True and status["default_threshold"] == 75.0
    assert client.get("/static/app.js").status_code == 200


def test_upload_and_process(client):
    files = [("files", ("clear.jpg", scene_bytes(0.1, 1), "image/jpeg")),
             ("files", ("cloudy.jpg", scene_bytes(0.95, 2), "image/jpeg"))]
    data = client.post("/api/process", files=files, data={"threshold": "75", "delete_mode": "permanent"}).json()
    assert data["counts"] == {"total": 2, "saved": 1, "deleted": 1, "errors": 0}
    by_name = {r["name"]: r for r in data["results"]}
    assert by_name["clear.jpg"]["status"] == "saved" and by_name["cloudy.jpg"]["status"] == "deleted"
    assert "clear.jpg" in data["message"] and "cloudy.jpg" in data["message"]
    assert (client.folders.saved / "clear.jpg").exists()
    assert not any(client.folders.incoming.iterdir())                       # облачный стёрт, ясный перенесён
    assert client.get(f"/previews/{data['run_id']}/{by_name['cloudy.jpg']['overlay']}").status_code == 200
    report = client.get(f"/api/report/{data['run_id']}?format=txt")
    assert report.status_code == 200 and "УДАЛЁННЫЕ ФАЙЛЫ" in report.text


def test_process_existing_folder_with_trash(client):
    (client.folders.incoming / "cloudy.jpg").write_bytes(scene_bytes(0.97, 3))
    assert client.get("/api/status").json()["incoming_waiting"] == ["cloudy.jpg"]
    data = client.post("/api/process-folder", json={"threshold": 75, "delete_mode": "trash"}).json()
    assert data["results"][0]["moved_to_trash"] is True
    assert (client.folders.trash / "cloudy.jpg").exists()


def test_bad_requests_are_refused(client):
    r = client.post("/api/process", files=[("files", ("notes.txt", b"text", "text/plain"))])
    assert r.status_code == 400 and "не снимок" in r.json()["detail"]
    assert not any(client.folders.incoming.iterdir())                       # ничего не сохранили
    assert client.post("/api/process-folder", json={"threshold": 140}).status_code == 400
    assert client.post("/api/process-folder", json={"delete_mode": "burn"}).status_code == 400
    assert client.get("/api/report/..%2F..%2Fsecret").status_code in (400, 404)
    assert client.get("/previews/20260101_000000_000/..%2Fapp.py").status_code in (400, 404)


def test_upload_name_cannot_escape_folder(client):
    files = [("files", ("../../evil.jpg", scene_bytes(0.1, 4), "image/jpeg"))]
    data = client.post("/api/process", files=files).json()
    assert data["results"][0]["name"] == "evil.jpg"
    assert not (client.folders.incoming.parent.parent / "evil.jpg").exists()


def test_labeling_flow(client):
    upload = client.post("/api/label/upload", files=[("files", ("real.jpg", scene_bytes(0.5, 5), "image/jpeg"))]).json()
    assert upload["saved"] == ["real.jpg"]
    assert client.get("/api/label/images").json()["images"] == [{"name": "real.jpg", "labeled": False}]

    picture = Image.open(io.BytesIO(client.get("/api/label/image/real.jpg").content))
    prediction = client.get("/api/label/prediction/real.jpg").json()
    assert 20 < prediction["cloud_percent"] < 80 and prediction["overlay"].startswith("data:image/png;base64,")

    # рисуем два мазка: красный (облако) и синий (не облако)
    canvas = np.zeros((picture.height, picture.width, 4), dtype=np.uint8)
    canvas[10:30, 10:60] = (255, 0, 0, 255)
    canvas[100:120, 10:60] = (0, 0, 255, 255)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, "PNG")
    body = {"name": "real.jpg", "image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}
    saved = client.post("/api/label/save", json=body).json()
    assert saved == {"cloud_pixels": 1000, "clear_pixels": 1000}
    assert client.get("/api/label/images").json()["images"][0]["labeled"] is True
    mask = np.asarray(Image.open(io.BytesIO(client.get("/api/label/mask/real.jpg").content)))
    assert tuple(mask[20, 20]) == (255, 0, 0, 255) and tuple(mask[110, 20]) == (0, 0, 255, 255)

    # пустой холст удаляет разметку
    buf = io.BytesIO()
    Image.fromarray(np.zeros_like(canvas)).save(buf, "PNG")
    body["image"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    assert client.post("/api/label/save", json=body).json() == {"cloud_pixels": 0, "clear_pixels": 0}
    assert client.get("/api/label/mask/real.jpg").status_code == 404
    assert client.get("/api/label/image/missing.jpg").status_code == 404


def test_smart_fill_region(client):
    client.post("/api/label/upload", files=[("files", ("region.jpg", scene_bytes(0.5, 6), "image/jpeg"))])
    image = webapp.load_real("region.jpg")
    mask = webapp.get_model().analyze(image["rgb"], image["alpha"])["mask"]
    ys, xs = np.nonzero(mask)
    assert len(xs) > 0

    x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    cloud = client.post("/api/label/region", json={"name": "region.jpg", "x": x, "y": y, "tool": "cloud"}).json()
    assert cloud["found"] is True and cloud["pixels"] > 0
    region = np.asarray(Image.open(io.BytesIO(base64.b64decode(cloud["image"].split(",", 1)[-1]))))
    assert tuple(region[y, x]) == (255, 0, 0, 255)
    assert int((region[..., 3] > 0).sum()) == cloud["pixels"]

    wrong = client.post("/api/label/region", json={"name": "region.jpg", "x": x, "y": y, "tool": "clear"}).json()
    assert wrong["found"] is False and "облак" in wrong["message"]

    clear_ys, clear_xs = np.nonzero(~mask)
    cx, cy = int(clear_xs[len(clear_xs) // 2]), int(clear_ys[len(clear_ys) // 2])
    clear = client.post("/api/label/region", json={"name": "region.jpg", "x": cx, "y": cy, "tool": "clear"}).json()
    assert clear["found"] is True
    region = np.asarray(Image.open(io.BytesIO(base64.b64decode(clear["image"].split(",", 1)[-1]))))
    assert tuple(region[cy, cx]) == (0, 0, 255, 255)


def test_prediction_cloud_transfer_contains_only_red_cloud_pixels(client):
    client.post("/api/label/upload", files=[("files", ("prediction.jpg", scene_bytes(0.5, 6), "image/jpeg"))])
    image = webapp.load_real("prediction.jpg")
    cloud_mask = webapp.get_model().analyze(image["rgb"], image["alpha"])["mask"]

    result = client.get("/api/label/prediction-clouds/prediction.jpg").json()

    rgba = np.asarray(Image.open(io.BytesIO(base64.b64decode(result["image"].split(",", 1)[-1]))))
    assert np.array_equal(rgba[..., 3] > 0, cloud_mask)
    assert np.all(rgba[cloud_mask] == (255, 0, 0, 255))
    assert result["cloud_pixels"] == int(cloud_mask.sum())


def test_cloud_area_fill_returns_only_predicted_clouds_inside_contour(client):
    client.post("/api/label/upload", files=[("files", ("cloud-area.jpg", scene_bytes(0.5, 6), "image/jpeg"))])
    image = webapp.load_real("cloud-area.jpg")
    analysis = webapp.get_model().analyze(image["rgb"], image["alpha"])
    ys, xs = np.nonzero(analysis["mask"])
    assert len(xs) > 0
    x, y = int(xs[len(xs) // 2]), int(ys[len(ys) // 2])
    h, w = analysis["mask"].shape
    points = [
        {"x": max(0, x - 20), "y": max(0, y - 20)},
        {"x": min(w - 1, x + 20), "y": max(0, y - 20)},
        {"x": min(w - 1, x + 20), "y": min(h - 1, y + 20)},
        {"x": max(0, x - 20), "y": min(h - 1, y + 20)},
    ]

    result = client.post("/api/label/cloud-area", json={"name": "cloud-area.jpg", "points": points}).json()

    assert result["found"] is True
    rgba = np.asarray(Image.open(io.BytesIO(base64.b64decode(result["image"].split(",", 1)[-1]))))
    selected = rgba[..., 3] > 0
    assert selected[y, x]
    assert np.all(~selected | analysis["mask"])
    assert np.all(rgba[selected] == (255, 0, 0, 255))


def test_model_reset_without_base_is_explained(client):
    r = client.post("/api/model/reset")
    assert r.status_code == 404 and "Базовая модель" in r.json()["detail"]


def test_missing_model_is_reported(client):
    client.folders.model.unlink()
    r = client.post("/api/process-folder", json={})
    assert r.status_code == 503 and "train.py" in r.json()["detail"]
    assert client.get("/api/status").json()["model_ready"] is False
