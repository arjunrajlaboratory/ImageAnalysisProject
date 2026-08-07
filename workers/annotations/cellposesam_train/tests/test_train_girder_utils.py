import importlib.util
from pathlib import Path

import pytest


_GIRDER_UTILS_PATH = Path(__file__).resolve().parent.parent / 'girder_utils.py'
_SPEC = importlib.util.spec_from_file_location(
    'cellposesam_train_girder_utils', _GIRDER_UTILS_PATH)
girder_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(girder_utils)


class RecordingGirderClient:
    def __init__(self, *, materialize_download=True, fail_upload=False):
        self.materialize_download = materialize_download
        self.fail_upload = fail_upload
        self.events = []

    def downloadItem(self, item_id, destination, name):
        self.events.append(('download', item_id, Path(destination), name))
        if self.materialize_download:
            (Path(destination) / name).write_bytes(b'checkpoint')

    def uploadFileToFolder(self, folder_id, model_path):
        self.events.append(('upload', folder_id, Path(model_path)))
        if self.fail_upload:
            raise RuntimeError('upload failed')
        return {'itemId': 'new-item'}

    def delete(self, resource_path):
        self.events.append(('delete', resource_path))


def test_download_custom_model_returns_verified_checkpoint(
        tmp_path, monkeypatch):
    client = RecordingGirderClient()
    monkeypatch.setattr(girder_utils, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(
        girder_utils,
        'list_girder_models',
        lambda gc: ([{'_id': 'model-item', 'name': 'custom'}], 'models-folder'))

    model_path = girder_utils.download_girder_model(client, 'custom')

    assert model_path == tmp_path / 'custom'
    assert model_path.is_file()


def test_download_custom_model_rejects_missing_girder_item(
        tmp_path, monkeypatch):
    client = RecordingGirderClient()
    monkeypatch.setattr(girder_utils, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(
        girder_utils, 'list_girder_models',
        lambda gc: ([], 'models-folder'))

    with pytest.raises(FileNotFoundError, match='custom'):
        girder_utils.download_girder_model(client, 'custom')


def test_download_custom_model_requires_downloaded_file(
        tmp_path, monkeypatch):
    client = RecordingGirderClient(materialize_download=False)
    stale_model = tmp_path / 'custom'
    stale_model.write_bytes(b'stale checkpoint')
    monkeypatch.setattr(girder_utils, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(
        girder_utils,
        'list_girder_models',
        lambda gc: ([{'_id': 'model-item', 'name': 'custom'}], 'models-folder'))

    with pytest.raises(FileNotFoundError, match='download'):
        girder_utils.download_girder_model(client, 'custom')

    assert not stale_model.exists()


def test_upload_replaces_old_model_only_after_new_upload_succeeds(
        tmp_path, monkeypatch):
    client = RecordingGirderClient()
    (tmp_path / 'custom').write_bytes(b'checkpoint')
    monkeypatch.setattr(girder_utils, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(
        girder_utils,
        'list_girder_models',
        lambda gc: ([
            {'_id': 'old-item', '_modelType': 'item', 'name': 'custom'},
        ], 'models-folder'))

    result = girder_utils.upload_girder_model(client, 'custom')

    assert result == {'itemId': 'new-item'}
    assert [event[0] for event in client.events] == ['upload', 'delete']


def test_upload_failure_preserves_old_model(tmp_path, monkeypatch):
    client = RecordingGirderClient(fail_upload=True)
    (tmp_path / 'custom').write_bytes(b'checkpoint')
    monkeypatch.setattr(girder_utils, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(
        girder_utils,
        'list_girder_models',
        lambda gc: ([
            {'_id': 'old-item', '_modelType': 'item', 'name': 'custom'},
        ], 'models-folder'))

    with pytest.raises(RuntimeError, match='upload failed'):
        girder_utils.upload_girder_model(client, 'custom')

    assert [event[0] for event in client.events] == ['upload']
