import importlib.util
import sys
import types
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent.parent / 'download_models.py'


class RecordingPiscis:
    model_names = []

    def __init__(self, model_name):
        self.model_names.append(model_name)


class RecordingFilesystem:
    def __init__(self):
        self.downloads = []

    def download(self, source, destination):
        self.downloads.append((source, Path(destination)))


def _load_download_script(monkeypatch):
    huggingface_hub = types.ModuleType('huggingface_hub')
    huggingface_hub.HfFileSystem = RecordingFilesystem

    piscis = types.ModuleType('piscis')
    piscis.Piscis = RecordingPiscis

    monkeypatch.setitem(sys.modules, 'huggingface_hub', huggingface_hub)
    monkeypatch.setitem(sys.modules, 'piscis', piscis)

    spec = importlib.util.spec_from_file_location(
        'piscis_download_models_under_test', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configured_models_download_converted_torch_checkpoint(
        tmp_path, monkeypatch):
    module = _load_download_script(monkeypatch)
    filesystem = RecordingFilesystem()

    module.download_models(
        models_dir=tmp_path,
        piscis_cls=RecordingPiscis,
        filesystem=filesystem,
    )

    assert filesystem.downloads == [(
        'rajlab/ps_20240419_112256/ps_20240419_112256.pt',
        tmp_path / 'ps_20240419_112256.pt',
    )]


def test_preloads_supported_builtin_models(tmp_path, monkeypatch):
    RecordingPiscis.model_names = []
    module = _load_download_script(monkeypatch)

    module.download_models(
        models_dir=tmp_path,
        piscis_cls=RecordingPiscis,
        filesystem=RecordingFilesystem(),
    )

    assert RecordingPiscis.model_names == [
        '20230616',
        '20230709',
        '20230905',
        '20251212',
    ]
