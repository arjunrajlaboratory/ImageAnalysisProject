from pathlib import Path


BUILTIN_MODEL_NAMES = (
    '20230616',
    '20230709',
    '20230905',
    '20251212',
)
MODEL_REPOSITORIES = ('rajlab/ps_20240419_112256',)
MODELS_DIR = Path.home() / '.piscis' / 'models'


def download_models(
        models_dir=MODELS_DIR,
        piscis_cls=None,
        filesystem=None):
    """Download only runtime-ready PyTorch checkpoints into the image."""
    if piscis_cls is None or filesystem is None:
        from huggingface_hub import HfFileSystem
        from piscis import Piscis

        piscis_cls = piscis_cls or Piscis
        filesystem = filesystem or HfFileSystem()

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    for model_name in BUILTIN_MODEL_NAMES:
        piscis_cls(model_name=model_name)

    for model_path in MODEL_REPOSITORIES:
        model_name = model_path.rsplit('/', 1)[-1]
        checkpoint_name = f'{model_name}.pt'

        # Configured models must be converted before deployment. Downloading an
        # extensionless legacy JAX checkpoint would make Piscis try to import
        # Flax at runtime, but inference images are intentionally torch-only.
        filesystem.download(
            f'{model_path}/{checkpoint_name}',
            str(models_dir / checkpoint_name))


if __name__ == '__main__':
    download_models()
