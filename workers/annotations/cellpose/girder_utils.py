from pathlib import Path

# Assuming these are defined elsewhere in your cellpose worker
CELLPOSE_DIR = Path.home() / '.cellpose'
MODELS_DIR = CELLPOSE_DIR / 'models'

def mkdir(gc, parent_id, folder_name):

    folders = gc.get('folder', parameters={'parentId': parent_id, 'parentType': 'folder', 'name': folder_name})
    
    if folders:
        _id = folders[0]['_id']
    else:
        new_folder = gc.post('folder', parameters={
            'parentId': parent_id,
            'parentType': 'folder',
            'name': folder_name
        })
        _id = new_folder['_id']

    return _id


def get_cellpose_dir(gc):

    user_id = gc.get('user/me')['_id']
    public_folder_id = gc.get('folder', parameters={'parentId': user_id, 'parentType': 'user', 'name': 'Public'})[0]['_id']
    cellpose_folder_id = mkdir(gc, public_folder_id, '.cellpose')

    return cellpose_folder_id


def list_girder_models(gc):

    cellpose_folder_id = get_cellpose_dir(gc)
    models_folder_id = mkdir(gc, cellpose_folder_id, 'models')
    girder_models = list(gc.listItem(models_folder_id))

    return girder_models, models_folder_id


def list_local_models():
    """List models in local .cellpose/models directory"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return list(MODELS_DIR.glob('*'))


def download_girder_model(gc, model_name):

    girder_models, _ = list_girder_models(gc)
    girder_model = [model for model in girder_models if model['name'] == model_name]
    if girder_model:
        gc.downloadItem(girder_model[0]['_id'], MODELS_DIR, girder_model[0]['name'])


def upload_girder_model(gc, model_name):

    girder_models, models_folder_id = list_girder_models(gc)
    girder_model = [model for model in girder_models if model['name'] == model_name]
    if girder_model:
        gc.delete(f"{girder_model[0]['_modelType']}/{girder_model[0]['_id']}")

    gc.uploadFileToFolder(models_folder_id, MODELS_DIR / model_name)
