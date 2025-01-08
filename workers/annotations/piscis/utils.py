from piscis.paths import CACHE_DIR, MODELS_DIR


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


def get_piscis_dir(gc):

    user_id = gc.get('user/me')['_id']
    public_folder_id = gc.get('folder', parameters={'parentId': user_id, 'parentType': 'user', 'name': 'Public'})[0]['_id']
    piscis_folder_id = mkdir(gc, public_folder_id, '.piscis')

    return piscis_folder_id


def list_girder_models(gc):

    piscis_folder_id = get_piscis_dir(gc)
    models_folder_id = mkdir(gc, piscis_folder_id, 'models')
    girder_models = list(gc.listItem(models_folder_id))

    return girder_models, models_folder_id


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


def list_girder_cache(gc, mode):

    piscis_folder_id = get_piscis_dir(gc)
    
    if mode == 'predict':
        cache_folder_id = mkdir(gc, piscis_folder_id, 'predict_cache')
    elif mode == 'train':
        cache_folder_id = mkdir(gc, piscis_folder_id, 'train_cache')
    else:
        cache_folder_id = None
    
    girder_cache = list(gc.listItem(cache_folder_id))

    return girder_cache, cache_folder_id


def download_girder_cache(gc, mode):
    
    girder_cache, _ = list_girder_cache(gc, mode)
    for c in girder_cache:
        gc.downloadItem(c['_id'], CACHE_DIR, c['name'])


def upload_girder_cache(gc, mode):
    
    girder_cache, cache_folder_id = list_girder_cache(gc, mode)
    girder_cache = [c['name'] for c in girder_cache]

    for cache_path in CACHE_DIR.glob('*'):
        if cache_path.is_dir():
            continue
        
        if cache_path.stem not in girder_cache:
            gc.uploadFileToFolder(cache_folder_id, cache_path)
