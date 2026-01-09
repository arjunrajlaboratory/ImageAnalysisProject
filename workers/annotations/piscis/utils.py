from piscis.paths import MODELS_DIR


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
    private_folder_id = gc.get('folder', parameters={'parentId': user_id, 'parentType': 'user', 'name': 'Private'})[0]['_id']
    piscis_folder_id = mkdir(gc, private_folder_id, '.piscis')

    return piscis_folder_id


def list_girder_models(gc):

    piscis_folder_id = get_piscis_dir(gc)
    models_folder_id = mkdir(gc, piscis_folder_id, 'models')
    girder_models = list(gc.listItem(models_folder_id))
    for model in girder_models:
        model['model_name'] = model['name'].rsplit('.pt', 1)[0]

    return girder_models, models_folder_id


def download_girder_model(gc, model_name):

    girder_models, _ = list_girder_models(gc)
    girder_model = [model for model in girder_models if model['name'] == f'{model_name}.pt']
    if not girder_model:
        girder_model = [model for model in girder_models if model['name'] == model_name]
    if girder_model:
        gc.downloadItem(girder_model[0]['_id'], MODELS_DIR, girder_model[0]['name'])


def upload_girder_model(gc, model_name):

    girder_models, models_folder_id = list_girder_models(gc)
    girder_model = [model for model in girder_models if model['name'] == f'{model_name}.pt']
    if girder_model:
        gc.delete(f"{girder_model[0]['_modelType']}/{girder_model[0]['_id']}")

    gc.uploadFileToFolder(models_folder_id, MODELS_DIR / f'{model_name}.pt')
