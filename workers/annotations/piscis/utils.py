from piscis.paths import MODELS_DIR


def mkdir(gc, parent_id, folder_name):
    """Create a folder if it doesn't exist, return its ID."""

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
    """Get the Private/.piscis folder, creating it if needed."""
    user_id = gc.get('user/me')['_id']
    private_folder_id = gc.get('folder', parameters={'parentId': user_id, 'parentType': 'user', 'name': 'Private'})[0]['_id']
    piscis_folder_id = mkdir(gc, private_folder_id, '.piscis')

    return piscis_folder_id


def get_public_piscis_dir(gc):
    """Get the legacy Public/.piscis folder. Returns None if it doesn't exist."""
    user_id = gc.get('user/me')['_id']
    public_folders = gc.get('folder', parameters={
        'parentId': user_id,
        'parentType': 'user',
        'name': 'Public'
    })
    if not public_folders:
        return None

    public_folder_id = public_folders[0]['_id']
    piscis_folders = gc.get('folder', parameters={
        'parentId': public_folder_id,
        'parentType': 'folder',
        'name': '.piscis'
    })
    if not piscis_folders:
        return None

    return piscis_folders[0]['_id']


def list_models_from_folder(gc, piscis_folder_id):
    """List models from a specific piscis folder. Returns empty list if models folder doesn't exist."""
    if piscis_folder_id is None:
        return []

    models_folders = gc.get('folder', parameters={
        'parentId': piscis_folder_id,
        'parentType': 'folder',
        'name': 'models'
    })
    if not models_folders:
        return []

    models_folder_id = models_folders[0]['_id']
    girder_models = list(gc.listItem(models_folder_id))
    for model in girder_models:
        model['model_name'] = model['name'].rsplit('.pt', 1)[0]

    return girder_models


def list_girder_models(gc):
    """List models from both Private and Public locations. Private takes precedence."""
    # Get models from Private (primary location)
    piscis_folder_id = get_piscis_dir(gc)
    models_folder_id = mkdir(gc, piscis_folder_id, 'models')
    private_models = list(gc.listItem(models_folder_id))
    for model in private_models:
        model['model_name'] = model['name'].rsplit('.pt', 1)[0]

    # Get models from Public (legacy location) - read-only, don't create
    public_piscis_folder_id = get_public_piscis_dir(gc)
    public_models = list_models_from_folder(gc, public_piscis_folder_id)

    # Merge: Private takes precedence
    private_names = {m['model_name'] for m in private_models}
    merged_models = private_models + [m for m in public_models if m['model_name'] not in private_names]

    return merged_models, models_folder_id


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
