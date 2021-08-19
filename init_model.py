import os
import sentence_transformers
from sentence_transformers.util import snapshot_download


model = os.getenv('MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')
models_path = './models'
model_path = os.path.join(models_path, model)

if not os.path.exists(models_path):
    os.mkdir(models_path)

model_tmp_path = snapshot_download(
    repo_id='sentence-transformers/' + model,
    cache_dir=models_path,
    library_name='sentence-transformers',
    library_version=sentence_transformers.__version__,
    ignore_files=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5']
)

os.rename(model_tmp_path, model_path)
