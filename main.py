### Dataset to be used
# from toy_analysis import *
# from mooc_analysis import *
from go_analysis import *

### Different ways of building vectors are being explored
# DW = DeepWalk
# FMBPV = fuzzy-membership-based property vectors
embedding_method = "FMBPV"

### Utils for calling the APIs
def generate_atoms():
  populate_atomspace()
  generate_subsets()
  calculate_truth_values()
  infer_attractions()
  export_all_atoms()

def generate_embeddings():
  if embedding_method == "DW":
    train_deepwalk_model()
    export_deepwalk_model()
  elif embedding_method == "FMBPV":
    build_property_vectors()
    # do_pca()
    do_kpca()
    export_property_vectors()

def load_atoms():
  load_all_atoms()

def load_embeddings():
  if embedding_method == "DW":
    load_deepwalk_model()
  elif embedding_method == "FMBPV":
    load_property_vectors()

def get_results():
  compare(embedding_method)

### Main ###
# load_atoms()
# load_embeddings()
generate_atoms()
generate_embeddings()
get_results()
