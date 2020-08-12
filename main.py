# To run:
# python3 -B main.py

# from toy_analysis import *
from mooc_analysis import *

# Different ways of building vectors are being explored, e.g.
# DW = DeepWalk
# FMBPV = fuzzy-membership-based property vectors
embedding_method = "FMBPV"

### Main ###
load_all_atoms()
if embedding_method == "DW":
  load_deepwalk_model()
elif embedding_method == "FMBPV":
  load_property_vectors()

# populate_atomspace()
# generate_subsets()
# calculate_truth_values()
# infer_attractions()
# export_all_atoms()

# if embedding_method == "DW":
#   train_deepwalk_model()
#   export_deepwalk_model()
# elif embedding_method == "FMBPV":
#   build_property_vectors()
#   export_property_vectors()

compare(embedding_method)
