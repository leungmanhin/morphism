# To run:
# python3 -B main.py

from analysis import *

# Different ways of building vectors are being explored, e.g.
# dw = DeepWalk
# mb = Membership-based
embedding_method = "mb"

### Main ###
load_all_atomes()
# load_deepwalk_model()

# populate_atomspace()
# generate_subsets()
# calculate_truth_values()
# infer_attractions()
# export_all_atoms()

calculate_fuzzy_membership_values()

if embedding_method == "dw":
  train_deepwalk_model()
  export_deepwalk_model()
elif embedding_method == "mb":
  build_property_vectors(ica=True)

plot_pca(embedding_method)
compare(embedding_method)
