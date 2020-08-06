import csv
import os
import pickle
import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from opencog.atomspace import AtomSpace, types
from opencog.bindlink import execute_atom
from opencog.scheme_wrapper import scheme_eval
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog
from scipy.spatial import distance
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA

base_datasets_dir = os.getcwd() + "/datasets/"
base_results_dir = os.getcwd() + "/results/mooc/"

mooc_actions_tsv = base_datasets_dir + "mooc_actions.tsv"
mooc_action_labels_tsv = base_datasets_dir + "mooc_action_labels.tsv"
mooc_action_features_tsv = base_datasets_dir + "mooc_action_features.tsv"
member_links_scm = base_results_dir + "member-links.scm"
evaluation_links_scm = base_results_dir + "evaluation-links.scm"
subset_links_scm = base_results_dir + "subset-links.scm"
attraction_links_scm = base_results_dir + "attraction-links.scm"
sentences_pickle = base_results_dir + "sentences.pickle"
deepwalk_bin = base_results_dir + "deepwalk.bin"
pca_png = base_results_dir + "pca.png"
results_csv = base_results_dir + "results.csv"

course_id_prefix = "course:"
user_id_prefix = "user:"
target_id_prefix = "target:"
feature_prefix = "feature:"

deepwalk = None

### Utils ###
def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

def get_concepts(str_prefix):
  return [x for x in atomspace.get_atoms_by_type(types.ConceptNode) if x.name.startswith(str_prefix)]

def intensional_difference(c1, c2):
  cn1 = "(Concept \"{}\")".format(c1)
  cn2 = "(Concept \"{}\")".format(c2)
  intdiff = "(IntensionalDifference {} {})".format(cn1, cn2)
  scm("(pln-bc {})".format(intdiff))
  tv_mean = float(scm("(cog-mean {})".format(intdiff)))
  tv_conf = float(scm("(cog-confidence {})".format(intdiff)))
  return TruthValue(tv_mean, tv_conf)

def intensional_similarity(c1, c2):
  cn1 = "(Concept \"{}\")".format(c1)
  cn2 = "(Concept \"{}\")".format(c2)
  intsim = "(IntensionalSimilarity {} {})".format(cn1, cn2)
  scm("(pln-bc {})".format(intsim))
  tv_mean = float(scm("(cog-mean {})".format(intsim)))
  tv_conf = float(scm("(cog-confidence {})".format(intsim)))
  return TruthValue(tv_mean, tv_conf)

### Initialize the AtomSpace ###
atomspace = AtomSpace()
initialize_opencog(atomspace)

### Guile setup ###
scm("(add-to-load-path \"/usr/share/guile/site/2.2/opencog\")")
scm("(add-to-load-path \".\")")
scm("(use-modules (opencog) (opencog ure) (opencog pln))")
scm(" ".join([
  "(define (write-atoms-to-file file atoms)",
    "(define fp (open-output-file file))",
    "(for-each",
      "(lambda (x) (display x fp))",
      "atoms)",
    "(close-port fp))"]))

def load_all_atoms():
  print("--- Loading Atoms from files")
  scm("(use-modules (opencog persist-file))")
  scm("(load-file \"" + member_links_scm + "\")")
  scm("(load-file \"" + evaluation_links_scm + "\")")
  scm("(load-file \"" + subset_links_scm + "\")")
  scm("(load-file \"" + attraction_links_scm + "\")")

def export_all_atoms():
  def write_atoms_to_file(filename, atom_list_str):
    scm(" ".join([
      "(write-atoms-to-file",
      "\"" + filename + "\"",
      atom_list_str + ")"]))

  print("--- Exporting Atoms to files")
  write_atoms_to_file(member_links_scm, "(cog-get-atoms 'MemberLink)")
  write_atoms_to_file(evaluation_links_scm, "(cog-get-atoms 'EvaluationLink)")
  write_atoms_to_file(subset_links_scm, "(cog-get-atoms 'SubsetLink)")
  write_atoms_to_file(attraction_links_scm, "(cog-get-atoms 'AttractionLink)")

def load_deepwalk_model():
  global deepwalk
  print("--- Loading an existing model from \"{}\"".format(deepwalk_bin))
  deepwalk = Word2Vec.load(deepwalk_bin)

def export_deepwalk_model():
  global deepwalk
  deepwalk.save(deepwalk_bin)

### Populate the AtomSpace ###
# Notes for this dataset:
# - Each action is taken by one and only one user
#   i.e. it's not very useful, so everything associated with it
#   will be passed to the user directly
# - A user will not come back once he/she has dropped-out,
#   i.e. each user is assumed to have taken only one course
def populate_atomspace():
  print("--- Populating the AtomSpace")
  action_feature_dict = {}
  with open(mooc_action_features_tsv) as f:
    action_features = []

    def process_feature(action_id, feature):
      if feature not in action_features:
        action_features.append(feature)

      feature_id = str(action_features.index(feature))
      feature_name = feature_prefix + feature_id

      if action_feature_dict.get(action_id):
        action_feature_dict[action_id].add(feature_name)
      else:
        action_feature_dict[action_id] = {feature_name}

    next(f)
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      feature_0 = content[1].strip()
      feature_1 = content[2].strip()
      feature_2 = content[3].strip()
      feature_3 = content[4].strip()
      process_feature(action_id, feature_0)
      process_feature(action_id, feature_1)
      process_feature(action_id, feature_2)
      process_feature(action_id, feature_3)

  dropout_action_ids = []
  with open(mooc_action_labels_tsv) as f:
    next(f)
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      label = content[1].strip()
      if label == "1":
        dropout_action_ids.append(action_id)

  with open(mooc_actions_tsv) as f:
    def evalink(pred, node1, node2):
      return "\n".join([
        "(EvaluationLink",
        "\t(PredicateNode \"" + pred + "\")",
        "\t(ListLink",
        "\t\t(ConceptNode \"" + node1 + "\")",
        "\t\t(ConceptNode \"" + node2 + "\")))\n"])

    def memblink(node1, node2):
      return "\n".join([
        "(MemberLink",
        "\t(ConceptNode \"" + node1 + "\")",
        "\t(ConceptNode \"" + node2 + "\"))\n"])

    dropout_users = []
    next(f)
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      user_id = content[1].strip()
      target_id = content[2].strip()

      course_name = course_id_prefix + user_id
      user_name = user_id_prefix + user_id
      target_name = target_id_prefix + target_id
      feature_names = action_feature_dict[action_id]

      scm(memblink(course_name, user_name))
      scm(memblink(course_name, target_name))
      scm(evalink("has_action_target", user_name, target_name))
      for feature_name in feature_names:
        scm(memblink(course_name, feature_name))
        scm(evalink("has_action_feature", user_name, feature_name))
      if action_id in dropout_action_ids:
        scm(memblink(course_name, "dropped-out"))
        scm(evalink("has", user_name, "dropped-out"))
        dropout_users.append(user_name)
      scm(memblink(course_name, "not-dropped-out"))
    for user in get_concepts(user_id_prefix):
      user_name = user.name
      if user_name not in dropout_users:
        scm(evalink("has", user_name, "not-dropped-out"))

def generate_subsets():
  print("--- Generating SubsetLinks")
  for evalink in atomspace.get_atoms_by_type(types.EvaluationLink):
    source = evalink.out[1].out[0]
    target = evalink.out[1].out[1]
    SubsetLink(source, target)
    SubsetLink(target, source)

def calculate_truth_values():
  print("--- Calculating Truth Values")

  node_member_dict = {}
  def get_members(node):
    if node_member_dict.get(node):
      return node_member_dict[node]
    else:
      memblinks = [x for x in node.incoming if x.type == types.MemberLink and x.out[1] == node]
      members = [x.out[0] for x in tuple(memblinks)]
      node_member_dict[node] = members
      return members

  def get_confidence(count):
    return float(scm("(count->confidence " + str(count) + ")"))

  # EvaluationLinks are generated directly from the data, can be considered as true
  for e in atomspace.get_atoms_by_type(types.EvaluationLink):
    e.tv = TruthValue(1, 1)

  # MemberLinks are generated directly from the data, can be considered as true
  for m in atomspace.get_atoms_by_type(types.MemberLink):
    m.tv = TruthValue(1, 1)

  # ConceptNode "A" (stv s c)
  # where:
  # s = |A| / |universe|
  # c = |universe|
  universe_size = len(get_concepts(course_id_prefix))
  tv_confidence = get_confidence(universe_size)
  for c in atomspace.get_atoms_by_type(types.ConceptNode):
    member_size = len(get_members(c))
    tv_strength = member_size / universe_size
    c.tv = TruthValue(tv_strength, tv_confidence)

  # SubsetLink (stv s c)
  #   A
  #   B
  # where:
  # s = |B intersect A| / |A|
  # c = |A|
  for s in atomspace.get_atoms_by_type(types.SubsetLink):
    node1 = s.out[0]
    node2 = s.out[1]
    node1_members = get_members(node1)
    node2_members = get_members(node2)
    common_members = set(node2_members).intersection(node1_members)
    tv_strength = len(common_members) / len(node1_members)
    tv_confidence = get_confidence(len(node1_members))
    s.tv = TruthValue(tv_strength, tv_confidence)

def infer_attractions():
  print("--- Inferring AttractionLinks")
  scm("(pln-load 'empty)")
  # (Subset A B) |- (Subset (Not A) B)
  scm("(pln-add-rule \"subset-condition-negation-rule\")")
  # (Subset A B) (Subset (Not A) B) |- (Attraction A B)
  scm("(pln-add-rule \"subset-attraction-introduction-rule\")")
  scm(" ".join(["(pln-bc",
                  "(Attraction (Variable \"$X\") (Variable \"$Y\"))",
                  "#:vardecl",
                    "(VariableSet",
                      "(TypedVariable (Variable \"$X\") (Type \"ConceptNode\"))",
                      "(TypedVariable (Variable \"$Y\") (Type \"ConceptNode\")))",
                  "#:maximum-iterations 12",
                  "#:complexity-penalty 10)"]))

def train_deepwalk_model():
  global deepwalk
  next_words_dict = {}
  sentences = []

  def add_to_next_words_dict(w, nw):
    if next_words_dict.get(w):
      next_words_dict[w].add(nw)
    else:
      next_words_dict[w] = {nw}

  def get_reverse_pred(pred):
    if pred == "has_action_target":
      return "is_an_action_target_of"
    elif pred == "has_action_feature":
      return "is_an_action_feature_of"
    elif pred == "has":
      return "was_the_result_of"
    else:
      raise Exception("The reverse of predicate \"{}\" is not defined!".format(pred))

  print("--- Gathering next words")
  evalinks = atomspace.get_atoms_by_type(types.EvaluationLink)
  for evalink in evalinks:
    pred = evalink.out[0].name
    rev_pred = get_reverse_pred(pred)
    source = evalink.out[1].out[0].name
    target = evalink.out[1].out[1].name
    add_to_next_words_dict(source, (pred, target))
    add_to_next_words_dict(target, (rev_pred, source))
  for k, v in next_words_dict.items():
    next_words_dict[k] = tuple(v)

  print("--- Generating sentences")
  num_sentences = 10000000
  num_walks = 15
  first_words = [x.name for x in get_concepts(user_id_prefix)]
  for i in range(num_sentences):
    sentence = [random.choice(first_words)]
    for j in range(num_walks):
      last_word = sentence[-1]
      next_words = random.choice(next_words_dict.get(last_word))
      sentence.append(next_words[0])
      sentence.append(next_words[1])
    sentences.append(sentence)
    if len(sentences) % 10000 == 0:
      print(len(sentences))
  with open(sentences_pickle, "wb") as f:
    pickle.dump(sentences, f)

  print("--- Training model")
  deepwalk = Word2Vec(sentences, min_count=1)

def plot_pca():
  print("--- Plotting")
  X = deepwalk[deepwalk.wv.vocab]
  pca = PCA(n_components = 2)
  result = pca.fit_transform(X)
  pyplot.scatter(result[:, 0], result[:, 1])
  words = list(deepwalk.wv.vocab)
  for i, word in enumerate(words):
    pyplot.annotate(word, xy = (result[i, 0], result[i, 1]))
  pyplot.savefig(pca_png, dpi=1000)

def compare(embedding_method):
  print("--- Comparing PLN vs " + embedding_method)

  node_pattern_dict = {}
  def get_properties(node):
    def get_attractions(node):
      return [x for x in node.incoming if x.type == types.AttractionLink and x.out[0] == node]
    if node_pattern_dict.get(node):
      return node_pattern_dict[node]
    else:
      pats = [x.out[1] for x in get_attractions(node)]
      node_pattern_dict[node] = pats
    return pats

  # Get the user pairs
  print("--- Generating user pairs")
  users = [x.name for x in get_concepts(user_id_prefix)]
  random.shuffle(users)
  user_pairs = list(zip(users[::2], users[1::2]))

  print("--- Generating results")
  # PLN setup
  scm("(pln-load 'empty)")
  scm("(pln-add-rule \"intensional-similarity-direct-introduction-rule\")")

  # Output file
  results_csv_fp = open(results_csv, "w")
  first_row = ",".join([
    "User 1",
    "User 2",
    "No. of user 1 properties",
    "No. of user 2 properties",
    "No. of common properties",
    "Intensional Similarity",
    "Vector distance",
    "Pearson",
    "Spearman",
    "Kendall"])
  results_csv_fp.write(first_row + "\n")

  # Generate the results
  for pair in user_pairs:
    u1 = pair[0]
    u2 = pair[1]
    u1_properties = get_properties(ConceptNode(u1))
    u2_properties = get_properties(ConceptNode(u2))
    u1_pattern_size = len(u1_properties)
    u2_pattern_size = len(u2_properties)
    common_properties = set(u1_properties).intersection(u2_properties)
    common_pattern_size = len(common_properties)
    intsim_tv = intensional_similarity(u1, u2).mean
    if embedding_method == "DW":
      # DeepWalk euclidean distance
      v1 = deepwalk[u1]
      v2 = deepwalk[u2]
    else:
      v1 = property_vectors[u1]
      v2 = property_vectors[u2]
    vec_dist = distance.euclidean(v1, v2)
    row = ",".join([
      u1,
      u2,
      str(u1_pattern_size),
      str(u2_pattern_size),
      str(common_pattern_size),
      str(intsim_tv),
      str(vec_dist)])
    results_csv_fp.write(row + "\n")
  results_csv_fp.close()

  # For correlations
  csv_reader = csv.reader(open(results_csv))
  intsim_col_idx = first_row.split(",").index("Intensional Similarity")
  vecdist_col_idx = first_row.split(",").index("Vector distance")
  # Skip the first row (column names) before sorting the floats
  next(csv_reader)
  sorted_results = sorted(csv_reader, key=lambda row: float(row[intsim_col_idx]), reverse=True)

  results_csv_fp = open(results_csv, "w")
  results_csv_fp.write(first_row + "\n")

  intsim_n = []
  vecdist_n = []

  for row in sorted_results:
    intsim_n.append(float(row[intsim_col_idx]))
    vecdist_n.append(float(row[vecdist_col_idx]))

    if len(intsim_n) >= 10:
      pearson = pearsonr(intsim_n, vecdist_n)[0]
      spearman = spearmanr(intsim_n, vecdist_n)[0]
      kendall = kendalltau(intsim_n, vecdist_n)[0]
    else:
      pearson = "-"
      spearman = "-"
      kendall = "-"

    new_row = ",".join([",".join(row), ",".join([str(pearson), str(spearman), str(kendall)])])
    results_csv_fp.write(new_row + "\n")

  print("pearson = {}\nspearman = {}\nkendall = {}".format(pearson, spearman, kendall))
  results_csv_fp.close()

# ----------
fuzzy_membership_values = {}
property_vectors = {}

def calculate_fuzzy_membership_values():
  global fuzzy_membership_values
  num_users = len(get_concepts(user_id_prefix))

  def get_user_with_property(prop, pred):
    return execute_atom(atomspace,
             GetLink(
               EvaluationLink(
                 PredicateNode(pred),
                 ListLink(
                   VariableNode("$X"),
                   prop)))).out

  for p in get_concepts(target_id_prefix):
    fuzzy_membership_values[p.name] = 1 - (len(get_user_with_property(p, "has_action_target")) / num_users)
    print("FMV for '{}' = {}".format(p.name, fuzzy_membership_values[p.name]))

  for p in get_concepts(feature_prefix):
    fuzzy_membership_values[p.name] = 1 - (len(get_user_with_property(p, "has_action_feature")) / num_users)
    print("FMV for '{}' = {}".format(p.name, fuzzy_membership_values[p.name]))

  for p in [ConceptNode("dropped-out"), ConceptNode("not-dropped-out")]:
    fuzzy_membership_values[p.name] = 1 - (len(get_user_with_property(p, "has")) / num_users)
    print("FMV for '{}' = {}".format(p.name, fuzzy_membership_values[p.name]))

def build_property_vectors():
  print("--- Building vectors")

  global property_vectors
  all_evalinks = atomspace.get_atoms_by_type(types.EvaluationLink)

  for c in get_concepts(user_id_prefix):
    pvec = []
    listlinks = [x for x in c.incoming if x.type == types.ListLink]
    evalinks = [x.incoming[0] for x in listlinks if x.incoming[0].type == types.EvaluationLink]
    for e in all_evalinks:
      if e in evalinks:
        pvec.append(fuzzy_membership_values[e.out[1].out[1].name])
      else:
        pvec.append(0)
    property_vectors[c.name] = pvec

### Main ###
# load_all_atoms()
# load_deepwalk_model()

populate_atomspace()
generate_subsets()
calculate_truth_values()
infer_attractions()
export_all_atoms()
train_deepwalk_model()
export_deepwalk_model()
plot_pca()

# calculate_fuzzy_membership_values()
# build_property_vectors()

compare("DW")
