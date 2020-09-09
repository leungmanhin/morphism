import numpy
import os
import pickle
import random
from gensim.models import Word2Vec
from matplotlib import pyplot
from opencog.atomspace import AtomSpace, types
from opencog.bindlink import execute_atom
from opencog.logger import log
from opencog.scheme_wrapper import scheme_eval
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog
from scipy.spatial import distance
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA, KernelPCA

log.set_level("ERROR")

base_datasets_dir = os.getcwd() + "/datasets/"
base_results_dir = os.getcwd() + "/results/mooc/"
if not os.path.exists(base_results_dir):
  os.makedirs(base_results_dir)

mooc_actions_tsv = base_datasets_dir + "mooc_actions.tsv"
mooc_action_labels_tsv = base_datasets_dir + "mooc_action_labels.tsv"
mooc_action_features_tsv = base_datasets_dir + "mooc_action_features.tsv"
member_links_scm = base_results_dir + "member-links.scm"
evaluation_links_scm = base_results_dir + "evaluation-links.scm"
subset_links_scm = base_results_dir + "subset-links.scm"
attraction_links_scm = base_results_dir + "attraction-links.scm"
sentences_pickle = base_results_dir + "sentences.pickle"
deepwalk_bin = base_results_dir + "deepwalk.bin"
property_vector_pickle = base_results_dir + "property_vectors.pickle"
results_csv = base_results_dir + "results.csv"
course_id_prefix = "course:"
user_id_prefix = "user:"
target_id_prefix = "target:"
feature_prefix = "feature:"

property_vectors = {}
deepwalk = None
num_sentences = 10000000
num_walks = 15

### Initialize the AtomSpace ###
atomspace = AtomSpace()
initialize_opencog(atomspace)

### Guile setup ###
def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

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

def build_property_vectors():
  print("--- Building property vectors")

  global property_vectors
  all_evalinks = atomspace.get_atoms_by_type(types.EvaluationLink)

  for user in get_concepts(user_id_prefix):
    pvec = []
    for e in all_evalinks:
      prop = e.out[1].out[1]
      attraction = AttractionLink(user, prop)
      attraction_tv = attraction.tv
      pvec.append(attraction.tv.mean * attraction.tv.confidence)
    property_vectors[user.name] = pvec

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

def compare(embedding_method):
  print("--- Comparing PLN vs " + embedding_method)

  node_pattern_dict = {}
  def get_properties(node):
    def get_attractions(node):
      return [x for x in node.incoming if x.type == types.AttractionLink and x.out[0] == node]
    if node_pattern_dict.get(node):
      return node_pattern_dict[node]
    else:
      pats = [x.out[1] for x in get_attractions(node) if x.tv.mean * x.tv.confidence > 0]
      node_pattern_dict[node] = pats
    return pats

  # Get the user pairs
  print("--- Picking random user pairs")
  users = [x.name for x in get_concepts(user_id_prefix)]
  random.shuffle(users)
  user_pairs = list(zip(users[::2], users[1::2]))

  print("--- Generating results")

  # PLN setup
  scm("(pln-load 'empty)")
  scm("(pln-add-rule \"intensional-similarity-direct-introduction-rule\")")

  # Generate the results
  all_rows = []
  for pair in user_pairs:
    p1 = pair[0]
    p2 = pair[1]
    p1_properties = get_properties(ConceptNode(p1))
    p2_properties = get_properties(ConceptNode(p2))
    p1_pattern_size = len(p1_properties)
    p2_pattern_size = len(p2_properties)
    common_properties = set(p1_properties).intersection(p2_properties)
    common_pattern_size = len(common_properties)
    # PLN intensional similarity
    intsim = intensional_similarity(p1, p2)
    intsim_tv = intsim.mean if intsim.confidence > 0 else 0
    # Vector distance
    if embedding_method == "DW":
      v1 = deepwalk[p1]
      v2 = deepwalk[p2]
    elif embedding_method == "FMBPV":
      v1 = property_vectors[p1]
      v2 = property_vectors[p2]
    # vec_dist = distance.euclidean(v1, v2)
    vec_dist = distance.cosine(v1, v2)
    # vec_dist = fuzzy_jaccard(v1, v2)
    # vec_dist = tanimoto(v1, v2)
    row = [
      p1,
      p2,
      p1_pattern_size,
      p2_pattern_size,
      common_pattern_size,
      intsim_tv,
      vec_dist]
    all_rows.append(row)

  # Sort in descending order of intensional similarity
  intsim_col_idx = 5
  vecdist_col_idx = 6
  all_rows = numpy.array(all_rows)
  all_rows_sorted = all_rows[all_rows[:, intsim_col_idx].argsort()][::-1]

  intsim_columns = all_rows_sorted[:, intsim_col_idx].astype(numpy.float)
  vecdist_columns = all_rows_sorted[:, vecdist_col_idx].astype(numpy.float)
  pearson = pearsonr(intsim_columns, vecdist_columns)[0]
  spearman = spearmanr(intsim_columns, vecdist_columns)[0]
  kendall = kendalltau(intsim_columns, vecdist_columns)[0]

  print("Pearson = {}\nSpearman = {}\nKendall = {}".format(pearson, spearman, kendall))

  # Write to file
  with open(results_csv, "w") as f:
    first_row = ",".join([
      "Person 1",
      "Person 2",
      "No. of person 1 properties",
      "No. of person 2 properties",
      "No. of common properties",
      "Intensional Similarity",
      "Vector Distance"])
    f.write(first_row + "\n")
    for row in all_rows_sorted:
      f.write(",".join(row) + "\n")

def do_kpca():
  def kernel_func(X):
    dist_dict = {}
    mat = []
    i = 0
    cnt = 0
    total = X.shape[0] ** 2
    for a in X:
      row = []
      j = 0
      for b in X:
        cnt += 1
        print("--- Working on {}/{}...".format(cnt, total))
        if (i, j) in dist_dict:
          row.append(dist_dict[(i, j)])
        elif (j, i) in dist_dict:
          row.append(dist_dict[(j, i)])
        else:
          # dist = fuzzy_jaccard(a, b)
          dist = tanimoto(a, b)
          row.append(dist)
          dist_dict[(i, j)] = dist
        j += 1
      mat.append(row)
      i += 1
    return numpy.array(mat)

  print("--- Doing KPCA")

  X = numpy.array(list(property_vectors.values()))
  X_kpca = KernelPCA(kernel = "precomputed").fit_transform(kernel_func(X))

  for k, kpca_v in zip(property_vectors.keys(), X_kpca):
    property_vectors[k] = kpca_v

def do_pca():
  print("--- Doing PCA")
  pca = PCA()
  pca_results = pca.fit_transform(list(property_vectors.values()))
  cum_sum = numpy.cumsum(pca.explained_variance_ratio_)
  if cum_sum[-1] >= 1:
    print("--- Best PCA components is {} (sum to 1)".format(numpy.where(cum_sum >= 1)[0][0] + 1))
  else:
    print("--- Best PCA components is {} (sum to {})".format(pca.n_components_, cum_sum[-1]))
  for k, pca_v in zip(property_vectors.keys(), pca_results):
    property_vectors[k] = pca_v

def export_all_atoms():
  def write_atoms_to_file(filename, atom_list_str):
    if not os.path.exists(filename):
      open(filename, "w").close()
    scm(" ".join([
      "(write-atoms-to-file",
      "\"" + filename + "\"",
      atom_list_str + ")"]))

  print("--- Exporting Atoms to files")
  write_atoms_to_file(member_links_scm, "(cog-get-atoms 'MemberLink)")
  write_atoms_to_file(evaluation_links_scm, "(cog-get-atoms 'EvaluationLink)")
  write_atoms_to_file(subset_links_scm, "(cog-get-atoms 'SubsetLink)")
  write_atoms_to_file(attraction_links_scm, "(cog-get-atoms 'AttractionLink)")

def export_deepwalk_model():
  print("--- Exporting DeepWalk model to \"{}\"".format(deepwalk_bin))
  deepwalk.save(deepwalk_bin)

def export_property_vectors():
  print("--- Exporting property vectors to \"{}\"".format(property_vector_pickle))
  with open(property_vector_pickle, "wb") as f:
    pickle.dump(property_vectors, f)

def fuzzy_jaccard(v1, v2):
  '''
  This function is supposed to reflect the actual calculation done in the PLN rule.
  Check out the 'intensional-similarity-direct-introduction-rule' for details.
  '''
  numerator = 0
  denominator = 0
  for x, y in zip(v1, v2):
    # The "intersect"
    if x > 0 and y > 0:
      numerator = numerator + min(x,y)
    # The "union"
    denominator = denominator + max(x,y)
  tvs = (numerator / denominator) if denominator > 0 else 0
  return tvs

def generate_subsets():
  print("--- Generating SubsetLinks")
  for evalink in atomspace.get_atoms_by_type(types.EvaluationLink):
    source = evalink.out[1].out[0]
    target = evalink.out[1].out[1]
    SubsetLink(source, target)
    SubsetLink(target, source)

def get_concepts(str_prefix):
  return [x for x in atomspace.get_atoms_by_type(types.ConceptNode) if x.name.startswith(str_prefix)]

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

def load_all_atoms():
  print("--- Loading Atoms from files")
  scm("(use-modules (opencog persist-file))")
  scm("(load-file \"" + member_links_scm + "\")")
  scm("(load-file \"" + evaluation_links_scm + "\")")
  scm("(load-file \"" + subset_links_scm + "\")")
  scm("(load-file \"" + attraction_links_scm + "\")")

def load_deepwalk_model():
  global deepwalk
  print("--- Loading an existing model from \"{}\"".format(deepwalk_bin))
  deepwalk = Word2Vec.load(deepwalk_bin)

def load_property_vectors():
  global property_vectors
  print("--- Loading property vectors from \"{}\"".format(property_vector_pickle))
  property_vectors = pickle.load(open(property_vector_pickle, "rb"))

def populate_atomspace():
  # Notes for this dataset:
  # - Each action is taken by one and only one user
  #   i.e. it's not very useful, so everything associated with it
  #   will be passed to the user directly
  # - A user will not come back once he/she has dropped-out,
  #   i.e. each user is assumed to have taken only one course
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

def tanimoto(v1, v2):
  v1_v2 = numpy.dot(v1, v2)
  v1_sq = numpy.sum(numpy.square(v1))
  v2_sq = numpy.sum(numpy.square(v2))
  return v1_v2 / (v1_sq + v2_sq - v1_v2)

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
