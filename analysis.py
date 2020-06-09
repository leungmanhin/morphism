import os
import pickle
import random
from gensim.models import Word2Vec
from opencog.atomspace import AtomSpace, types
from opencog.scheme_wrapper import scheme_eval
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog
from scipy.spatial import distance

mooc_actions_tsv = os.getcwd() + "/datasets/mooc_actions.tsv"
mooc_action_labels_tsv = os.getcwd() + "/datasets/mooc_action_labels.tsv"
mooc_action_features_tsv = os.getcwd() + "/datasets/mooc_action_features.tsv"
member_links_scm = os.getcwd() + "/results/member-links.scm"
evaluation_links_scm = os.getcwd() + "/results/evaluation-links.scm"
subset_links_scm = os.getcwd() + "/results/subset-links.scm"
attraction_links_scm = os.getcwd() + "/results/attraction-links.scm"
sentences_pickle = os.getcwd() + "/results/sentences.pickle"
deepwalk_bin = os.getcwd() + "/results/deepwalk.bin"
results_csv = os.getcwd() + "/results/results.csv"

course_id_prefix = "course:"
user_id_prefix = "user:"
target_id_prefix = "target:"
feature_prefix = "feature:"

deepwalk = None

### Utils ###
def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

def get_concepts(name_prefix):
  return list(
           filter(
             lambda x : x.name.startswith(name_prefix),
             atomspace.get_atoms_by_type(types.ConceptNode)))

def intensional_difference(c1, c2):
  cn1 = "(Concept \"{}\")".format(c1)
  cn2 = "(Concept \"{}\")".format(c2)
  intdiff = "(IntensionalDifference {} {})".format(cn1, cn2)
  scm("(pln-bc {})".format(intdiff))
  tv_mean = float(scm("(cog-mean {})".format(intdiff)))
  tv_conf = float(scm("(cog-confidence {})".format(intdiff)))
  return TruthValue(tv_mean, tv_conf)

### Initialize the AtomSpace ###
atomspace = AtomSpace()
initialize_opencog(atomspace)

### Guile setup ###
scm("(add-to-load-path \"/usr/share/guile/site/2.2/opencog\")")
scm("(add-to-load-path \".\")")
scm("(use-modules (opencog) (opencog bioscience) (opencog ure) (opencog pln))")
scm(" ".join([
  "(define (write-atoms-to-file file atoms)",
    "(define fp (open-output-file file))",
    "(for-each",
      "(lambda (x) (display x fp))",
      "atoms)",
    "(close-port fp))"]))

def load_all_atomes():
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
      memblinks = filter(lambda x : x.type == types.MemberLink and x.out[1] == node, node.incoming)
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
  scm("(pln-add-rule-by-name \"subset-condition-negation-rule\")")
  # (Subset A B) (Subset (Not A) B) |- (Attraction A B)
  scm("(pln-add-rule-by-name \"subset-attraction-introduction-rule\")")
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
  sentences = []

  def get_reverse_pred(pred):
    if pred == "has_action_target":
      return "is_an_action_target_of"
    elif pred == "has_action_feature":
      return "is_an_action_feature_of"
    elif pred == "has":
      return "was_the_result_of"
    else:
      raise Exception("The reverse of predicate \"{}\" is not defined!".format(pred))

  if os.path.exists(sentences_pickle):
    print("--- Loading generated sentences from \"{}\"".format(sentences_pickle))
    with open(sentences_pickle, "rb") as f:
      sentences = pickle.load(f)
  else:
    next_word_dict = {}

    def add_to_next_word_dict(w, nw):
      if next_word_dict.get(w):
        next_word_dict[w].add(nw)
      else:
        next_word_dict[w] = {nw}

    print("--- Gathering next words")
    evalinks = atomspace.get_atoms_by_type(types.EvaluationLink)
    for evalink in evalinks:
      pred = evalink.out[0].name
      rev_pred = get_reverse_pred(pred)
      source = evalink.out[1].out[0].name
      target = evalink.out[1].out[1].name
      add_to_next_word_dict(source, pred)
      add_to_next_word_dict(pred, target)
      add_to_next_word_dict(target, rev_pred)
      add_to_next_word_dict(rev_pred, source)
    for k, v in next_word_dict.items():
      next_word_dict[k] = tuple(v)

    print("--- Generating sentences")
    num_sentences = 10000000
    sentence_length = 15
    first_words = [x.name for x in get_concepts(user_id_prefix)]
    for i in range(num_sentences):
      sentence = []
      for j in range(sentence_length):
        if j == 0:
          sentence.append(random.choice(first_words))
        else:
          last_word = sentence[-1]
          next_words = next_word_dict.get(last_word)
          sentence.append(random.choice(next_words))
      sentences.append(sentence)
      if len(sentences) % 10000 == 0:
        print(len(sentences))
    with open(sentences_pickle, "wb") as f:
      pickle.dump(sentences, f)

    print("--- Training model")
    deepwalk = Word2Vec(sentences, min_count=1)

def compare():
  print("--- Comparing PLN vs DW")

  node_pattern_dict = {}
  def get_patterns(node):
    def get_subsets(node):
      return list(
               filter(
                 lambda x : x.type == types.SubsetLink and x.out[0] == node,
                 node.incoming))
    if node_pattern_dict.get(node):
      return node_pattern_dict[node]
    else:
      pats = [x.out[1] for x in get_subsets(node)]
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
  scm("(pln-load-from-path \"intensional-difference-direct-introduction-mooc.scm\")")
  scm("(pln-add-rule-by-name \"intensional-difference-direct-introduction-rule-mooc\")")

  # Output file
  results_csv_fp = open(results_csv, "w")
  first_row = ",".join([
    "Pair",
    "P1 patterns",
    "P2 patterns",
    "Common patterns",
    "Common patterns (%)",
    "IntDiff strength",
    "IntDiff confidence",
    "Vector distance"])
  results_csv_fp.write(first_row + "\n")

  # Generate the results
  for pair in user_pairs:
    p1 = pair[0]
    p2 = pair[1]
    p1_patterns = get_patterns(ConceptNode(p1))
    p2_patterns = get_patterns(ConceptNode(p2))
    p1_pattern_size = len(p1_patterns)
    p2_pattern_size = len(p2_patterns)
    common_patterns = set(p1_patterns).intersection(p2_patterns)
    common_pattern_size = len(common_patterns)
    common_pattern_percent = (common_pattern_size / ((p1_pattern_size + p2_pattern_size) / 2)) * 100
    # PLN intensional difference
    intdiff_tv = intensional_difference(p1, p2)
    tv_mean = intdiff_tv.mean
    tv_conf = intdiff_tv.confidence
    # DeepWalk euclidean distance
    v1 = deepwalk[p1]
    v2 = deepwalk[p2]
    vec_dist = distance.euclidean(v1, v2)
    row = ",".join([
      p1 + " " + p2,
      str(p1_pattern_size),
      str(p2_pattern_size),
      str(common_pattern_size),
      str(common_pattern_percent),
      str(tv_mean),
      str(tv_conf),
      str(vec_dist)])
    results_csv_fp.write(row + "\n")
  results_csv_fp.close()

### Main ###
load_all_atomes()
load_deepwalk_model()

# populate_atomspace()
# generate_subsets()
# calculate_truth_values()
# infer_attractions()
# export_all_atoms()
# train_deepwalk_model()
# export_deepwalk_model()

compare()
