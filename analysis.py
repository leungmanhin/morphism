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
atomese_mooc_scm = os.getcwd() + "/datasets/atomese-mooc.scm"
atomese_preprocessed_scm = os.getcwd() + "/results/atomese-preprocessed.scm"
deepwalk_sentences = os.getcwd() + "/datasets/deepwalk_sentences.pickle"
deepwalk_model = os.getcwd() + "/results/deepwalk.bin"
results_csv = os.getcwd() + "/results/results.csv"

user_id_prefix = "user:"
action_id_prefix = "action:"
target_id_prefix = "target:"
feature_prefix = "feature:"

def evalink(pred, node1, node2):
  return "\n".join(["(EvaluationLink",
                    "\t(PredicateNode \"" + pred + "\")",
                    "\t(ListLink",
                    "\t\t(ConceptNode \"" + node1 + "\")",
                    "\t\t(ConceptNode \"" + node2 + "\")))\n"])

def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

def get_concepts(name_prefix):
  return list(
           filter(
             lambda x : x.name.startswith(name_prefix),
             atomspace.get_atoms_by_type(types.ConceptNode)))

def get_subsets(node):
  return list(
           filter(
             lambda x : x.type == types.SubsetLink and x.out[1] == node,
             node.incoming))

def get_patterns(node):
  return [x.out[0] for x in get_subsets(node)]

def tv_mean(node, usize):
  return len(get_patterns(node)) / usize

def tv_confidence(cnt):
  return float(scm("(count->confidence " + str(cnt) + ")"))

def get_reverse_pred(pred):
  if pred == "takes":
    return "is_taken_by"
  elif pred == "has_target":
    return "is_a_target_of"
  elif pred == "has_feature":
    return "is_a_feature_of"
  elif pred == "leads_to":
    return "is_led_by"
  else:
    raise Exception("The reverse of predicate \"{}\" is not defined!".format(pred))

### Initialize the AtomSpace ###
atomspace = AtomSpace()
initialize_opencog(atomspace)

### Guile setup ###
scm("(add-to-load-path \"/usr/share/guile/site/2.2/opencog\")")
scm("(add-to-load-path \".\")")
scm("(use-modules (opencog) (opencog bioscience) (opencog pln) (opencog persist-file))")
scm("(load \"utils.scm\")")

### Load dataset ###
if not os.path.isfile(atomese_mooc_scm):
  print("--- Generating Atomese from dataset...")
  atomese_mooc_scm_fp = open(atomese_mooc_scm, "w")

  with open(mooc_actions_tsv) as f:
    next(f)
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      user_id = content[1].strip()
      target_id = content[2].strip()
      atomese_1 = evalink("takes", user_id_prefix + user_id, action_id_prefix + action_id)
      atomese_2 = evalink("has_target", action_id_prefix + action_id, target_id_prefix + target_id)
      atomese_mooc_scm_fp.write(atomese_1)
      atomese_mooc_scm_fp.write(atomese_2)
      scm(atomese_1)
      scm(atomese_2)

  with open(mooc_action_labels_tsv) as f:
    next(f)
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      label = content[1].strip()
      if label == "1":
        atomese = evalink("leads_to", action_id_prefix + action_id, "dropout")
        atomese_mooc_scm_fp.write(atomese)
        scm(atomese)

  features = []
  with open(mooc_action_features_tsv) as f:
    def process_feature(action_id, feature):
      if feature not in features:
        features.append(feature)
      return evalink("has_feature",
                     action_id_prefix + action_id,
                     feature_prefix + str(features.index(feature)))
    next(f)
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      feature_0 = content[1].strip()
      feature_1 = content[2].strip()
      feature_2 = content[3].strip()
      feature_3 = content[4].strip()
      atomese_1 = process_feature(action_id, feature_0)
      atomese_2 = process_feature(action_id, feature_1)
      atomese_3 = process_feature(action_id, feature_2)
      atomese_4 = process_feature(action_id, feature_3)
      atomese_mooc_scm_fp.write(atomese_1)
      atomese_mooc_scm_fp.write(atomese_2)
      atomese_mooc_scm_fp.write(atomese_3)
      atomese_mooc_scm_fp.write(atomese_4)
      scm(atomese_1)
      scm(atomese_2)
      scm(atomese_3)
      scm(atomese_4)

  atomese_mooc_scm_fp.close()
else:
  print("--- Loading dataset Atomese from \"{}\"...".format(atomese_mooc_scm))
  scm("(load-file \"" + atomese_mooc_scm + "\")")

### Pre-processing ###
if os.path.isfile(atomese_preprocessed_scm):
  print("--- Loading PLN-preprocessed Atomese from \"{}\"...".format(atomese_preprocessed_scm))
  scm("(load-file \"" + atomese_preprocessed_scm + "\")")
else:
  print("--- Translating EvaluationLinks into SubsetLinks...")
  # Minimum translation -- directly turn EvaluationLink relations into SubsetLinks, which generates
  # what's needed for the experiment. The satisfying sets (meta-concepts) will not be generated here.
  # TODO: Turn the below into actual PLN rules
  for el in atomspace.get_atoms_by_type(types.EvaluationLink):
    source = el.out[1].out[0]
    target = el.out[1].out[1]
    SubsetLink(target, source)

  # Infer new subsets via transitivity
  print("--- Inferring new subsets...")
  scm("(pln-load 'empty)")
  scm("(pln-load-from-path \"transitivity.scm\")")
  # (Subset C1 C2) (Subset C2 C3) |- (Subset C1 C3)
  scm("(pln-add-rule-by-name \"present-subset-transitivity-rule\")")
  scm(" ".join(["(pln-fc",
                  "(Subset (Variable \"$X\") (Variable \"$Y\"))",
                  "#:vardecl",
                    "(VariableSet",
                      "(TypedVariable (Variable \"$X\") (Type \"ConceptNode\"))",
                      "(TypedVariable (Variable \"$Y\") (Type \"ConceptNode\")))",
                  "#:maximum-iterations 10",
                  "#:fc-full-rule-application #t)"]))

  # Calculate & assign TVs
  print("--- Calculating and assigning TVs...")
  for subsetlink in atomspace.get_atoms_by_type(types.SubsetLink):
    subsetlink.tv = TruthValue(1, 1)
  users = get_concepts(user_id_prefix)
  actions = get_concepts(action_id_prefix)
  user_universe_size = len(atomspace.get_atoms_by_type(types.ConceptNode)) - len(users)
  action_universe_size = user_universe_size - len(actions)
  for user in users:
    tv = TruthValue(tv_mean(user, user_universe_size), tv_confidence(user_universe_size))
    user.tv = tv
  for action in actions:
    tv = TruthValue(tv_mean(action, action_universe_size), tv_confidence(action_universe_size))
    action.tv = tv

  # Infer inverse SubsetLinks
  print("--- Inferring inverse SubsetLinks...")
  scm("(map true-subset-inverse (cog-get-atoms 'SubsetLink))")

  # Infer AttractionLinks
  print("--- Inferring AttractionLinks...")
  scm("(pln-load 'empty)")
  scm("(pln-add-rule-by-name \"subset-condition-negation-rule\")")
  scm("(pln-add-rule-by-name \"subset-attraction-introduction-rule\")")
  scm(" ".join(["(pln-bc",
                  "(Attraction (Variable \"$X\") (Variable \"$Y\"))",
                  "#:vardecl",
                    "(VariableSet",
                      "(TypedVariable (Variable \"$X\") (Type \"ConceptNode\"))",
                      "(TypedVariable (Variable \"$Y\") (Type \"ConceptNode\")))",
                  "#:maximum-iterations 12",
                  "#:complexity-penalty 10)"]))

### DeepWalk ###
# Generate the sentences for model training
sentences = []
if os.path.exists(deepwalk_sentences):
  print("--- Loading generated sentences from \"{}\"...".format(deepwalk_sentences))
  with open(deepwalk_sentences, "rb") as f:
    sentences = pickle.load(f)
else:
  print("--- Generating sentences...")
  evalinks = atomspace.get_atoms_by_type(types.EvaluationLink)
  for evalink in evalinks:
    pred = evalink.out[0].name
    rev_pred = get_reverse_pred(pred)
    source = evalink.out[1].out[0].name
    target = evalink.out[1].out[1].name
    sentences.append([source, pred, target])
    sentences.append([target, rev_pred, source])
  print("--- Total number of sentences generated: {}".format(len(sentences)))
  with open(deepwalk_sentences, "wb") as f:
    pickle.dump(sentences, f)

# Train a Word2Vec model
if os.path.exists(deepwalk_model):
  print("--- Loading existing model from \"{}\"...".format(deepwalk_model))
  deepwalk = Word2Vec.load(deepwalk_model)
else:
  print("--- Training model...")
  deepwalk = Word2Vec(sentences, min_count=1)
  deepwalk.save(deepwalk_model)

### Compare: PLN vs DW ###
# Get the user pairs
users = [x.name for x in get_concepts(user_id_prefix)]
random.shuffle(users)
user_pairs = list(zip(users[::2], users[1::2]))

# PLN setup
scm("(pln-load 'empty)")
scm("(pln-add-rule-by-name \"intensional-difference-direct-introduction-rule\")")

# Output file
results_csv_fp = open(results_csv, "w")
first_row = ",".join([
  "Pair",
  "P1 patterns",
  "P2 patterns",
  "Common patterns",
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
  # PLN intensional difference
  intdiff = scm(" ".join([
    "(define bc",
      "(gar",
        "(pln-bc",
          "(IntensionalDifference",
            "(Concept \"" + p1 + "\")",
            "(Concept \"" + p2 + "\")))))"]))
  tv_mean = float(scm("(cog-mean bc)"))
  tv_conf = float(scm("(cog-confidence bc)"))
  # DeepWalk euclidean distance
  v1 = deepwalk[p1]
  v2 = deepwalk[p2]
  vec_dist = distance.euclidean(v1, v2)
  row = ",".join([
    p1 + " " + p2,
    str(p1_pattern_size),
    str(p2_pattern_size),
    str(common_pattern_size),
    str(tv_mean),
    str(tv_conf),
    str(vec_dist)])
  results_csv_fp.write(row + "\n")
results_csv_fp.close()
