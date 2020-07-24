import csv
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
from sklearn.decomposition import PCA

log.set_level("ERROR")

base_results_dir = os.getcwd() + "/results/"

member_links_scm = base_results_dir + "member-links.scm"
evaluation_links_scm = base_results_dir + "evaluation-links.scm"
subset_links_scm = base_results_dir + "subset-links.scm"
attraction_links_scm = base_results_dir + "attraction-links.scm"
sentences_pickle = base_results_dir + "sentences.pickle"
deepwalk_bin = base_results_dir + "deepwalk.bin"
pca_png = base_results_dir + "pca.png"
results_csv = base_results_dir + "results.csv"

subuniverse_prefix = "subuniverse:"
person_prefix = "person:"
property_prefix = "property:"

deepwalk = None
property_vectors = []

num_people = 10
num_properties = 20
num_properties_per_person = 10
num_sentences = 10000000
num_walks = 9

### Utils ###
def scm(atomese):
  return scheme_eval(atomspace, atomese).decode("utf-8")

def get_concepts(str_prefix):
  return [x for x in atomspace.get_atoms_by_type(types.ConceptNode) if x.name.startswith(str_prefix)]

def get_concepts_with_property(prop):
  return execute_atom(atomspace,
           GetLink(
             EvaluationLink(
               PredicateNode("has_property"),
               ListLink(
                 VariableNode("$X"),
                 prop)))).out

def get_concept_properties(concept):
  return execute_atom(atomspace,
           GetLink(
             EvaluationLink(
               PredicateNode("has_property"),
               ListLink(
                 concept,
                 VariableNode("$X"))))).out

def print_property_prevalence():
  property_cnt = 0
  sum_prevalence = 0
  for p in get_concepts(property_prefix):
    prevalence = len(get_concepts_with_property(p))
    property_cnt = property_cnt + 1
    sum_prevalence = sum_prevalence + prevalence
    print("{}, {}".format(p.name, prevalence))
  print("Total no. of people: {}\nTotal no. of properties: {}\nAverage property prevalence: {} ({}%)".format(
      len(get_concepts(person_prefix)),
      property_cnt,
      sum_prevalence/property_cnt,
      100 * sum_prevalence/property_cnt/num_people))

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
def populate_atomspace():
  print("--- Populating the AtomSpace")

  # Create people and the properties linked to them
  for i in range(0, num_people):
    subuniverse_conceptnode = ConceptNode(subuniverse_prefix + str(i))
    person_conceptnode = ConceptNode(person_prefix + str(i))

    MemberLink(
      subuniverse_conceptnode,
      person_conceptnode)

    for j in random.sample(range(num_properties), num_properties_per_person):
      property_conceptnode = ConceptNode(property_prefix + str(j))

      MemberLink(
        subuniverse_conceptnode,
        property_conceptnode)

      EvaluationLink(
        PredicateNode("has_property"),
        ListLink(
          person_conceptnode,
          property_conceptnode))

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
  universe_size = len(get_concepts(subuniverse_prefix))
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
  next_words_dict = {}
  sentences = []

  def add_to_next_words_dict(w, nw):
    if next_words_dict.get(w):
      next_words_dict[w].add(nw)
    else:
      next_words_dict[w] = {nw}

  def get_reverse_pred(pred):
    if pred == "has_property":
      return "is_a_property_of"

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
  first_words = [x.name for x in get_concepts(person_prefix)]
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

def build_property_vectors():
  global property_vectors
  all_properties = get_concepts(property_prefix)

  for person in get_concepts(person_prefix):
    pvec = []
    properties = get_concept_properties(person)
    for p in all_properties:
      if p in properties:
        pvec.append(1)
      else:
        pvec.append(0)
    property_vectors.append(pvec)

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

def compare():
  print("--- Comparing PLN vs DW")

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

  # Get the pairs
  print("--- Generating pairs")
  people = [x.name for x in get_concepts(person_prefix)]
  people_pairs = []
  for person1 in people:
    for person2 in people:
      if person1 != person2:
        people_pairs.append((person1, person2))

  print("--- Generating results")
  # PLN setup
  scm("(pln-load 'empty)")
  scm("(pln-add-rule-by-name \"intensional-similarity-direct-introduction-rule\")")

  # Output file
  results_csv_fp = open(results_csv, "w")
  first_row = ",".join([
    "Person 1",
    "Person 2",
    "No. of person 1 properties",
    "No. of person 2 properties",
    "No. of common properties",
    "Intensional Similarity",
    "Vector distance",
    "Pearson",
    "Spearman",
    "Kendall"])
  results_csv_fp.write(first_row + "\n")

  # Generate the results
  for pair in people_pairs:
    p1 = pair[0]
    p2 = pair[1]
    p1_properties = get_properties(ConceptNode(p1))
    p2_properties = get_properties(ConceptNode(p2))
    p1_pattern_size = len(p1_properties)
    p2_pattern_size = len(p2_properties)
    common_properties = set(p1_properties).intersection(p2_properties)
    common_pattern_size = len(common_properties)
    # PLN intensional similarity
    intsim_tv = intensional_similarity(p1, p2).mean if intensional_similarity(p1, p2).confidence > 0 else 0
    # DeepWalk euclidean distance
    v1 = deepwalk[p1]
    v2 = deepwalk[p2]
    vec_dist = distance.euclidean(v1, v2)
    row = ",".join([
      p1,
      p2,
      str(p1_pattern_size),
      str(p2_pattern_size),
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

  results_csv_fp.close()
