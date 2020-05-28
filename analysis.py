import os
from opencog.atomspace import AtomSpace, types
from opencog.scheme_wrapper import scheme_eval
from opencog.type_constructors import *
from opencog.utilities import initialize_opencog

mooc_actions_tsv = os.getcwd() + "/datasets/mooc_actions.tsv"
mooc_action_labels_tsv = os.getcwd() + "/datasets/mooc_action_labels.tsv"
mooc_action_features_tsv = os.getcwd() + "/datasets/mooc_action_features.tsv"
mooc_all_scm = os.getcwd() + "/datasets/mooc_all.scm"

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

### Initialize the AtomSpace ###
atomspace = AtomSpace()
initialize_opencog(atomspace)

### Guile setup ###
scm("(add-to-load-path \"/usr/share/guile/site/2.2/opencog\")")
scm("(add-to-load-path \".\")")
scm("(use-modules (opencog) (opencog bioscience) (opencog pln))")

### Load dataset ###
print("--- Loading dataset...")
if not os.path.isfile(mooc_all_scm):
  mooc_all_scm_fp = open(mooc_all_scm, "w")

  with open(mooc_actions_tsv) as f:
    next(f)
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      user_id = content[1].strip()
      target_id = content[2].strip()
      atomese_1 = evalink("takes", user_id_prefix + user_id, action_id_prefix + action_id)
      atomese_2 = evalink("has_target", action_id_prefix + action_id, target_id_prefix + target_id)
      mooc_all_scm_fp.write(atomese_1)
      mooc_all_scm_fp.write(atomese_2)
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
        mooc_all_scm_fp.write(atomese)
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
      mooc_all_scm_fp.write(atomese_1)
      mooc_all_scm_fp.write(atomese_2)
      mooc_all_scm_fp.write(atomese_3)
      mooc_all_scm_fp.write(atomese_4)
      scm(atomese_1)
      scm(atomese_2)
      scm(atomese_3)
      scm(atomese_4)

  mooc_all_scm_fp.close()
else:
  scm("(use-modules (opencog persist-file))")
  scm("(load-file \"" + mooc_all_scm + "\")")

### Pre-processing ###
# TODO: Use/Turn the below into actual PLN rules
print("--- Translating links...")
# Minimum translation -- directly turn EvaluationLink relations into SubsetLinks, which generates
# what's needed for the experiment. The satisfying sets (meta-concepts) will not be generated here.
for el in atomspace.get_atoms_by_type(types.EvaluationLink):
  source = el.out[1].out[0]
  target = el.out[1].out[1]
  SubsetLink(target, source)
