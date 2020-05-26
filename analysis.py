import os

mooc_actions_tsv = os.getcwd() + "/datasets/mooc_actions.tsv"
mooc_action_labels_tsv = os.getcwd() + "/datasets/mooc_action_labels.tsv"
mooc_action_features_tsv = os.getcwd() + "/datasets/mooc_action_features.tsv"
mooc_all_scm = os.getcwd() + "/datasets/mooc_all.scm"

user_id_prefix = "user-"
action_id_prefix = "action-"
target_id_prefix = "target-"
feature_prefix = "feature-"

def evalink(pred, node1, node2):
  return "\n".join(["(Evaluation",
                    "\t(Predicate \"" + pred + "\")",
                    "\t(List",
                    "\t\t(Concept \"" + node1 + "\")",
                    "\t\t(Concept \"" + node2 + "\")))"])

if not os.path.isfile(mooc_all_scm):
  with open(mooc_actions_tsv) as f:
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      user_id = content[1].strip()
      target_id = content[2].strip()
      print(evalink("takes", user_id_prefix + user_id, action_id_prefix + action_id))
      print(evalink("has_target", action_id_prefix + action_id, target_id_prefix + target_id))

  with open(mooc_action_labels_tsv) as f:
    for line in f:
      content = line.split("\t")
      action_id = content[0].strip()
      label = content[1].strip()
      if label == "1":
        print(evalink("leads_to", action_id_prefix + action_id, "dropout"))

  features = []
  with open(mooc_action_features_tsv) as f:
    def process_feature(action_id, feature):
      if feature not in features:
        features.append(feature)
      print(evalink("has_feature",
                    action_id_prefix + action_id,
                    feature_prefix + str(features.index(feature))))
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
