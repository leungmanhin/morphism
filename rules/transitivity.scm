;; Crisp rules about transitivity of predicates or inheritance links

(load-from-path "rules/rule-utils.scm")

;; Helpers
(define ConceptT (TypeInh "ConceptNode"))
(define GeneT (Type "GeneNode"))

;; Subset transitivity (deduction) rule
(define present-subset-transitivity-rule
  (gen-present-link-transitivity-rule SubsetLink ConceptT))
(define present-subset-transitivity-rule-name
  (DefinedSchemaNode "present-subset-transitivity-rule"))
(DefineLink present-subset-transitivity-rule-name
  present-subset-transitivity-rule)

;; Mixed (Member A B), (Subset B C) |- (Member A C)
(define present-mixed-member-subset-transitivity-rule
  (gen-present-mixed-link-transitivity-rule MemberLink
                                            SubsetLink
                                            GeneT ConceptT ConceptT))
(define present-mixed-member-subset-transitivity-rule-name
  (DefinedSchemaNode "present-mixed-member-subset-transitivity-rule"))
(DefineLink present-mixed-member-subset-transitivity-rule-name
  present-mixed-member-subset-transitivity-rule)
