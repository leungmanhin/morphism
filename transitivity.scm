;;; Content in this file is copied from https://github.com/ngeiswei/reasoning-bio-as-xp/tree/master/rules

(define ConceptT (Type "ConceptNode"))

;; Rule generator for the transitivity of LINK-TYPE (equivalent to a
;; crisp deduction rule).
;;
;; (LINK-TYPE X Y)
;; (LINK-TYPE Y Z)
;; |-
;; (LINK-TYPE X Z)
(define (gen-present-link-transitivity-rule LINK-TYPE VAR-TYPE)
  (let* ((X (Variable "$X"))
         (Y (Variable "$Y"))
         (Z (Variable "$Z"))
         (XY (LINK-TYPE X Y))
         (YZ (LINK-TYPE Y Z))
         (XZ (LINK-TYPE X Z)))
    (Bind
      (VariableList
        (TypedVariable X VAR-TYPE)
        (TypedVariable Y VAR-TYPE)
        (TypedVariable Z VAR-TYPE))
      (And
        (Present
          XY
          YZ)
        (Not (Identical X Z)))
      XZ)))

;; Rule generator for the transitivity of some link type via some
;; other link type. Specifically
;;
;; (LINK-TYPE-1 X Y)
;; (LINK-TYPE-2 Y Z)
;; |-
;; (LINK-TYPE-1 X Z)
(define (gen-present-mixed-link-transitivity-rule LINK-TYPE-1 LINK-TYPE-2
                                                  X-TYPE Y-TYPE Z-TYPE)
  (let* ((X (Variable "$X"))
         (Y (Variable "$Y"))
         (Z (Variable "$Z"))
         (XY (LINK-TYPE-1 X Y))
         (YZ (LINK-TYPE-2 Y Z))
         (XZ (LINK-TYPE-1 X Z)))
    (Bind
      (VariableList
        (TypedVariable X X-TYPE)
        (TypedVariable Y Y-TYPE)
        (TypedVariable Z Z-TYPE))
      (And
        (Present
          XY
          YZ)
        (Not (Identical X Z)))
      XZ)))

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
                                            ConceptT ConceptT ConceptT))
(define present-mixed-member-subset-transitivity-rule-name
  (DefinedSchemaNode "present-mixed-member-subset-transitivity-rule"))
(DefineLink present-mixed-member-subset-transitivity-rule-name
  present-mixed-member-subset-transitivity-rule)
