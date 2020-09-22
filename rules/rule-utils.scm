;; Useful rule generators

;;;;;;;;;;;;;;;;;
;; Translation ;;
;;;;;;;;;;;;;;;;;

;; Rule generator for the translation of LINK-TYPE-1 to LINK-TYPE-2
;;
;; (LINK-TYPE-1 X Y)
;; |-
;; (LINK-TYPE-2 X Y)
(define (gen-present-link-translation-rule LINK-TYPE-1 LINK-TYPE-2 VAR-TYPE)
  (let* ((X (Variable "$X"))
         (Y (Variable "$Y"))
         (XY-1 (LINK-TYPE-1 X Y))
         (XY-2 (LINK-TYPE-2 X Y)))
    (Bind
      (VariableList
        (TypedVariable X VAR-TYPE)
        (TypedVariable Y VAR-TYPE))
      (Present
        XY-1)
      XY-2)))

;;;;;;;;;;;;;;;;;;
;; Transitivity ;;
;;;;;;;;;;;;;;;;;;

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
