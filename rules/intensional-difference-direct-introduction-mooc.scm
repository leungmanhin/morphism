;; NOTE: This is a copy of the original intensional-difference-direct-introduction rule,
;; in the PLN repo, with a slight modification to fit the dataset in this experiment

;; Rule
(define intensional-difference-direct-introduction-rule-mooc
  (let* ((A (Variable "$A"))
         (B (Variable "$B"))
         (X (Variable "$X"))
         (CT (Type "ConceptNode")))
    (Bind
      (VariableSet
        (TypedVariable A CT)
        (TypedVariable B CT))
      (And
        (Present
          A
          B)
        ;; There exists X such that
        ;;
        ;; (Attraction A X)
        ;; (Attraction B X)
        ;;
        ;; are present in the atomspace
        (Satisfaction
          (TypedVariable X CT)
          (Present
            (Attraction A X)
            (Attraction B X)))
        ;; A and B are different
        (Not (Equal A B)))
      (ExecutionOutput
        (GroundedSchema "scm: intensional-difference-direct-introduction-mooc")
        (List
          ;; Conclusion
          (IntensionalDifference A B)
          ;; Premises
          A
          B)))))

;; Formula
(define (intensional-difference-direct-introduction-mooc conclusion . premises)
  ;; Given a concept return all attraction link
  ;;
  ;; Attraction <TV>
  ;;   A
  ;;   X
  (define (get-attractions A)
    (let* ((at-links (cog-filter 'AttractionLink (cog-incoming-set A)))
           (A-at? (lambda (x) (equal? A (gar x)))))
      (filter A-at? at-links)))

  ;; The pattern strength is the product of the mean and the
  ;; confidence of the TV on the attraction link
  ;;
  ;; Attraction <TV>
  ;;   A
  ;;   X
  (define (get-pattern-strength A pat)
    (let* ((A-at (cog-link 'AttractionLink A pat)))
      (if (null? A-at) 0 (* (cog-mean A-at) (cog-confidence A-at)))))

  ;; Given the attraction links of A and B calculate the fuzzy
  ;; difference between the patterns of A and B, expressed as
  ;;
  ;; Sum_x min(pattern-of(X,A), 1 - pattern-of(X,B))
  ;;
  ;; where pattern-of(X,A) is the strength of the TV of
  ;;
  ;; Attraction <TV>
  ;;   A
  ;;   X
  (define (numerator A-ats B)
    (define (fuzzy-difference A-at)
      (let* ((pat (gdr A-at)))
        (min (cog-mean A-at) (- 1 (get-pattern-strength B pat)))))
    (fold + 0 (map fuzzy-difference A-ats)))

  (define (get-courses)
    (filter
      (lambda (x) (string-prefix? "course:" (cog-name x)))
      (cog-get-atoms 'ConceptNode)))

  ;; (cog-logger-debug "(intensional-difference-direct-introduction conclusion=~a . premises=~a)" conclusion premises)
  (if (= (length premises) 2)
      (let* ((IntDiff conclusion)
             (A (car premises))
             (B (cadr premises))
             ;; Fetch all pattern attraction links and patterns
             (A-ats (get-attractions A))
             (B-ats (get-attractions B))
             (A-pats (map gdr A-ats))
             (usize (length (get-courses)))  ; Universe size
             ;; Calculate denominator, numerator and TV
             (dnt usize)
             (TVs (if (< 0 dnt) (/ (numerator A-ats B) dnt) 1))
             (TVc (count->confidence dnt))
             (TV (stv TVs TVc)))
        (if (< 0 TVc) (cog-merge-hi-conf-tv! IntDiff TV)))))

; Name the rule
(define intensional-difference-direct-introduction-rule-name-mooc
  (DefinedSchemaNode "intensional-difference-direct-introduction-rule-mooc"))
(DefineLink intensional-difference-direct-introduction-rule-name-mooc
  intensional-difference-direct-introduction-rule-mooc)
