;;; Content in this file is copied from https://github.com/ngeiswei/reasoning-bio-as-xp/blob/master/bio-as-utils.scm

(define (true-subset-inverse S)
"
  Given a subset with a true value

  Subset (stv 1 1)
    A <ATV>
    B <BTV>

  Return

  Subset <TV>
    B <BTV>
    A <ATV>

  where TV is calculated as follows

  TV.strength = (ATV.strength * ATV.count) / (BTV.strength * BTV.count)
  TV.count = (BTV.strength * BTV.count)

  Which is technically correct since (Subset A B) is true.
"
(let* ((A (gar S))
       (B (gdr S))
       (ATV (cog-tv A))
       (BTV (cog-tv B))
       (A-positive-count (* (cog-tv-mean ATV) (cog-tv-count ATV)))
       (B-positive-count (* (cog-tv-mean BTV) (cog-tv-count BTV)))
       (TV-strength (if (< 0 B-positive-count)
                        (exact->inexact (/ A-positive-count B-positive-count))
                        1))
       (TV-count B-positive-count)
       (TV-confidence (count->confidence TV-count))
       (TV (stv TV-strength TV-confidence)))
  (Subset TV B A)))
