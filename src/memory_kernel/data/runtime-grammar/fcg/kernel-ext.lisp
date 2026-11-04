;;;; kernel-ext.lisp -- the memory kernel's extensions to the FCG Go engine for the runtime grammar.
;;;;
;;;; mk-kernel-ext version 1, pinned to the FCG Go image 2cc1f08127cf (LispWorks 8.1.0).
;;;;
;;;; runtime/engine.py sends this file at every engine start through /fcg-read-eval-print, in package FCG. It refuses
;;;; to start an engine whose binary is not the pinned image, and it refuses an answer other than T. The route reads
;;;; ONE form, so the file is a single PROGN. Every FCG symbol is written package-qualified (fcg::): a single-colon
;;;; reference to an unexported symbol makes the route answer a bare HTTP 400. Each extension has a test against the
;;;; pinned image in tests/unit/runtime/test_engine_search.py.
;;;;
;;;; This is the search setup "R:KSH-A1c-nodup" of the CLEVR search spike, pinned in the 2026-10-19 runtime grammar
;;;; amendment, section 3.2. The grammar configuration that uses these extensions is
;;;; runtime/engine.py's unit_configuration:
;;;;   :mk-budget              node test: a node cap plus a per-request deadline; when it trips it EMPTIES the queue,
;;;;                           so the search ends inside the engine;
;;;;   :mk-canonical           node test: inside one tier, only one order of independent applications is kept;
;;;;   :mk-trigger             hash mode: a construction files under the triggers it carries, never under NIL;
;;;;   :mk-no-strings-in-root  goal test: every word of the turn was consumed (FCG Go ships no such test);
;;;;   mk-comprehend-units     the bounded comprehension of a raw form (one turn's units).
;;;;
;;;; The raw form names its units w1 .. wn, 1-based, in token order: :mk-canonical reads the token index from that name.

(progn
 ;; ------------------------------------------------------------------------------------------------ per-request state
 ;; Bound by mk-comprehend-units for one request (special variables, so per Hunchentoot worker thread).
 (defvar fcg::*mk-deadline* nil
   "Internal real time after which the :mk-budget node test trips, or NIL (no deadline).")
 (defvar fcg::*mk-tripped* nil
   "Which budget limit tripped during this request: \"time\", \"nodes\" or NIL.")
 (defvar fcg::*mk-triggers* nil
   "The turn's trigger keys (mk-form-triggers of its initial raw form): the node side of the :mk-trigger hash.")

 ;; ------------------------------------------------------------------------------------------------ :mk-budget
 ;; The stock :restrict-nr-of-nodes only stops children from being enqueued: the queue keeps draining and every
 ;; queued parent keeps calling fcg-apply, and a client that gives up does not cancel its Hunchentoot worker. This
 ;; test empties the processor's queue and marks the parent fully expanded instead, so next-cip-solution's loop ends
 ;; by itself and returns the best node, with goal-test-failed. It runs between expansions: the overshoot is at most
 ;; one node expansion. Place it FIRST in :node-tests.
 (defmethod fcg::cip-node-test ((node fcg::cip-node) (mode (eql :mk-budget)))
   (let ((why (cond ((and fcg::*mk-deadline* (> (get-internal-real-time) fcg::*mk-deadline*)) "time")
                    ((>= (fcg::created-at node)
                         (fcg::get-configuration (fcg::construction-inventory (fcg::cip node)) :max-nr-of-nodes))
                     "nodes"))))
     (if why
         (progn (setf fcg::*mk-tripped* why)
                (push 'fcg::mk-budget-exhausted (fcg::statuses node))
                (setf (fcg::queue (fcg::cip node)) nil)
                (when (fcg::parent node) (setf (fcg::fully-expanded? (fcg::parent node)) t))
                nil)
         t)))

 ;; ------------------------------------------------------------------------------------------------ :mk-canonical
 ;; Partial-order reduction inside a tier. Independent applications in one tier commute, so only the order whose
 ;; right edge does not decrease is kept. The right edge of an application is the largest token index under the units
 ;; its bindings name. A dependent application spans its dependency, so its right edge is >= the dependency's, and
 ;; every derivation keeps one ordering; alternatives at the same position still branch. The sequentially applied
 ;; tiers are exempt. A rejected node also gets its `duplicate` slot set, else :check-duplicate would flag the kept
 ;; ordering as a duplicate of the dropped one (the pinned setup runs without :check-duplicate).
 (defun fcg::mk-yield-right (name ts depth)
   "Largest token index under unit NAME of the transient structure TS (w<k> = token k), -1 when none."
   (let ((sn (symbol-name name)))
     (if (and (> (length sn) 1) (char-equal (char sn 0) #\W) (every #'digit-char-p (subseq sn 1)))
         (parse-integer sn :start 1)
         (let ((u (find name ts :key #'first)))
           (if (or (null u) (> depth 12))
               -1
               (let ((subs (second (find "SUBUNITS" (rest u)
                                         :key (lambda (f) (if (consp f) (symbol-name (first f)) ""))
                                         :test #'string=))))
                 (reduce #'max (mapcar (lambda (s) (if (symbolp s) (fcg::mk-yield-right s ts (1+ depth)) -1))
                                       (if (listp subs) subs nil))
                         :initial-value -1)))))))
 (defun fcg::mk-right (node)
   "Right edge of the application that created NODE."
   (let ((ts (fcg::left-pole-structure (fcg::car-resulting-cfs (fcg::cipn-car node)))))
     (reduce #'max (mapcar (lambda (b)
                             (let ((v (cdr b)))
                               (if (and (symbolp v) v (find v ts :key #'first)) (fcg::mk-yield-right v ts 0) -1)))
                           (fcg::car-match-bindings (fcg::cipn-car node)))
             :initial-value -1)))
 (defun fcg::mk-label (node)
   (let ((c (first (fcg::applied-constructions node))))
     (when c (symbol-name (fcg::attr-val c :label)))))
 (defmethod fcg::cip-node-test ((node fcg::cip-node) (mode (eql :mk-canonical)))
   (let* ((p (fcg::parent node)) (lc (fcg::mk-label node)) (lp (and p (fcg::mk-label p))))
     (if (and lc lp (string= lc lp)
              (not (member lc (fcg::get-configuration (fcg::construction-inventory (fcg::cip node))
                                                      :cxn-sets-with-sequential-application)
                           :key #'symbol-name :test #'string-equal))
              (< (fcg::mk-right node) (fcg::mk-right p)))
         (progn (push 'fcg::non-canonical (fcg::statuses node)) (setf (fcg::duplicate node) node) nil)
         t)))

 ;; ------------------------------------------------------------------------------------------------ :mk-trigger
 ;; Construction side: the construction's :triggers attribute, a list of strings derived from its own conditions by
 ;; the compiler. A construction without one is refused when it is added (it would file under the NIL key, and
 ;; constructions-for-application-hashed offers the NIL bucket at every node).
 (defmethod fcg::hash ((c fcg::construction) (mode (eql :mk-trigger)) &key &allow-other-keys)
   (let ((ts (fcg::attr-val c :triggers)))
     (unless ts
       (error "mk-trigger: construction ~(~a~) has no :triggers (it would file under the NIL hash key)"
              (fcg::name c)))
     (remove-duplicates (mapcar (lambda (x) (string-downcase (format nil "~a" x))) (if (listp ts) ts (list ts)))
                        :test #'string=)))
 ;; Node side: the turn's trigger set, computed once per request from the INITIAL form (the lexical constructions
 ;; consume the cat predicates, so a node's own root no longer holds them).
 (defmethod fcg::hash ((node fcg::cip-node) (mode (eql :mk-trigger)) &key &allow-other-keys)
   fcg::*mk-triggers*)
 (defun fcg::mk-form-triggers (form)
   "Trigger keys of a raw form: every string; cat:X; acat:X, which also adds cat:X (so the nominal and verb-phrase
    constructions stay offered for an ambiguous word); lem:x for every lemma predicate."
   (let ((out nil))
     (dolist (p form out)
       (let ((h (symbol-name (first p))))
         (cond ((string= h "STRING") (push (string-downcase (third p)) out))
               ((string= h "CAT") (push (concatenate 'string "cat:" (string-downcase (third p))) out))
               ((string= h "ACAT")
                (push (concatenate 'string "acat:" (string-downcase (third p))) out)
                (push (concatenate 'string "cat:" (string-downcase (third p))) out))
               ((member h '("NLEMMA" "VLEMMA" "LEMMA") :test #'string=)
                (push (concatenate 'string "lem:" (string-downcase (format nil "~a" (third p)))) out)))))))

 ;; ------------------------------------------------------------------------------------------------ goal test
 (defmethod fcg::cip-goal-test ((node fcg::cip-node) (mode (eql :mk-no-strings-in-root)))
   "Every string predicate of the raw form was consumed by some construction (no word left unused)."
   (let* ((ts (fcg::left-pole-structure (fcg::car-resulting-cfs (fcg::cipn-car node))))
          (root (find "ROOT" ts :key (lambda (u) (symbol-name (first u))) :test #'string=))
          (form (second (find "FORM" (rest root) :key (lambda (f) (symbol-name (first f))) :test #'string=))))
     (notany (lambda (p) (string= (symbol-name (first p)) "STRING")) form)))

 ;; ------------------------------------------------------------------------------------------------ entry
 (defun fcg::mk-atom (x)
   "A meaning argument as plain data: a string as (\"s\" x), a number as itself, NIL as \"nil\", a symbol as its
    lowercased name."
   (cond ((stringp x) (list "s" x))
         ((numberp x) x)
         ((null x) "nil")
         ((symbolp x) (string-downcase (symbol-name x)))
         (t (format nil "~a" x))))
 (defun fcg::mk-comprehend-units (inv form deadline-ms)
   "Comprehend the raw FORM with the inventory INV under the :mk-budget node test, DEADLINE-MS milliseconds from now
    (NIL: no deadline; the inventory's :max-nr-of-nodes is the node cap), with the :mk-trigger node keys of FORM.
    The id counters are reset first, so the same form on the same constructions gives the same meaning.
    Answers (meaning applied statuses left ms nodes tripped): applied in application order, left = the strings
    still in the root, ms = the engine's own time, nodes = nodes created, tripped = \"time\", \"nodes\" or \"\"."
   (utils::reset-id-counters)
   (setf *gensym-counter* 0)
   (let* ((t0 (get-internal-real-time))
          (fcg::*mk-deadline*
            (when deadline-ms (+ t0 (round (* deadline-ms internal-time-units-per-second) 1000))))
          (fcg::*mk-tripped* nil)
          (fcg::*mk-triggers* (fcg::mk-form-triggers form)))
     (multiple-value-bind (m node) (fcg::comprehend form :cxn-inventory inv :silent t)
       (let* ((ms (round (* 1000 (- (get-internal-real-time) t0)) internal-time-units-per-second))
              (ts (when node (fcg::left-pole-structure (fcg::car-resulting-cfs (fcg::cipn-car node)))))
              (root (find "ROOT" ts :key (lambda (u) (symbol-name (first u))) :test #'string=))
              (rootform (second (find "FORM" (rest root) :key (lambda (f) (symbol-name (first f))) :test #'string=)))
              (left (remove-if-not (lambda (p) (string= (symbol-name (first p)) "STRING")) rootform)))
         (list (mapcar (lambda (p) (mapcar #'fcg::mk-atom p)) m)
               (when node
                 (reverse (mapcar (lambda (c) (string-downcase (symbol-name (fcg::name c))))
                                  (fcg::applied-constructions node))))
               (when node (mapcar (lambda (s) (string-downcase (symbol-name s))) (fcg::statuses node)))
               (mapcar #'third left)
               ms
               (if node (fcg::node-counter (fcg::cip node)) 0)
               (or fcg::*mk-tripped* ""))))))

 (defun fcg::mk-kernel-ext ()
   "This file's name, version and the FCG Go image (sha256, first 12 hex digits) it is pinned to."
   (list "mk-kernel-ext" 1 "2cc1f08127cf"))
 t)
