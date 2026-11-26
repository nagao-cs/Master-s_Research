## Recently Done
+ Studied design patterns and refactored the code
+ Basic implementation of the tracker
+ Visualization and evaluation of the tracker
+ Identified the cause of the rapid transition to a two-version state when using the tracker and the direction for improvement -> the inequality was reversed for `uncertainty >= threshold`
+ Detection tests on KITTI
+ Implementation of online execution
+ Set up the basic environment for fine-tuning the model on KITTI
+ Implemented experiments on KITTI
+ Fixed the `invalid value encountered in divide` issue in mAP calculation
+ Created the state transition graph
+ Reached the stage of fine-tuning the YOLO-based model on KITTI
+ Fine-tuned the PyTorch model on KITTI
+ Fixed the broken `vevn`
+ Corrected issues in the tracker section
+ Moved dataset-related work to the D drive
+ Verified that diversity correlates with errors (comparing error rates for objects detected by multiple models vs. those detected by a single model) (model-only evaluation)
+ Verified that diversity correlates with errors (same comparison as above) (including the tracker)
+ Verified that detection reliability changes depending on whether diversity is high or low
+ Analyzed the resolution of errors in only one model -> also analyze the resolution of errors in both models
+ Compared the tracker’s predictions from the previous frame with the current frame’s detection results and computed TP/FP rates for each
+ It appears that higher detector precision suggests that tracker-only missed detections are being flagged more often

### What to Do Today
+ So far, the 3-version approach has always terminated within a single frame; determine where it should stop. When are 2-model results no longer meaningfully different from a single model, and when is it sufficient?
+ Determine how to build the optimal configuration. For the transition from 1 -> 2 versions, it seems possible to choose a fairly mechanical rule (if there are many previously unseen instances relative to the tracker, move in the direction of suppressing false positives; if the previous ones are missing, add models in the direction of reducing missed detections).
+
+ Fix the model switching so it can be changed by dataset
+ Fine-tune on the CARLA dataset


## Current Challenges
+ Verify that diversity is an effective proxy for uncertainty (`confidence`, `entropy`, `CertainNet`, etc.)
+ Which diversity measure should be selected?
+ Methods to use the previous frame beyond the tracker
+ Modularization of modules such as `file_reader`
+ Comparison with existing methods
+ Refactoring of rough areas

## Feedback on Progress Reports

### Feedback
