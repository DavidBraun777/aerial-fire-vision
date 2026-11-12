# Project Charter

**Wildfire Detection in Drone Imagery with Little or No Labeled Data: Comparing Supervised CNNs, Variational Autoencoders, and CLIP**

| Charter field | Value |
| --- | --- |
| Project / team | Wildfire Detection from Drone Imagery / **The Visionaries** |
| Academic context | University of St. Thomas — SEIS 766, Vision AI, Fall 2026 |
| Version / status | **0.1 — Draft for team review and approval** |
| Prepared | November 11, 2026 |
| Project sponsor / approver | **To be confirmed:** course instructor / designated project approver |
| Project manager / coordination lead | **To be assigned by the team** |
| Source of proposal | Team-provided “Wildfire Detection from Drone Imagery” project description |

## 1. Business Case and Project Purpose

Detecting visible fire in aerial imagery could help researchers develop earlier and more efficient fire-monitoring approaches. However, collecting and labeling representative wildfire images can be costly and challenging. This academic research project investigates whether image-classification and anomaly-detection methods can identify **visible fire** in existing drone imagery when few or no **task-specific fire labels** are available to a method.

The project will compare a fully supervised convolutional neural network (CNN), a variational autoencoder (VAE) trained on known no-fire examples, and CLIP-based zero-shot/few-shot recognition. It is a **comparative proof-of-concept study**, not an operational wildfire early-warning system. The team will use previously collected FLAME data; it will not fly drones or acquire original aerial footage.

## 2. Project Objectives and Research Questions

**Primary research question:** Can modern vision models detect visible fire in drone imagery with little or no task-specific labeled training data?

1. Establish the classification performance of a fully supervised ResNet/CNN as the reference baseline.
2. Determine whether a VAE trained on `No_Fire` imagery can identify fire as an anomaly using reconstruction error, and explore whether error maps roughly correspond to fire regions.
3. Evaluate CLIP with zero task-specific labeled examples and with an agreed small label budget (proposed: **10–100 labeled images**) to assess label efficiency relative to the supervised baseline.
4. Compare performance on a held-out test partition acquired under different camera and scene conditions, without treating that comparison as a controlled camera-only test.
5. Document reproducible methodology, quantitative results, failure modes, and limitations.

**Terminology:** “Zero-shot” refers to no *FLAME-specific training labels* for the CLIP evaluation; CLIP has substantial prior pretraining. The VAE needs examples identified as `No_Fire`, even though it does not use labeled fire examples for training.

## 3. Scope Statement

### In scope

- Acquire, access, inventory, and quality-check the published FLAME RGB classification and available segmentation files.
- Document data provenance, class mapping, image/mask pairing (if used), archive identities, and a leakage-aware training/validation procedure.
- Train and evaluate a supervised ResNet/CNN using the available labeled training data.
- Train and evaluate a normal-data VAE and analyze reconstruction-error anomaly scores; explore qualitative localization.
- Run zero-shot CLIP and a modest few-shot evaluation using a defined sample budget and reproducible sampling.
- Evaluate methods using a consistent, documented comparison protocol, preserving the held-out test set for final evaluation.
- Produce Colab notebooks, shared metrics, representative success/failure examples, a presentation, and a final research report.

### Out of scope unless separately approved

- New drone flights, primary data collection, or real-time wildfire monitoring/deployment.
- Claims of performance on uncontrolled wildfires, new geographic regions, or a production-ready alert system.
- Thermal-video modeling, temporal tracking, drone integration, and operational latency guarantees.
- Training a dedicated pixel-segmentation model as a required deliverable.
- **Stretch goal only:** generating synthetic fire images with a GAN for augmentation; this requires an explicit team decision, controls for label quality, and time remaining after core work.

**Scope control:** The team's proposal lists CNN, VAE, and CLIP as planned methods. The current GitHub roadmap identifies CNN + CLIP as the minimum viable deliverable and VAE as conditional. The team must **resolve and approve VAE's required/conditional status at the next scope review**; changes thereafter should be recorded in the change log. Until then, this charter treats VAE as a proposed in-scope workstream, not a completed commitment.

## 4. Dataset, Assumptions, and Boundaries

**Source:** FLAME, an open-access IEEE DataPort dataset collected by Northern Arizona University and the Air Force Research Laboratory during a prescribed pile burn in an Arizona pine forest. The project uses existing RGB aerial imagery.

| Subset | Records | Resolution | Fire | No_Fire |
| --- | ---: | --- | ---: | ---: |
| Classification: train/validation | 39,375 images | 254 × 254 | 25,018 | 14,357 |
| Classification: test | 8,617 images | 254 × 254 | 5,137 | 3,480 |
| Segmentation | 2,003 images + 2,003 masks | 3840 × 2160 reported for masks; verify image dimensions | — | — |

Image attributes include RGB pixel values (0–255), binary class labels (`Fire` and `No_Fire`), and—where available—binary masks (`1 = fire`, `0 = background`). The train/validation and test partitions have different camera/acquisition conditions. This is a useful **cross-acquisition evaluation**, but camera differences may be confounded with scene, lighting, viewpoint, and time.

The classes are moderately imbalanced (approximately **63.5% Fire** in training/validation and **59.6% Fire** in test). As frames were extracted from video, adjacent examples may be highly correlated; random frame-level splitting may cause leakage. Small fire-pixel regions can make segmentation metrics difficult to interpret. A known discrepancy between a published segmentation dimension and observed mask dimensions requires verification before localization metrics are reported.

## 5. Technical Approach and Quality Criteria

| Workstream | Approach | Key evidence |
| --- | --- | --- |
| Data quality and split protocol | Inventory archives, check labels, inspect samples, document source grouping and train/validation logic | Manifest, validation protocol, data-quality findings |
| Supervised CNN | Fine-tune ResNet on labeled training examples | Reproducible run, per-class metrics, saved predictions |
| VAE anomaly detection | Fit VAE using selected `No_Fire` images; score reconstruction error | Anomaly scores, selected threshold, analysis of false alarms/misses and optional error maps |
| CLIP | Predefined text prompts for zero-shot; separate few-shot experiment | Prompt set, label budget, reproducible sample IDs, per-class metrics |
| Comparative evaluation | Consistent held-out evaluation after model/threshold selection on development data | Comparison table, confusion matrices, limitations, failure examples |

**Required classification reporting:** accuracy, precision, recall, F1 (especially for the `Fire` class), confusion matrix, and ROC-AUC where valid. Include balanced accuracy and/or average precision if feasible, and explicitly discuss false negatives (missed fires). **IoU is only applicable to spatial localization or segmentation outputs** evaluated against verified matched masks; image-level classifiers do not receive a meaningful pixel-level IoU by default.

For fair comparison, record label budgets, seeds, preprocessing, prompts, model versions, validation selection procedures, class mapping, and relevant compute conditions. No changes to prompt sets, thresholds, or model selection may be based on final held-out test labels.

## 6. Deliverables and Acceptance Criteria

| Deliverable | Acceptance criteria |
| --- | --- |
| Approved research protocol | Team signs off on label mapping, train/validation approach, test-set isolation, metrics, and core model commitments |
| Dataset audit / manifest | Four dataset ZIP inventories documented; provenance, class counts, sample inspection, and open discrepancies recorded |
| CNN reference model | Re-runnable training/evaluation notebook or script; configuration, predictions, per-class metrics, confusion matrix |
| CLIP comparison | Re-runnable zero-shot evaluation and agreed small-label experiment if feasible; prompts and label budgets recorded |
| VAE experiment | Re-runnable normal-data training/inference and documented anomaly-scoring evaluation **if confirmed mandatory at scope approval** |
| Comparative analysis | Same held-out examples and label conventions, clear supervision-budget differences, missed-fire and failure analysis |
| Presentation and final report | Research question, methods, reproducibility details, results, limitations, team contributions, and references |

**Minimum viable research outcome:** a reproducible CNN and CLIP comparison on a defensible evaluation protocol, plus a presentation and final report. This minimum does not silently remove the VAE from the proposal; any decision to make it conditional must be explicitly approved and documented.

## 7. Milestones and Schedule Baseline

*The following dates reflect the repository's November roadmap, not verified course requirements. Confirm official presentation and submission dates in Canvas before treating them as baselined commitments.*

| Target date | Milestone | Exit criterion |
| --- | --- | --- |
| Nov 11, 2026 | Team access checkpoint | Each member confirms GitHub/Drive/Colab access or reports blocker |
| Nov 16, 2026 | Scope and experimental-design review | Scope decision on VAE, owners, data protocol, and metrics documented |
| Nov 22, 2026 | Initial baseline | Reproducible split/manifest and first CNN experiment |
| Dec 6, 2026 | Evaluation freeze target | Core comparison and approved model/threshold selections completed |
| Dec 8, 2026 (tentative) | Group presentation | Results and limitations ready for presentation |
| Dec 15, 2026 (internal target) | Deliverable freeze | Reviewed notebooks, figures, and report draft |
| Dec 19, 2026 (tentative) | Final submission | Group-approved submission package |

## 8. People, Roles, and Governance

| Team member | Email | Workstream responsibility |
| --- | --- | --- |
| Sinchana Bettadapura | bett7689@stthomas.edu | To be agreed |
| David Braun | brau2455@stthomas.edu | To be agreed |
| Thomas Hayes | tlhayes@stthomas.edu | To be agreed |
| Brahmee Adhikari | adhi4471@stthomas.edu | To be agreed |

**Proposed governance:** designate one coordinator/project manager; one accountable owner and one reviewer for dataset QA, CNN, VAE, CLIP/evaluation, and final deliverables. Assign specific people during the next team meeting rather than presuming their roles. The course instructor or designated approver is the academic sponsor/acceptance authority once confirmed.

**Communications and controls:** GitHub is the source of truth for notebooks, code, and light documentation; Google Drive holds raw archives and larger outputs; Colab is the execution environment. Use pull requests for material changes; email for brief status and blockers; Teams for meetings. Record decisions, action owners, dates, and scope changes in the proposed `project-management/` folder. Keep raw datasets, weights, credentials, and personal Drive listings out of GitHub.

## 9. Principal Risks and Responses

| Risk / dependency | Impact | Planned response | Owner |
| --- | --- | --- | --- |
| Near-duplicate frame leakage | Inflated validation performance | Agree leakage-aware validation grouping before training | TBD |
| Test acquisition differs from training | Ambiguous generalization claims | Describe cross-acquisition limitations and report error patterns | TBD |
| Scope exceeds available time | Missing core deliverables | Gate VAE/augmentation expansions; protect CNN + CLIP minimum | TBD |
| Colab/GPU or Drive access problems | Blocked experiments | Access checkpoint; document dependencies and fallback compute | TBD |
| Mask dimensions/pairing inconsistencies | Invalid localization results | Verify matches and dimensions before IoU calculation | TBD |
| Class imbalance and missed-fire cost | Misleading accuracy | Use per-class metrics, recall, F1, confusion matrices | TBD |
| Unconfirmed course deadlines | Schedule/rework risk | Verify dates in Canvas and update baseline | TBD |

**Key assumptions:** team members can obtain access to published dataset archives; available Colab resources suffice for modest experiments; the project stays within the Fall 2026 course period. **Constraints:** limited calendar time, shared compute/storage, dataset representativeness, and the availability of trustworthy independent validation groups. There is no approved expenditure budget in the source proposal; any planned paid tools or cloud charges require team approval.

## 10. Project Authorization and Approval

This draft authorizes **planning and team review only** until the group agrees on scope, ownership, acceptance criteria, and course dates. It does not claim instructor approval or assignment of individual workstreams.

| Approval item | Decision / name / date |
| --- | --- |
| Confirm sponsor / academic approver | Pending |
| Approve core methodology and VAE status | Pending team decision |
| Assign coordination lead and workstream owners | Pending |
| Confirm presentation and submission dates | Pending Canvas verification |
| Team scope baseline approved | Pending |

## References

- Team-provided proposal: *Wildfire Detection from Drone Imagery*, The Visionaries.
- FLAME dataset: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
- Project repository: https://github.com/DavidBraun777/aerial-fire-vision
- Current project README and roadmap: https://github.com/DavidBraun777/aerial-fire-vision/blob/master/README.md
