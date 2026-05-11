# Strawberry VLA System — Phase 4 Plan

**Robotic Harvesting Integration: SO-101 + Jetson + ROS2 + GR00T N1.7**

**Date:** May 2026
**Author:** Strawberry VLA Team
**Status:** Plan (hardware arrival scheduled week of 2026-05-11)
**Target Hardware:** SO-101 leader-follower pair + NVIDIA Jetson AGX Orin 32GB (or Jetson Thor) + 2× USB webcams

---

## 1. Executive Summary

Phase 4 moves the Strawberry VLA system from a desktop demo to a physical harvesting prototype. The perception stack from Phases 1–3 is retained as the *reasoning layer*; a new *action layer* is introduced using NVIDIA GR00T N1.7 — a robot foundation model with a Qwen3-VL-derived backbone — fine-tuned on teleoperation demonstrations of strawberry picking with the SO-101 arm.

| Component | Choice | Rationale |
|---|---|---|
| Arm | SO-101 leader-follower pair | Low cost, open source, mature LeRobot ecosystem |
| Compute | Jetson AGX Orin 32GB (or Thor) | Runs YOLO + VLM + ROS2 + GR00T concurrently |
| Middleware | ROS2 Humble + MoveIt2 | Camera, motion planning, motor I/O |
| Perception (detection) | YOLO11s, exported to TensorRT FP16 | Carried forward from Phase 2 |
| Perception (reasoning) | Qwen 3 VL 8B + LoRA via vLLM | Carried forward — ripeness, disease, harvest decisions |
| Action policy | NVIDIA GR00T N1.7 fine-tuned | Built-in action expert; Seeed publishes SO-101 recipe |
| Cameras | 2× USB webcam (wrist + scene) | BOM cost; depth via Phase 3 monocular method + visual servo |
| Annotation | Gemini 3 Flash + Robotics-ER 1.6 | Auto-label boxes, grasp points, success/failure |
| Calibration | OpenCV intrinsic + easy_handeye2 (ChArUco) | Standard ROS2 hand-eye procedure |

**Success criterion for Phase 4:** the SO-101 follower arm autonomously detects, approaches, and picks a single ripe strawberry from a controlled benchtop scene, given only camera input and the language instruction "pick the ripe strawberry."

---

## 2. Robotics Concepts (Glossary)

Terms used throughout this document. The team's background is software/ML; this glossary makes the rest of the document self-contained.

| Term | Plain-English meaning |
|---|---|
| **Leader arm** | The smaller, back-drivable arm an operator moves by hand. Acts as a puppet controller. Has a trigger grip in place of a real gripper. |
| **Follower arm** | The arm that does real work — has the actual gripper. Mirrors the leader's joint angles in real time during teleop. |
| **Teleoperation (teleop)** | Driving the follower remotely by moving the leader. The way training data is generated. |
| **Episode** | One full attempt at a task — e.g. one pick attempt, from approach to retract. A dataset contains hundreds of episodes. |
| **Teleop dataset** | A recording of (camera frames, joint angles, action targets, language instruction) tuples across many episodes, stored in LeRobot's standard format. |
| **End-effector** | The tool at the end of the arm (a gripper, scissor, suction cup, etc.). Robotics jargon for "the bit that touches the world." |
| **6-DOF** | Six degrees of freedom — the SO-101 has six independently controllable motors (one of which is the gripper). |
| **URDF** | XML file describing the robot's geometry (link lengths, joint positions, mass). Like a JSON schema for the robot — ROS2, simulators, and motion planners all read it. |
| **Joint state / joint targets** | The current angles of the six motors / the target angles the system wants to move to. The basic units of control. |
| **Forward kinematics (FK)** | Given joint angles, compute where the gripper is in 3D space. |
| **Inverse kinematics (IK)** | Given a desired gripper pose in 3D space, compute joint angles to get there. |
| **Hand-eye calibration** | Determining where the camera is, in millimeters, relative to the arm's base. Without this, a pixel detection can't be translated into a gripper target. |
| **Eye-in-hand** | Camera mounted on the gripper/wrist — moves with the arm. |
| **Eye-to-hand** | Camera mounted in the scene, fixed — looking at the workspace. |
| **ROS2** | Robot Operating System 2 — not an OS, but a message bus + library of robotics packages. Camera, detector, planner, and motor controller all publish/subscribe to topics like `/camera/image`. |
| **MoveIt2** | Motion-planning library on top of ROS2. Computes collision-free joint trajectories to reach a target pose. |
| **VLA** | Vision-Language-Action — a neural network that takes images + a language instruction and outputs motor commands. |
| **Action head / action expert** | The part of a VLA model that actually emits motor commands (as opposed to text or pixels). |
| **Action chunk** | A sequence of N future motor commands emitted at once — used for smoother control than predicting one step at a time. |
| **Foundation model (for robots)** | A large neural network pre-trained on millions of robot demonstrations, designed to be fine-tuned on small task-specific datasets. GR00T is one. |
| **GR00T N1.7** | NVIDIA's open robot foundation model. Backbone derived from Qwen3-VL. Includes a built-in action expert. |
| **ChArUco board** | A checkerboard with embedded ArUco markers — used for camera calibration. More robust than plain checkerboards under partial occlusion. |
| **Visual servoing** | Closing the loop on a target using live camera feedback during the final approach, rather than executing a pre-planned trajectory blindly. |

---

## 3. System Architecture

### 3.1 High-Level Data Flow

```
   ┌──────────────────────────────────────────────────────────────────┐
   │                  Jetson AGX Orin 32GB                            │
   │                                                                  │
   │  ┌──────────┐     ┌──────────────┐                               │
   │  │ wrist cam│────▶│ camera node  │──┐                            │
   │  └──────────┘     │  (gscam2)    │  │                            │
   │  ┌──────────┐     └──────────────┘  │                            │
   │  │ scene cam│──────────────────────▶│                            │
   │  └──────────┘                       ▼                            │
   │                              ┌──────────────┐                    │
   │                              │  YOLO11s-TRT │── Detection2DArray │
   │                              │  (yolo_ros)  │                    │
   │                              └──────────────┘                    │
   │                                     │                            │
   │                                     ▼                            │
   │                              ┌──────────────┐                    │
   │                              │  Qwen 3 VL   │── ripeness, disease│
   │                              │  via vLLM    │── pick / skip      │
   │                              └──────────────┘                    │
   │                                     │                            │
   │                                     ▼                            │
   │                              ┌──────────────┐                    │
   │                              │  GR00T N1.7  │── joint targets    │
   │                              │  (fine-tuned)│                    │
   │                              └──────────────┘                    │
   │                                     │                            │
   │                                     ▼                            │
   │                              ┌──────────────┐                    │
   │                              │  MoveIt2 +   │── motor commands   │
   │                              │  ros2_control│                    │
   │                              └──────────────┘                    │
   │                                     │                            │
   └─────────────────────────────────────┼────────────────────────────┘
                                         ▼
                              ┌──────────────────┐
                              │ SO-101 follower  │
                              │ (6× Feetech servos)
                              └──────────────────┘
```

### 3.2 Two Distinct Modes

The system runs in two modes that use very different parts of the stack:

**Mode A — Teleop / data collection (training time).**
Human operates the leader arm. Follower mirrors it. The Jetson records camera frames, joint states, and action targets at 30 Hz into a LeRobot dataset. GR00T is not in the loop. This mode produces the data that GR00T is fine-tuned on.

**Mode B — Autonomous (inference time).**
Leader arm is put away. Follower arm runs on its own. YOLO + Qwen-VL identify a target ripe strawberry; GR00T (fine-tuned on Mode A data) outputs joint trajectories; the arm executes. The leader plays no role.

### 3.3 Division of Labor Between Reasoning (Qwen-VL) and Action (GR00T)

This is a deliberate split:

- **Qwen 3 VL 8B (Phase 2 fine-tune, reused)** answers *high-level perception/reasoning* questions: "Is this berry ripe? Is it diseased? Should we pick it?" Runs on demand (event-driven, not every frame) when YOLO flags a candidate.
- **GR00T N1.7 (Phase 4 fine-tune)** answers *the action question*: "Given the camera image and the instruction 'pick the ripe berry,' what should the next joint trajectory be?" Runs at control rate during a pick.

The two models do not call each other directly. The orchestrator (a ROS2 node) reads Qwen-VL's pick/skip decision and gates whether GR00T is invoked.

---

## 4. Implementation Plan

### 4.1 Pre-Arrival (current week, before 2026-05-11)

Goal: every minute of post-arrival time is spent on the physical arm, not on environment setup. Maximize what can be prepared in advance.

| Task | Owner | Output |
|---|---|---|
| Place equipment order: 2× USB webcam, USB powered hub, ChArUco board (printed on rigid backing), 3D-printed scissor end-effector files queued for printing | — | Parts arriving alongside arm |
| Decide AGX Orin 32GB vs Jetson Thor | — | Final compute choice |
| Re-export the Qwen 3 VL LoRA from the original PEFT checkpoint (not from MLX — MLX format is Mac-only and not Jetson-portable) | — | `adapter_model.safetensors` ready for vLLM |
| Set up `dusty-nv/jetson-containers` build env on x86 dev box; script YOLO11s `.pt` → TensorRT `.engine` export (FP16) | — | Reproducible export script + validated `.engine` |
| Pull Seeed's GR00T-SO101 fine-tuning recipe; read end-to-end; clone and stage | — | Local repo + understanding of inputs |
| Pull MuammerBay's SO-ARM101 MoveIt2 + Isaac Sim repo; load URDF in Isaac Sim; teleop a simulated arm with keyboard input | — | Working sim — full motion-planning stack verified before hardware |
| Pilot Gemini Robotics-ER 1.6 annotation on ~100 existing Phase-2 strawberry frames; lock down JSON schema for `box_2d`, `ripeness`, `grasp_point`, `pedicel_point`, `pick_action` | — | Validated annotation prompt + cost estimate |

### 4.2 Week 1 After Arrival (2026-05-11 to 2026-05-17)

Day-by-day, sequential. Each day blocks on the previous.

**Day 1 — Hardware bring-up.**
Assemble both leader and follower arms (LeRobot docs walk through this; expect 4–6 hours total). Flash Jetson with JetPack 6.2. Verify USB connectivity to both arms; verify motor IDs and zero positions per LeRobot's calibration script.

**Day 2 — Camera mounting and software install.**
3D-print or screw-mount a wrist camera bracket onto the follower's last link. Position scene camera on a tripod or fixed mount looking at a 40 cm × 40 cm workspace. Install ROS2 Humble + so101-ros2 workspace + yolo_ros + easy_handeye2 + vLLM container.

**Day 3 — Camera intrinsic calibration.**
Use OpenCV's `calibrateCamera` with 20–30 ChArUco views per camera. Save to YAML. This is identical to the Phase 3 procedure.

**Day 4 — Hand-eye (extrinsic) calibration.**
Run `easy_handeye2` for both cameras. Eye-in-hand for the wrist camera (board fixed in workspace, arm moves through 15–30 poses ≥ 30° rotation diversity per axis). Eye-to-hand for the scene camera (board attached to gripper, arm moves through poses). Compare `DANIILIDIS` vs `PARK` solvers; pick whichever has lower hold-out error.

**Day 5 — End-to-end perception loop.**
Run YOLO TensorRT on live wrist-cam feed, publishing `Detection2DArray`. Map bbox center + monocular depth → 3D point in `base_link` frame using the calibration from Day 4. Verify "this pixel = this 3D point" agrees with a measured ruler ground truth within ~1 cm.

**Day 6 — First teleop session.**
Use a paper strawberry or a real one. Operator drives the leader; system records ~30 teleop episodes of "pick the berry from a fixed position." Validate the LeRobot dataset format is being written correctly.

**Day 7 — Push dataset to HF Hub; baseline.**
Upload the 30 episodes to a private HuggingFace dataset. Run an initial GR00T N1.7 fine-tune on this tiny set (will not produce a working policy, but validates the full pipeline end-to-end). Deploy resulting checkpoint to Jetson, attempt one autonomous pick — even a bad one — to confirm Mode B closes the loop.

### 4.3 Weeks 2–3: Data Collection and First Working Policy

**Goal:** 200–300 successful teleop episodes covering pose variation, lighting variation, and 2–3 distractor berries per frame. Fine-tune GR00T on the full set. Target: ≥50% success rate on novel berry positions in the same lighting.

Activities:
- Daily teleop sessions (1–2 hours, ~30–50 episodes per session).
- Gemini Robotics-ER 1.6 annotates each episode with success/failure verification, grasp-point quality, and language instruction variants — feeds back into training as auxiliary signals.
- After each fine-tune run, evaluate on a held-out scene; track success rate over time.
- Iterate gripper design if grasp failures cluster around stem damage.

### 4.4 Weeks 4+: Real Strawberry Plants, Disease Routing, Generalization

- Move from benchtop to a small potted strawberry plant; collect another 100–200 episodes in this harder visual domain.
- Wire the Phase-2 Qwen 3 VL ripeness/disease classifier in as the picker's "should-pick" gate: skip unripe, skip diseased.
- Stretch goal: harvest multiple berries in sequence from one plant without operator intervention.

---

## 5. Bill of Materials (Indicative)

Confirm specific SKUs before ordering — these are categories, not vendor lock-ins.

| Item | Notes | Approx cost |
|---|---|---|
| SO-101 leader-follower kit (assembled) | Already ordered | ~$500 |
| Jetson AGX Orin 32GB Dev Kit | Or Jetson Thor (~$3,500) if budget allows | ~$2,000 |
| USB webcam — wrist | Small UVC module (e.g. ELP USB module) | ~$30 |
| USB webcam — scene | Standard 1080p UVC (different model from wrist — identical models cause USB conflicts) | ~$60 |
| Powered USB 3 hub | 4+ port, separate power adapter | ~$30 |
| ChArUco board, A3, rigid backing | Print 10x7 with 30mm squares, mount on foam-core or aluminum plate | ~$20 |
| 3D-printed strawberry end-effector | Scissor/blade design; open files on Printables | filament cost only |
| Workspace mat + lighting | Diffused white LED panel, matte mat for vision background | ~$80 |

---

## 6. Risks and Open Questions

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| 8B Qwen-VL + GR00T + YOLO won't fit AGX Orin 32GB memory | Medium | High | Plan-B is event-driven Qwen-VL (only invoked on YOLO trigger), or upgrade to Thor |
| Monocular depth error >10% causes consistent grasp miss | Medium | Medium | Wrist camera close-range visual servoing in the final 5 cm absorbs depth error |
| SO-101 default gripper damages berries even in teleop | High | Medium | Plan-B scissor end-effector is the actual solution; budget print time in week 1 |
| First GR00T fine-tune produces near-random policy | High initially | Low | Expected — first run is pipeline validation. Real policy expected after week 2–3 data volume |
| Hand-eye calibration drifts after physical bump | Medium | Medium | Re-run easy_handeye2 weekly; checklist includes "did anything touch the camera mount?" |
| Japanese-greenhouse domain shift from benchtop training | High | Low (deferred) | Phase 5 concern; benchtop validation is Phase 4 success criterion |

---

## 7. Reports and Cross-References

- [Phase 1 Report](report.md) — YOLO baseline + Qwen VL baseline + RGB ripeness
- [Phase 2 Report](report_phase2.md) — Fine-tuning (YOLO + Qwen LoRA), model comparison
- [Phase 3 Report](report_phase3.md) — 3D coordinates, camera calibration, validation
- **Phase 4 Plan (this document)** — Robot integration
- Phase 4 Report (TBD) — Will document actual results after week 4

---

## 8. Key External References

- [SO-101 official docs (LeRobot)](https://huggingface.co/docs/lerobot/so101)
- [SO-ARM100/101 hardware repo](https://github.com/TheRobotStudio/SO-ARM100)
- [so101-ros2 workspace](https://so101-ros2.readthedocs.io)
- [MuammerBay/SO-ARM101 MoveIt + Isaac Sim](https://github.com/MuammerBay/SO-ARM101_MoveIt_IsaacSim)
- [Seeed: Fine-tune GR00T for SO-101 on Jetson Thor](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.5_for_lerobot_so_arm_and_deploy_on_jetson_thor/)
- [Seeed: Fine-tune GR00T on AGX Orin](https://wiki.seeedstudio.com/fine_tune_gr00t_n1.6_for_lerobot_so_arm_and_deploy_on_agx_orin/)
- [GR00T N1.7 announcement](https://huggingface.co/blog/nvidia/gr00t-n1-7)
- [Isaac GR00T training repo](https://github.com/NVIDIA/Isaac-GR00T)
- [easy_handeye2](https://github.com/marcoesposito1988/easy_handeye2)
- [yolo_ros (mgonzs13)](https://github.com/mgonzs13/yolo_ros)
- [Jetson AI Lab](https://www.jetson-ai-lab.com)
- [Gemini Robotics-ER 1.6](https://ai.google.dev/gemini-api/docs/robotics-overview)
