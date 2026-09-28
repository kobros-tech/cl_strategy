# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""SplitMNIST Skill Memory experiment with persistent fingerprint routing.

The experiment uses:

* Skill Memory for continual training
* frozen per-class behavior fingerprints
* an independent normal-ML reverse router
* anonymous fingerprint-based skill selection
* an independent ML evaluator for final x -> y evaluation
* Avalanche's normal train/eval lifecycle
* accuracy/loss tracking
* forgetting metrics

Unlike `demo_splitmnist.py`, this demo explicitly installs
`PersistentFingerprintSkillMemoryPlugin` as the Skill Memory plugin.

The fingerprint router is trained after each training experience from
frozen skill snapshots and deterministic reference samples. During
evaluation, the router receives only x and selects the Skill Memory
skill. No evaluation label, task ID, experience ID, or skill ID is given
to the router.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.models import SimpleMLP
from torch import nn

from skill_memory import (
    PersistentFingerprintSkillMemoryPlugin,
    SkillMemory,
    SkillMemoryStrategy,
)
from skill_memory.diagnostics import (
    diagnose_evaluator_probe,
    evaluate_class_oracle,
    evaluate_skill_memory,
    timing_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate Skill Memory with persistent "
            "fingerprint routing on SplitMNIST."
        )
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=5)
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=1,
        help="Epochs used by the independent ML evaluator.",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-learning-rate", type=float, default=0.01)
    parser.add_argument("--probe-behavior-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument(
        "--reverse-epochs",
        type=int,
        default=10,
        help="Normal-ML fingerprint router training epochs.",
    )
    parser.add_argument(
        "--reverse-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the fingerprint reverse router.",
    )
    parser.add_argument(
        "--reverse-hidden-size",
        type=int,
        default=64,
        help="Hidden size of the fingerprint reverse router.",
    )
    parser.add_argument(
        "--reverse-batch-size",
        type=int,
        default=256,
        help="Batch size for fingerprint reverse-router training.",
    )
    parser.add_argument(
        "--reverse-training-mode",
        choices=("listwise", "pairwise"),
        default="listwise",
        help="Training objective used by the normal-ML reverse router.",
    )
    parser.add_argument(
        "--reverse-warm-start",
        action="store_true",
        help=(
            "Warm-start the fingerprint router from the previous "
            "experience instead of fitting each router from scratch."
        ),
    )
    parser.add_argument(
        "--reverse-warm-start-epochs",
        type=int,
        default=None,
        help=(
            "Training epochs for warm-started router fits. "
            "If omitted, reverse-epochs is used."
        ),
    )

    parser.add_argument(
        "--eval-routing",
        choices=("none", "probe"),
        default="probe",
        help=(
            "Evaluation routing used by the fingerprint Skill Memory "
            "plugin. The independent ML evaluator always uses its "
            "normal anonymous x -> y path."
        ),
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Enable routing diagnostics, alignment reports, and timing measurements."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benchmark = SplitMNIST(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

    print("=== SplitMNIST Skill Memory fingerprint experiment ===")
    print("Training method: Skill Memory")
    print("Routing method: persistent normal-ML fingerprint router")
    print("Evaluation method: independent anonymous ML evaluator")
    print(f"Evaluation routing: {args.eval_routing}")
    print(f"Fingerprint warm start: {args.reverse_warm_start}")
    print(f"Diagnostics: {'enabled' if args.diagnose else 'disabled'}")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    # print(
    #     "Evaluation memory per class:",
    #     args.eval_memory_per_class,
    # )
    print("Fingerprint reverse epochs:", args.reverse_epochs)

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    # ------------------------------------------------------------------
    # Main Skill Memory model.
    # ------------------------------------------------------------------

    model = SimpleMLP(num_classes=10).to(device)

    # ------------------------------------------------------------------
    # Persistent fingerprint Skill Memory plugin.
    #
    # This is the important difference from demo_splitmnist.py.
    #
    # The normal SkillMemoryStrategy uses EvaluationMemoryPlugin by
    # default. Here we explicitly install the persistent fingerprint
    # implementation instead.
    # ------------------------------------------------------------------

    fingerprint_memory = SkillMemory(max_skills=args.max_skills)
    fingerprint_plugin = PersistentFingerprintSkillMemoryPlugin(
        memory=fingerprint_memory,
        max_skills=args.max_skills,
        forgetting_margin=0.05,
        score_floor=0.9,
        probe_batch_size=args.eval_batch_size,
        probe_batches=5,
        probe_seed=args.seed,
        class_train_epochs=args.train_epochs,
        class_train_batch_size=args.batch_size,
        reuse_is_mutable=False,
        # eval_memory_per_class=args.eval_memory_per_class,
        # eval_memory_seed=args.seed,
        verbose=True,
        eval_routing=args.eval_routing,
        diagnose=args.diagnose,
        reverse_hidden_size=args.reverse_hidden_size,
        reverse_epochs=args.reverse_epochs,
        reverse_learning_rate=args.reverse_learning_rate,
        reverse_seed=args.seed,
        reverse_batch_size=args.reverse_batch_size,
        reverse_training_mode=args.reverse_training_mode,
        reverse_warm_start=args.reverse_warm_start,
        reverse_warm_start_epochs=args.reverse_warm_start_epochs,
        record_candidate_diagnostics=args.diagnose,
    )

    # ------------------------------------------------------------------
    # SkillMemoryStrategy owns the public Avalanche lifecycle.
    #
    # The fingerprint plugin is the actual Skill Memory implementation.
    # The independent ML evaluator remains separate from the fingerprint
    # router.
    # ------------------------------------------------------------------

    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
        ),
        criterion=nn.CrossEntropyLoss(),
        max_skills=args.max_skills,
        train_mb_size=args.batch_size,
        train_epochs=args.train_epochs,
        eval_mb_size=args.eval_batch_size,
        evaluator_model_factory=lambda: SimpleMLP(
            num_classes=10,
            input_size=28 * 28,
            hidden_size=2048,
            hidden_layers=1,
            drop_rate=0.1,
        ),
        # eval_memory_per_class=args.eval_memory_per_class,
        eval_epochs=args.eval_epochs,
        eval_learning_rate=args.eval_learning_rate,
        probe_seed=args.seed,
        device=device,
        diagnose=args.diagnose,
        verbose=True,
        eval_routing=args.eval_routing,
        probe_behavior_weight=args.probe_behavior_weight,
        skill_memory_plugin=fingerprint_plugin,
    )

    # ------------------------------------------------------------------
    # Complete Avalanche lifecycle.
    # ------------------------------------------------------------------

    for experience_index, experience in enumerate(benchmark.train_stream):
        print()
        print(f"========== Training experience {experience_index} ==========")
        print(
            "Classes:",
            sorted(int(class_id) for class_id in experience.classes_in_this_experience),
        )

        strategy.train(experience)

        # The fingerprint plugin has now:
        #
        #   1. updated Skill Memory
        #   2. frozen the relevant skill generations
        #   3. updated behavior fingerprints
        #   4. fitted the normal-ML reverse router
        #
        # The router therefore exists before this evaluation pass.

        print(f"========== Evaluation after experience {experience_index} ==========")

        strategy.eval(benchmark.test_stream)

        # --------------------------------------------------------------
        # Fingerprint routing diagnostics.
        # --------------------------------------------------------------

        if args.diagnose:
            print("---------- Fingerprint diagnostics ----------")

            print(
                "fingerprint_router_model=",
                type(fingerprint_plugin.reverse_engineer.model).__name__,
            )

            print(
                "fingerprint_records=",
                len(fingerprint_plugin.behavior.state_dict()["records"]),
            )

            print(
                "fingerprint_output_dim=",
                fingerprint_plugin._reverse_output_dim,
            )

            print(
                "fingerprint_candidate_dim=",
                fingerprint_plugin._reverse_candidate_dim,
            )

            if fingerprint_plugin.last_alignment_report:
                print("fingerprint_class_alignment:")
                for (
                    skill_id,
                    report,
                ) in fingerprint_plugin.last_alignment_report.items():
                    print(f"  skill {skill_id}: {report}")

            routing_diagnostics = fingerprint_plugin.last_routing_diagnostics

            if routing_diagnostics:
                print("fingerprint_routing_diagnostics:")
                for key, value in routing_diagnostics.items():
                    print(f"  {key}: {value}")

            # Keep the original diagnostics from the reference demo too.
            class_oracle = evaluate_class_oracle(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=10,
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=args.diagnose,
            )

            direct_probe = evaluate_skill_memory(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=10,
                routing="probe",
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=args.diagnose,
            )

            evaluator_probe = diagnose_evaluator_probe(
                strategy,
                benchmark.test_stream,
                batch_size=args.eval_batch_size,
                diagnose=args.diagnose,
            )

            print(
                "class_oracle_mean_accuracy=",
                f"{np.mean([item['accuracy'] for item in class_oracle.values()]):.4f}",
            )
            print(
                "direct_probe_mean_accuracy=",
                f"{np.mean([item['accuracy'] for item in direct_probe.values()]):.4f}",
            )
            print(
                "evaluator_probe_routing_accuracy=",
                f"{evaluator_probe['probe_routing_accuracy']:.4f}",
            )
            print(
                "evaluator_probe_mean_confidence=",
                f"{evaluator_probe['probe_mean_confidence']:.4f}",
            )
            print(
                "evaluator_probe_mean_margin=",
                f"{evaluator_probe['probe_mean_margin']:.4f}",
            )

            for bucket, stats in timing_report(strategy).items():
                print(
                    f"timing[{bucket}]: "
                    f"total={stats['total_seconds']:.2f}s "
                    f"calls={stats['calls']} "
                    f"mean={stats['mean_seconds']:.3f}s"
                )

        # --------------------------------------------------------------
        # Show actual fingerprint routes.
        # --------------------------------------------------------------

        routes = fingerprint_plugin.fingerprint_route_history

        if routes:
            print("---------- Fingerprint routes ----------")

            for route in routes[:20]:
                print(
                    "  "
                    f"batch={route.get('batch_index')} "
                    f"sample={route.get('sample_index')} "
                    f"y={route.get('evaluation_y')} "
                    f"class={route.get('class')} "
                    f"skill={route.get('skill')} "
                    f"prob={route.get('probability', 0.0):.4f} "
                    f"pred={route.get('model_predicted_class')} "
                    f"correct={route.get('model_correct')}"
                )

            if len(routes) > 20:
                print(f"  ... {len(routes) - 20} additional routes omitted")

    # ------------------------------------------------------------------
    # Final results.
    # ------------------------------------------------------------------

    results = strategy.results()

    print()
    print("=== Summary ===")

    print(
        "mean_final_accuracy=",
        f"{results['mean_final_accuracy']:.4f}",
    )
    print(
        "mean_final_loss=",
        f"{results['mean_final_loss']:.4f}",
    )

    print("final_class_accuracy:")
    for class_id, accuracy in results["final_class_accuracy"].items():
        print(f"  class {class_id}: {accuracy:.4f}")

    print("final_class_loss:")
    for class_id, loss in results["final_class_loss"].items():
        print(f"  class {class_id}: {loss:.4f}")


if __name__ == "__main__":
    main()
