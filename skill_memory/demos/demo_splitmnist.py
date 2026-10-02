# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""SplitMNIST Skill Memory experiment with ML evaluation.

The public SkillMemoryStrategy owns the complete experiment lifecycle:

* Skill Memory training
* frozen per-class evaluation memory
* independent ML evaluator training
* anonymous x -> y evaluation through Avalanche's normal eval() lifecycle
* accuracy/loss tracking
* forgetting metrics

The demo only configures SplitMNIST and the strategy, then reports results.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR100
from avalanche.models import SimpleMLP, SlimResNet18
from torch import nn

from skill_memory import SkillMemoryStrategy
from skill_memory.demos.candidate_skill_routing_patch import (
    install_candidate_skill_routing,
)
from skill_memory.diagnostics import (
    diagnose_evaluator_probe,
    evaluate_class_oracle,
    evaluate_skill_memory,
    timing_report,
)
from skill_memory.evaluation.independent_evaluator import EvaluationMemory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate Skill Memory on SplitMNIST."
    )
    parser.add_argument("--dataset-root", default="data")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--n-experiences", type=int, default=20)
    parser.add_argument(
        "--experience-index",
        type=int,
        default=0,
        help="CIFAR-100 experience to run (0-19). Each experience contains 5 classes.",
    )
    parser.add_argument(
        "--eval-memory-per-class",
        type=int,
        default=20,
        help="Number of retained evaluation examples per class.",
    )
    parser.add_argument(
        "--skill-train-samples-per-class",
        type=int,
        default=20,
        help="Number of training samples per class used by Skill Memory.",
    )
    parser.add_argument(
        "--eval-epochs",
        type=int,
        default=1,
        help="Number of epochs used by the independent ML evaluator.",
    )
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument(
        "--class-train-mode",
        choices=("multiclass", "binary_one_vs_rest"),
        default="multiclass",
        help=(
            "Skill class-training objective: multiclass positive-only or "
            "binary one-vs-rest YES/NO."
        ),
    )
    parser.add_argument(
        "--binary-negative-source",
        choices=("seen_classes", "all_train_classes"),
        default="all_train_classes",
        help=(
            "Binary verifier negatives: seen/current classes for valid CL, or "
            "all training-stream classes for the explicit offline experiment."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-learning-rate", type=float, default=0.01)
    parser.add_argument("--probe-behavior-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-skills", type=int, default=20)
    parser.add_argument(
        "--force-decision",
        choices=("none", "reuse", "scratch"),
        default="none",
        help=(
            "Force Skill Memory decisions for new classes. Use 'reuse' in the "
            "candidate-routing experiment to create multi-class skills; "
            "'none' keeps the normal decision policy."
        ),
    )
    parser.add_argument(
        "--eval-method",
        choices=("ml", "cl"),
        default="ml",
        help=(
            "Normal evaluator: standalone ML evaluator or Skill Memory's "
            "own CL-only YES/NO evaluator."
        ),
    )
    parser.add_argument(
        "--eval-routing",
        choices=("none", "probe"),
        default="none",
        help=(
            "ML evaluation routing: independent evaluator only or anonymous "
            "Skill Memory probe."
        ),
    )
    parser.add_argument(
        "--candidate-skill-routing",
        action="store_true",
        help=(
            "Experimental: use evaluator top-k classes as candidates and "
            "let their canonical Skill Memory skills verify them."
        ),
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=3,
        help="Number of evaluator top classes offered to Skill Memory.",
    )
    parser.add_argument(
        "--skill-confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum canonical-skill confidence required for verification.",
    )
    parser.add_argument(
        "--rescue-skill-confidence-threshold",
        type=float,
        default=0.85,
        help="Minimum Skill Memory verification score for a CL override.",
    )
    parser.add_argument(
        "--rescue-skill-margin",
        type=float,
        default=0.10,
        help="Minimum Skill Memory score advantage over the ML top-1 skill.",
    )
    parser.add_argument(
        "--ml-uncertainty-threshold",
        type=float,
        default=0.40,
        help=(
            "ML confidence below which CL can override; "
            "unverified ML top-1s are also eligible."
        ),
    )
    parser.add_argument(
        "--candidate-routing-debug",
        action="store_true",
        help="Print ML candidates, Skill Memory verification, and final election.",
    )
    parser.add_argument(
        "--candidate-routing-debug-samples",
        type=int,
        default=20,
        help="Maximum number of evaluation samples to print in routing debug logs.",
    )
    parser.add_argument(
        "--skill-validation-fraction",
        type=float,
        default=0.2,
        help=(
            "Fraction held out from Skill Memory training for verification calibration."
        ),
    )
    parser.add_argument(
        "--skill-validation-seed",
        type=int,
        default=0,
        help="Seed for the disjoint Skill Memory verification holdout.",
    )
    parser.add_argument(
        "--routing-validation-precision",
        type=float,
        default=0.90,
        help="Minimum held-out precision required for candidate verification.",
    )
    parser.add_argument(
        "--routing-validation-min-samples",
        type=int,
        default=20,
        help="Minimum held-out samples required for threshold calibration.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Run optional Skill Memory diagnostics (class oracle and "
            "anonymous probe). Diagnostics never affect production evaluation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.eval_method == "cl" and args.eval_routing != "none":
        raise ValueError("--eval-routing is only available with --eval-method ml.")

    if args.eval_method == "cl" and args.candidate_skill_routing:
        raise ValueError(
            "--candidate-skill-routing requires --eval-method ml; "
            "the CL evaluator must remain independent."
        )

    if args.candidate_skill_routing and args.eval_routing != "none":
        raise ValueError(
            "--candidate-skill-routing is an alternative to --eval-routing=probe; "
            "use --eval-routing none."
        )

    if args.candidate_skill_routing:
        config = install_candidate_skill_routing(
            candidate_k=args.candidate_k,
            skill_confidence_threshold=args.skill_confidence_threshold,
            rescue_skill_confidence_threshold=args.rescue_skill_confidence_threshold,
            rescue_skill_margin=args.rescue_skill_margin,
            ml_uncertainty_threshold=args.ml_uncertainty_threshold,
            calibration_precision_target=args.routing_validation_precision,
            calibration_min_samples=args.routing_validation_min_samples,
            debug=(
                args.candidate_routing_debug or args.candidate_routing_debug_samples > 0
            ),
            debug_max_samples=args.candidate_routing_debug_samples,
        )
        print(
            "Experimental candidate-skill routing: "
            f"k={config.candidate_k}, "
            f"skill_threshold={config.skill_confidence_threshold:.3f}, "
            f"rescue_threshold={config.rescue_skill_confidence_threshold:.3f}, "
            f"rescue_margin={config.rescue_skill_margin:.3f}, "
            f"ml_uncertainty={config.ml_uncertainty_threshold:.3f}, "
            f"validation_precision={config.calibration_precision_target:.3f}, "
            f"validation_min_samples={config.calibration_min_samples}"
        )

    if not 0 <= args.experience_index < 20:
        raise ValueError("--experience-index must be between 0 and 19.")

    experience_index = args.experience_index
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # benchmark = SplitMNIST(
    #     n_experiences=args.n_experiences,
    #     seed=args.seed,
    #     dataset_root=args.dataset_root,
    # )
        
    print("Preparing CIFAR-100 dataset...")
    print(f"Dataset root: {args.dataset_root}")
    print("If CIFAR-100 is not already downloaded, downloading it now...")

    benchmark = SplitCIFAR100(
        n_experiences=args.n_experiences,
        seed=args.seed,
        dataset_root=args.dataset_root,
    )

    print("CIFAR-100 dataset is ready.")

    experience = benchmark.train_stream[experience_index]

    print(
        f"Selected experience {experience_index}: "
        f"classes={sorted(experience.classes_in_this_experience)} "
        f"samples={len(experience.dataset)}"
    )

    print("=== SplitMNIST Skill Memory experiment ===")
    print("Training method: Skill Memory")
    print(f"Skill class-training mode: {args.class_train_mode}")
    print(
        "Evaluation method: "
        + (
            "standalone Skill Memory CL evaluator"
            if args.eval_method == "cl"
            else "independent anonymous ML evaluator"
        )
    )
    if args.candidate_skill_routing:
        print("Evaluation routing: candidate-skill arbitration")
        candidate_routing_status = (
            "enabled"
            if (
                args.candidate_routing_debug or args.candidate_routing_debug_samples > 0
            )
            else "disabled"
        )
        print(
            "Candidate routing debug: "
            f"{candidate_routing_status} "
            f"(max samples={args.candidate_routing_debug_samples})"
        )
    else:
        print(f"Evaluation routing: {args.eval_routing}")
    print(f"Diagnostics: {'enabled' if args.diagnose else 'disabled'}")
    print(f"Device: {device}")
    print(f"Experiences: {len(benchmark.train_stream)}")
    print(
        "Evaluation memory samples per class:",
        args.eval_memory_per_class,
    )
    print(
        "Skill training samples per class:",
        args.skill_train_samples_per_class,
    )
    print(
        "Skill Memory decision policy:",
        args.force_decision,
    )

    for index, experience in enumerate(benchmark.train_stream):
        print(
            f"  Exp {index}: "
            f"classes={sorted(experience.classes_in_this_experience)} "
            f"samples={len(experience.dataset)}"
        )

    if args.download_only:
        print(f"SplitMNIST dataset prepared at {args.dataset_root}")
        return

    binary_negative_pool = None
    if args.class_train_mode == "binary_one_vs_rest":
        if args.binary_negative_source == "all_train_classes":
            samples_by_class = {}
            for train_experience in benchmark.train_stream:
                for index in range(len(train_experience.dataset)):
                    sample = train_experience.dataset[index]
                    class_id = int(sample[1])
                    samples_by_class.setdefault(class_id, []).append(
                        torch.as_tensor(sample[0]).detach().cpu()
                    )
            binary_negative_pool = [
                EvaluationMemory(
                    inputs=torch.stack(inputs),
                    targets=torch.full((len(inputs),), class_id, dtype=torch.long),
                    class_id=class_id,
                )
                for class_id, inputs in sorted(samples_by_class.items())
            ]
            print(
                "Binary negative source: all training-stream classes "
                "(explicit offline/full-dataset experiment)"
            )
        else:
            print("Binary negative source: seen/current classes (valid CL)")

    # model = SimpleMLP(num_classes=10).to(device)
    model = SlimResNet18(nclasses=100).to(device)

    force_decision = None if args.force_decision == "none" else args.force_decision

    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
        ),
        criterion=nn.CrossEntropyLoss(),
        max_skills=args.max_skills,
        class_train_mode=args.class_train_mode,
        skill_train_samples_per_class=args.skill_train_samples_per_class,
        validation_fraction=args.skill_validation_fraction,
        validation_seed=args.skill_validation_seed,
        force_decision=force_decision,
        train_mb_size=args.batch_size,
        train_epochs=args.train_epochs,
        eval_mb_size=args.eval_batch_size,
        evaluator_model_factory=lambda: SlimResNet18(nclasses=100),
        eval_memory_per_class=args.eval_memory_per_class,
        eval_epochs=args.eval_epochs,
        eval_learning_rate=args.eval_learning_rate,
        probe_seed=args.seed,
        device=device,
        diagnose=args.diagnose,
        verbose=True,
        eval_routing=args.eval_routing,
        binary_negative_pool=binary_negative_pool,
        probe_behavior_weight=args.probe_behavior_weight,
        eval_method=args.eval_method,
    )

    # for experience_index, experience in enumerate(benchmark.train_stream):
    for experience_index, experience in [
        (args.experience_index, benchmark.train_stream[args.experience_index])
    ]:
        print()
        print(f"========== Training experience {experience_index} ==========")
        print(
            "Classes:",
            sorted(int(class_id) for class_id in experience.classes_in_this_experience),
        )

        strategy.train(experience)

        # Evaluate only classes introduced so far. This is the same cumulative
        # test population for both evaluator choices and prevents an early
        # CL evaluator from being penalized for classes that have no skill yet.
        # eval_stream = [
        #     benchmark.test_stream[index] for index in range(experience_index + 1)
        # ]

        eval_stream = [benchmark.test_stream[experience_index]]

        class_map = strategy.skill_memory_plugin.class_map
        memory = strategy.skill_memory_plugin.memory
        print("Skill Memory groups after training:")
        for skill in sorted(memory.slots()):
            classes = sorted(class_map.classes_for_skill(skill))
            print(f"  skill {skill}: classes={classes}")

        print(f"========== Evaluation after experience {experience_index} ==========")
        strategy.eval(eval_stream)

        if args.candidate_skill_routing:
            routing_stats = strategy.ml_evaluation_plugin._candidate_skill_routing_stats
            print("---------- Candidate-skill routing ----------")
            print(
                "candidate_top1_accuracy=",
                f"{routing_stats.get('candidate_top1_accuracy', 0.0):.4f}",
            )
            print(
                f"candidate_top{args.candidate_k}_recall=",
                f"{routing_stats.get('candidate_topk_recall', 0.0):.4f}",
            )
            print(
                "skill_acceptance_rate=",
                f"{routing_stats.get('skill_acceptance_rate', 0.0):.4f}",
            )
            print(
                "fallback_rate=",
                f"{routing_stats.get('fallback_rate', 0.0):.4f}",
            )
            print(
                "top1_skill_verified_rate=",
                f"{routing_stats.get('top1_skill_verified_rate', 0.0):.4f}",
            )
            print(
                "rescue_rate=",
                f"{routing_stats.get('rescue_rate', 0.0):.4f}",
            )
            print(
                "skill_override_rate=",
                f"{routing_stats.get('skill_override_rate', 0.0):.4f}",
            )
            print(
                "ml_top1_accuracy=",
                f"{routing_stats.get('ml_top1_accuracy', 0.0):.4f}",
            )
            print(
                "routed_batch_accuracy=",
                f"{routing_stats.get('final_batch_accuracy', 0.0):.4f}",
            )
            print(
                "cl_override_rate=",
                f"{routing_stats.get('cl_override_rate', 0.0):.4f}",
            )
            print(
                "cl_corrected_ml_error_rate=",
                f"{routing_stats.get('cl_corrected_ml_error_rate', 0.0):.4f}",
            )
            print(
                "cl_introduced_error_rate=",
                f"{routing_stats.get('cl_introduced_error_rate', 0.0):.4f}",
            )
            print(
                "cl_net_accuracy_gain=",
                f"{routing_stats.get('cl_net_accuracy_gain', 0.0):+.4f}",
            )
            print(
                "cl_override_precision=",
                f"{routing_stats.get('cl_override_precision', 0.0):.4f}",
            )

        if args.diagnose:
            print("---------- Diagnostics ----------")
            class_oracle = evaluate_class_oracle(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=100,
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=args.diagnose,
            )
            direct_probe = evaluate_skill_memory(
                strategy.model,
                strategy.skill_memory_plugin,
                benchmark.test_stream,
                experience_index,
                num_classes=100,
                routing="probe",
                batch_size=args.eval_batch_size,
                device=device,
                diagnose=args.diagnose,
            )
            evaluator_probe = None
            if args.eval_method == "ml":
                evaluator_probe = diagnose_evaluator_probe(
                    strategy,
                    eval_stream,
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
            if evaluator_probe is not None:
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
                    f"timing[{bucket}]: total={stats['total_seconds']:.2f}s "
                    f"calls={stats['calls']} mean={stats['mean_seconds']:.3f}s"
                )

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
