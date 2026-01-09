import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from peft import load_peft_weights

from hyper_llm_modulator.trajectory_search import (
    ActionSpace,
    LoRABasis,
    TrajectoryState,
    coeffs_to_lora,
    mcts_search,
    uniform_policy,
)
from hyper_llm_modulator.utils.lora_formatting import lora_state_dict_to_tensor_dict
import yaml

from hyper_llm_modulator.utils import get_target_lora_dirs


def parse_list(val: str, cast):
    if val is None or val == "":
        return []
    return [cast(x) for x in val.split(",")]


def get_oracle_lora_dirs(
    train_ds_names: Sequence[str],
    model_dir: str,
    oracle_root: str,
    allow_missing: bool,
) -> Dict[str, str]:
    lora_dirs: Dict[str, str] = {}
    if not os.path.isdir(oracle_root):
        if allow_missing:
            return lora_dirs
        raise ValueError(f"Oracle LoRA root not found: {oracle_root}")

    for subdir in Path(oracle_root).glob("*/"):
        args_path = subdir / "args.yaml"
        model_path = subdir / "adapter_model.safetensors"
        if not args_path.exists() or not model_path.exists():
            continue
        lora_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        if model_dir.strip("/") != str(lora_args.get("model_dir", "")).strip("/"):
            continue
        train_datasets = lora_args.get("train_ds_names", lora_args.get("datasets", []))
        if len(train_datasets) != 1:
            continue
        ds_name = train_datasets[0]
        if ds_name in train_ds_names:
            lora_dirs[ds_name] = str(subdir)

    missing = set(train_ds_names) - set(lora_dirs.keys())
    if missing and not allow_missing:
        raise ValueError(f"Missing oracle lora directories for datasets: {missing}")
    return lora_dirs


def infer_layer_indices(lora_sd: Dict[str, torch.Tensor]) -> List[int]:
    layers = set()
    for key in lora_sd.keys():
        if ".layers." not in key:
            continue
        layer_str = key.split("layers.")[-1].split(".")[0]
        if layer_str.isdigit():
            layers.add(int(layer_str))
    if not layers:
        raise ValueError("Could not infer any layer indices from LoRA state dict.")
    return sorted(layers)


def infer_module_shapes(
    lora_sd: Dict[str, torch.Tensor], target_modules: Sequence[str]
) -> Tuple[Dict[str, int], Dict[str, int], int]:
    in_features: Dict[str, int] = {}
    out_features: Dict[str, int] = {}
    r = None
    for module in target_modules:
        for key, weight in lora_sd.items():
            if module not in key:
                continue
            if "lora_A" in key:
                r = weight.shape[0]
                in_features[module] = weight.shape[1]
            elif "lora_B" in key:
                out_features[module] = weight.shape[0]
        if module not in in_features or module not in out_features:
            raise ValueError(f"Missing lora_A or lora_B for module: {module}")
    if r is None:
        raise ValueError("Could not infer LoRA rank r from state dict.")
    return in_features, out_features, r


def build_basis(
    lora_state_dicts: Sequence[Dict[str, torch.Tensor]],
    target_modules: Sequence[str],
    layer_indices: Sequence[int],
    basis_size: int,
    device: torch.device,
    center: bool,
) -> Tuple[LoRABasis, Dict[str, int], Dict[str, int], int]:
    in_features, out_features, r = infer_module_shapes(lora_state_dicts[0], target_modules)
    basis = LoRABasis.from_loras(
        lora_state_dicts=lora_state_dicts,
        target_modules=target_modules,
        layer_indices=layer_indices,
        r=r,
        in_features=in_features,
        out_features=out_features,
        basis_size=basis_size,
        device=device,
        center=center,
    )
    return basis, in_features, out_features, r


def compute_recon_value(
    state: TrajectoryState,
    basis: LoRABasis,
    target_lora_td: Dict[str, torch.Tensor],
    include_mean: bool,
) -> float:
    lora_A, lora_B = coeffs_to_lora(basis, state.coeffs, include_mean=include_mean)
    loss = 0.0
    for module in lora_A:
        loss += torch.nn.functional.l1_loss(lora_A[module], target_lora_td["A"][module]).item()
        loss += torch.nn.functional.l1_loss(lora_B[module], target_lora_td["B"][module]).item()
    loss /= (2 * len(lora_A))
    return -loss


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MCTS trajectories in LoRA-weight space.")
    parser.add_argument("--model-dir", type=str, default="", help="Base model dir (used with train_ds_names).")
    parser.add_argument("--train-ds-names", type=str, default="", help="Comma list of task names.")
    parser.add_argument("--lora-dirs", type=str, default="", help="Comma list of LoRA dirs (overrides train_ds_names).")
    parser.add_argument("--target-modules", type=str, default="", help="Comma list of target modules.")
    parser.add_argument("--basis-size", type=int, default=32, help="Number of basis vectors per module.")
    parser.add_argument("--scales", type=str, default="-1.0,-0.5,-0.25,0.25,0.5,1.0", help="Scale bins.")
    parser.add_argument("--max-depth", type=int, default=5, help="Trajectory length (K).")
    parser.add_argument("--num-simulations", type=int, default=128, help="MCTS rollouts per search.")
    parser.add_argument("--num-trajectories-per-task", type=int, default=4, help="Trajectories per task.")
    parser.add_argument("--max-actions", type=int, default=256, help="Max actions per expansion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda.")
    parser.add_argument("--center-basis", action="store_true", help="Center LoRA vectors before SVD.")
    parser.add_argument("--include-mean", action="store_true", help="Add mean back when reconstructing.")
    parser.add_argument(
        "--oracle-lora-root",
        type=str,
        default="train_outputs/sft/oracle_lora",
        help="Root directory containing oracle LoRA checkpoints.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip missing oracle LoRA tasks instead of failing.",
    )
    parser.add_argument("--output-path", type=str, default="data/mcts_trajectories.jsonl", help="Output JSONL.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    lora_dirs: Dict[str, str] = {}
    if args.lora_dirs:
        for lora_dir in parse_list(args.lora_dirs, str):
            lora_dirs[Path(lora_dir).name] = lora_dir
    else:
        train_ds_names = parse_list(args.train_ds_names, str)
        if not train_ds_names or not args.model_dir:
            raise ValueError("Provide --lora-dirs or both --train-ds-names and --model-dir.")
        try:
            lora_dirs = get_target_lora_dirs(train_ds_names, args.model_dir)
        except ValueError:
            lora_dirs = get_oracle_lora_dirs(
                train_ds_names,
                args.model_dir,
                args.oracle_lora_root,
                allow_missing=args.allow_missing,
            )
            if not lora_dirs:
                raise

    if not lora_dirs:
        raise ValueError("No LoRA directories found.")

    first_lora = next(iter(lora_dirs.values()))
    adapter_config_path = Path(first_lora) / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {first_lora}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    target_modules = parse_list(args.target_modules, str) or adapter_config.get("target_modules", [])
    if not target_modules:
        raise ValueError("No target_modules provided or found in adapter config.")

    lora_state_dicts = {task: load_peft_weights(path, device=device) for task, path in lora_dirs.items()}
    layer_indices = infer_layer_indices(next(iter(lora_state_dicts.values())))

    basis, _in_features, _out_features, _r = build_basis(
        list(lora_state_dicts.values()),
        target_modules=target_modules,
        layer_indices=layer_indices,
        basis_size=args.basis_size,
        device=device,
        center=args.center_basis,
    )

    scales = parse_list(args.scales, float)
    action_space = ActionSpace(
        modules=target_modules,
        layer_indices=layer_indices,
        basis_size=args.basis_size,
        scales=scales,
        max_actions=args.max_actions,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    meta = dict(
        model_dir=args.model_dir,
        target_modules=target_modules,
        layer_indices=layer_indices,
        basis_size=args.basis_size,
        scales=scales,
        max_depth=args.max_depth,
        num_simulations=args.num_simulations,
        num_trajectories_per_task=args.num_trajectories_per_task,
        lora_dirs=lora_dirs,
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    rows: List[Dict] = []
    for task_name, lora_sd in lora_state_dicts.items():
        target_lora_td = lora_state_dict_to_tensor_dict(
            lora_sd, target_modules, layer_indices, device
        )
        coeffs = {
            module: torch.zeros(len(layer_indices), args.basis_size, device=device)
            for module in target_modules
        }
        root_state = TrajectoryState(coeffs=coeffs, layer_indices=list(layer_indices))

        def policy_fn(state, action_space_ref):
            return uniform_policy(state, action_space_ref, max_actions=args.max_actions, rng=rng)

        def value_fn(state):
            return compute_recon_value(state, basis, target_lora_td, include_mean=args.include_mean)

        for traj_idx in range(args.num_trajectories_per_task):
            trajectory, root = mcts_search(
                root_state=root_state,
                action_space=action_space,
                policy_fn=policy_fn,
                value_fn=value_fn,
                num_simulations=args.num_simulations,
                max_depth=args.max_depth,
                c_puct=1.5,
                rng=rng,
            )
            final_state = root_state
            for action in trajectory:
                final_state = final_state.apply(action)
            final_value = value_fn(final_state)
            actions_serialized = [
                {
                    "module": action.module,
                    "layer_pos": action.layer_pos,
                    "layer_idx": layer_indices[action.layer_pos],
                    "basis_id": action.basis_id,
                    "scale": action.scale,
                }
                for action in trajectory
            ]
            rows.append(
                dict(
                    task_name=task_name,
                    lora_dir=lora_dirs[task_name],
                    trajectory_idx=traj_idx,
                    actions=actions_serialized,
                    final_value=final_value,
                )
            )

    write_jsonl(output_path, rows)


if __name__ == "__main__":
    main()
