import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from hyper_llm_modulator.utils.lora_formatting import lora_state_dict_to_tensor_dict


@dataclass(frozen=True)
class Action:
    module: str
    layer_pos: int
    basis_id: int
    scale: float


@dataclass
class LoRABasis:
    basis_A: Dict[str, torch.Tensor]
    basis_B: Dict[str, torch.Tensor]
    mean_A: Dict[str, torch.Tensor]
    mean_B: Dict[str, torch.Tensor]
    basis_size: int
    r: int
    in_features: Dict[str, int]
    out_features: Dict[str, int]

    @staticmethod
    def from_loras(
        lora_state_dicts: Sequence[Dict[str, torch.Tensor]],
        target_modules: Sequence[str],
        layer_indices: Sequence[int],
        r: int,
        in_features: Dict[str, int],
        out_features: Dict[str, int],
        basis_size: int,
        device: torch.device,
        center: bool = True,
    ) -> "LoRABasis":
        layer_indices = list(layer_indices)
        lora_tds = [
            lora_state_dict_to_tensor_dict(sd, target_modules, layer_indices, device)
            for sd in lora_state_dicts
        ]

        basis_A: Dict[str, torch.Tensor] = {}
        basis_B: Dict[str, torch.Tensor] = {}
        mean_A: Dict[str, torch.Tensor] = {}
        mean_B: Dict[str, torch.Tensor] = {}

        for module in target_modules:
            vectors: List[torch.Tensor] = []
            for lora_td in lora_tds:
                A = lora_td["A"][module]
                B = lora_td["B"][module]
                for layer_idx in range(len(layer_indices)):
                    vec = torch.cat([A[layer_idx].reshape(-1), B[layer_idx].reshape(-1)])
                    vectors.append(vec)

            if not vectors:
                raise ValueError(f"No LoRA tensors found for module: {module}")

            mat = torch.stack(vectors, dim=0)
            mean_vec = mat.mean(dim=0)
            if center:
                mat = mat - mean_vec

            # basis vectors are the top right-singular vectors
            _, _, v_h = torch.linalg.svd(mat, full_matrices=False)
            basis = v_h[:basis_size]

            a_dim = r * in_features[module]
            basis_A[module] = basis[:, :a_dim].reshape(basis_size, r, in_features[module])
            basis_B[module] = basis[:, a_dim:].reshape(basis_size, out_features[module], r)
            mean_A[module] = mean_vec[:a_dim].reshape(r, in_features[module])
            mean_B[module] = mean_vec[a_dim:].reshape(out_features[module], r)

        return LoRABasis(
            basis_A=basis_A,
            basis_B=basis_B,
            mean_A=mean_A,
            mean_B=mean_B,
            basis_size=basis_size,
            r=r,
            in_features=in_features,
            out_features=out_features,
        )


@dataclass
class TrajectoryState:
    coeffs: Dict[str, torch.Tensor]
    layer_indices: List[int]

    def clone(self) -> "TrajectoryState":
        return TrajectoryState(
            coeffs={k: v.clone() for k, v in self.coeffs.items()},
            layer_indices=list(self.layer_indices),
        )

    def apply(self, action: Action) -> "TrajectoryState":
        new_state = self.clone()
        new_state.coeffs[action.module][action.layer_pos, action.basis_id] += action.scale
        return new_state


class ActionSpace:
    def __init__(
        self,
        modules: Sequence[str],
        layer_indices: Sequence[int],
        basis_size: int,
        scales: Sequence[float],
        max_actions: Optional[int] = None,
    ) -> None:
        self.modules = list(modules)
        self.layer_indices = list(layer_indices)
        self.basis_size = basis_size
        self.scales = list(scales)
        self.max_actions = max_actions

    def all_actions(self) -> List[Action]:
        actions = [
            Action(module, layer_pos, basis_id, scale)
            for module in self.modules
            for layer_pos, _layer_id in enumerate(self.layer_indices)
            for basis_id in range(self.basis_size)
            for scale in self.scales
        ]
        if self.max_actions is not None and len(actions) > self.max_actions:
            return random.sample(actions, self.max_actions)
        return actions

    def sample_actions(self, k: int, rng: random.Random) -> List[Action]:
        all_actions = self.all_actions()
        if k >= len(all_actions):
            return all_actions
        return rng.sample(all_actions, k)


def coeffs_to_lora(
    basis: LoRABasis,
    coeffs: Dict[str, torch.Tensor],
    include_mean: bool = True,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    lora_A: Dict[str, torch.Tensor] = {}
    lora_B: Dict[str, torch.Tensor] = {}

    for module, coeff in coeffs.items():
        A = torch.einsum("lb,brf->lrf", coeff, basis.basis_A[module])
        B = torch.einsum("lb,bor->lor", coeff, basis.basis_B[module])
        if include_mean:
            A = A + basis.mean_A[module].unsqueeze(0)
            B = B + basis.mean_B[module].unsqueeze(0)
        lora_A[module] = A
        lora_B[module] = B
    return lora_A, lora_B


class MCTSNode:
    def __init__(
        self,
        state: TrajectoryState,
        parent: Optional["MCTSNode"],
        prior: float,
        depth: int,
    ) -> None:
        self.state = state
        self.parent = parent
        self.prior = prior
        self.depth = depth
        self.children: Dict[Action, "MCTSNode"] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def mcts_search(
    root_state: TrajectoryState,
    action_space: ActionSpace,
    policy_fn: Callable[[TrajectoryState, ActionSpace], Dict[Action, float]],
    value_fn: Callable[[TrajectoryState], float],
    num_simulations: int,
    max_depth: int,
    c_puct: float = 1.5,
    rng: Optional[random.Random] = None,
) -> Tuple[List[Action], MCTSNode]:
    rng = rng or random.Random()
    root = MCTSNode(state=root_state, parent=None, prior=1.0, depth=0)

    def ucb_score(parent: MCTSNode, child: MCTSNode) -> float:
        prior = child.prior
        prior_score = c_puct * prior * math.sqrt(parent.visit_count + 1) / (1 + child.visit_count)
        return child.value + prior_score

    def select(node: MCTSNode) -> MCTSNode:
        while node.children:
            node = max(node.children.values(), key=lambda child: ucb_score(node, child))
        return node

    def expand(node: MCTSNode) -> None:
        priors = policy_fn(node.state, action_space)
        for action, prior in priors.items():
            if action in node.children:
                continue
            child_state = node.state.apply(action)
            node.children[action] = MCTSNode(
                state=child_state, parent=node, prior=float(prior), depth=node.depth + 1
            )

    def backup(node: MCTSNode, value: float) -> None:
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    for _ in range(num_simulations):
        leaf = select(root)
        if leaf.depth < max_depth:
            expand(leaf)
        value = value_fn(leaf.state)
        backup(leaf, value)

    # extract greedy trajectory by visits
    trajectory: List[Action] = []
    node = root
    while node.children and node.depth < max_depth:
        action, node = max(node.children.items(), key=lambda item: item[1].visit_count)
        trajectory.append(action)
    return trajectory, root


def uniform_policy(
    state: TrajectoryState,
    action_space: ActionSpace,
    max_actions: int = 256,
    rng: Optional[random.Random] = None,
) -> Dict[Action, float]:
    rng = rng or random.Random()
    actions = action_space.sample_actions(max_actions, rng)
    if not actions:
        return {}
    prior = 1.0 / len(actions)
    return {action: prior for action in actions}
