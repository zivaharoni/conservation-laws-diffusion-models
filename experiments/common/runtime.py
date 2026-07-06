"""Runtime helpers for data/model construction and setup summaries."""

from __future__ import annotations

import torch
from omegaconf import DictConfig, OmegaConf

from .experiment_registry import require_experiment_kind
from src.sources_iid import IidSourceData
from src.sources_markov import MarkovSourceData


def loss_name(cfg: DictConfig) -> str:
    require_experiment_kind(cfg, "immse")
    return str(cfg.loss.name).lower()


def loss_estimator(cfg: DictConfig) -> str:
    require_experiment_kind(cfg, "immse")
    return str(cfg.loss.estimator).lower()


def loss_topk(cfg: DictConfig) -> int | None:
    require_experiment_kind(cfg, "immse")
    topk = cfg.loss.topk
    return None if topk is None else int(topk)


def eval_expectation_method(cfg: DictConfig) -> str:
    estimator = loss_estimator(cfg)
    return "probs" if estimator == "topk" else estimator


def eval_expectation_topk(cfg: DictConfig) -> int | None:
    return loss_topk(cfg) if loss_estimator(cfg) == "topk" else None


def _immse_data_family(cfg: DictConfig) -> str:
    if "source_distribution" in cfg.data:
        return "iid"
    if "init_distribution" in cfg.data:
        return "markov"
    raise ValueError(
        "Unable to infer IMMSE data family from cfg.data; "
        "expected IID data with 'source_distribution' or Markov data with 'init_distribution'."
    )


def validate_config(cfg: DictConfig) -> None:
    require_experiment_kind(cfg, "immse")
    model_name = str(cfg.model.name).lower()
    if model_name not in {"immse_iid_model", "immse_seq_model"}:
        raise ValueError(
            f"Unsupported model.name={cfg.model.name!r}; "
            "expected 'immse_iid_model' or 'immse_seq_model'."
        )
    data_family = _immse_data_family(cfg)
    if data_family == "markov":
        init_distribution = str(getattr(cfg.data, "init_distribution", "stationary")).lower()
        if init_distribution not in {"stationary", "sampled"}:
            raise ValueError(
                f"Unsupported data.init_distribution={cfg.data.init_distribution!r}; "
                "expected 'stationary' or 'sampled'."
            )
    if data_family == "iid" and model_name != "immse_iid_model":
        raise ValueError("IID IMMSE configs must use model.name='immse_iid_model'.")
    if data_family == "markov" and model_name != "immse_seq_model":
        raise ValueError("Markov IMMSE configs must use model.name='immse_seq_model'.")
    if int(cfg.eval.n_snr_points) < 16:
        raise ValueError(
            "eval.n_snr_points is too small for a reliable final I-MMSE area estimate. "
            "Use eval.n_snr_points>=16."
        )
    if loss_name(cfg) not in {"mse_nll", "mse_nll_ext", "nll_ext", "nll"}:
        raise ValueError(
            f"Unsupported loss.name={cfg.loss.name!r}; "
            "expected 'mse_nll', 'mse_nll_ext', 'nll_ext', or 'nll'."
        )
    estimator = loss_estimator(cfg)
    topk = loss_topk(cfg)
    if estimator not in {"regression", "probs", "topk"}:
        raise ValueError(
            f"Unsupported loss.estimator={cfg.loss.estimator!r}; "
            "expected 'regression', 'probs', or 'topk'."
        )
    if topk is not None and estimator != "topk":
        raise ValueError("loss.topk requires loss.estimator='topk'.")
    if estimator == "topk" and topk is None:
        raise ValueError("loss.topk must be set when loss.estimator='topk'.")
    if topk is not None and int(topk) <= 0:
        raise ValueError("loss.topk must be a positive integer when set.")


def build_data_spec(cfg: DictConfig):
    data_family = _immse_data_family(cfg)
    if data_family == "iid":
        return IidSourceData(
            alphabet_size=int(cfg.data.alphabet_size),
            seed=int(cfg.seed),
            source_distribution=str(cfg.data.source_distribution),
            dirichlet_alpha=None if cfg.data.dirichlet_alpha is None else float(cfg.data.dirichlet_alpha),
        )
    if data_family == "markov":
        return MarkovSourceData(
            alphabet_size=int(cfg.data.alphabet_size),
            dirichlet_alpha=float(cfg.data.dirichlet_alpha),
            seed=int(cfg.seed),
            train_seq_len=int(cfg.train.seq_len),
            eval_seq_len=int(cfg.eval.seq_len),
            init_distribution=str(getattr(cfg.data, "init_distribution", "stationary")),
        )
    raise ValueError(
        "Unable to build IMMSE data spec; "
        "expected IID data with 'source_distribution' or Markov data with 'init_distribution'."
    )


def build_model(cfg: DictConfig, data_spec) -> torch.nn.Module:
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    for key, value in data_spec.model_init_kwargs().items():
        model_cfg[key] = value
    target = str(model_cfg.pop("_target_"))
    if target == "src.models.DiscretePriorGaussianRegressionModelEmbedding":
        from src.models import DiscretePriorGaussianRegressionModelEmbedding

        return DiscretePriorGaussianRegressionModelEmbedding(**OmegaConf.to_container(model_cfg, resolve=True))
    if target == "src.models.DiscretePriorGaussianRegressionModelEmbeddingSequence":
        from src.models import DiscretePriorGaussianRegressionModelEmbeddingSequence

        return DiscretePriorGaussianRegressionModelEmbeddingSequence(**OmegaConf.to_container(model_cfg, resolve=True))
    raise ValueError(f"Unsupported model target {target!r}.")


def print_reference_summary(data_spec) -> None:
    for label, value in data_spec.data_attributes():
        print(f"{label.replace('_', ' ').title()}:".ljust(24) + str(value))
    print(f"Reference entropy:      {data_spec.reference_entropy_nats:.6f} {data_spec.reference_unit}")
    stationary_entropy_nats = getattr(data_spec, "stationary_entropy_nats", None)
    if stationary_entropy_nats is not None:
        print(f"Stationary entropy:     {stationary_entropy_nats:.6f} nats")


def print_model_forward_summary(cfg: DictConfig, model: torch.nn.Module, data_spec) -> None:
    del cfg
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    y, snr = data_spec.dummy_forward_inputs(model, device=device, dtype=dtype)
    rows: list[tuple[str, str, str, str, int, str]] = []
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    for module_name, module in model.named_modules():
        if module is model or any(True for _ in module.children()):
            continue

        def _hook(
            _module: torch.nn.Module,
            inputs: tuple[object, ...],
            output: object,
            *,
            name: str = module_name,
            current_module: torch.nn.Module = module,
        ) -> None:
            local_params = sum(param.numel() for param in current_module.parameters(recurse=False))
            parameter_names = ", ".join(param_name for param_name, _ in current_module.named_parameters(recurse=False)) or "-"
            rows.append(
                (
                    name,
                    current_module.__class__.__name__,
                    _format_structure_shape(inputs),
                    _format_structure_shape(output),
                    local_params,
                    parameter_names,
                )
            )

        hooks.append(module.register_forward_hook(_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(y, snr)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    print("[setup] model forward summary:")
    print(f"  class={model.__class__.__name__}")
    print(f"  params trainable={trainable_params:,} total={total_params:,}")
    print(f"  inputs y={_format_structure_shape(y)} snr={_format_structure_shape(snr)}")
    print(f"  outputs={_format_structure_shape(outputs)}")
    for name, module_type, input_shape, output_shape, local_params, parameter_names in rows:
        print(
            f"  - {name} [{module_type}]: "
            f"in={input_shape} out={output_shape} params={local_params:,} names={parameter_names}"
        )


def _format_structure_shape(value: object) -> str:
    if isinstance(value, torch.Tensor):
        return str(tuple(int(dim) for dim in value.shape))
    if isinstance(value, (list, tuple)):
        inner = ", ".join(_format_structure_shape(item) for item in value)
        open_bracket, close_bracket = ("[", "]") if isinstance(value, list) else ("(", ")")
        return f"{open_bracket}{inner}{close_bracket}"
    if isinstance(value, dict):
        inner = ", ".join(f"{key}: {_format_structure_shape(item)}" for key, item in value.items())
        return "{" + inner + "}"
    return type(value).__name__
