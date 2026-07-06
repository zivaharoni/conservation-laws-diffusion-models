"""Small reusable metric callback helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MetricHistory:
    steps: list[int] = field(default_factory=list)
    conservation_law: list[float] = field(default_factory=list)
    mse: list[float] = field(default_factory=list)
    divergence: list[float] = field(default_factory=list)
    nll: list[float] = field(default_factory=list)
    nll_ext: list[float] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    def append(
        self,
        step: int,
        conservation_law: float,
        mse: float,
        divergence: float,
        nll: float,
        nll_ext: float,
        loss: float,
    ) -> None:
        self.steps.append(int(step))
        self.conservation_law.append(float(conservation_law))
        self.mse.append(float(mse))
        self.divergence.append(float(divergence))
        self.nll.append(float(nll))
        self.nll_ext.append(float(nll_ext))
        self.loss.append(float(loss))

    def rows(self) -> list[dict[str, float]]:
        return [
            {
                "step": step,
                "conservation_law": conservation_law,
                "mse": mse,
                "divergence": divergence,
                "nll": nll,
                "nll_ext": nll_ext,
                "loss": loss,
            }
            for step, conservation_law, mse, divergence, nll, nll_ext, loss in zip(
                self.steps, self.conservation_law, self.mse, self.divergence, self.nll, self.nll_ext, self.loss
            )
        ]
