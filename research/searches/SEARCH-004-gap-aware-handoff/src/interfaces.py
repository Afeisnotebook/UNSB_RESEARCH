"""Target-blind SEARCH-004 state-transport interfaces.

These interfaces deliberately operate on complete immutable snapshots.  A
transport may change only components declared by its mask and must return an
audit record that can be checked without a paired restoration target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .protocol import assert_target_blind
from .state import ComponentMask, FullTrainingStateV2, cpu_clone, torch_digest


@dataclass(frozen=True)
class StateObservation:
    fields: dict

    def __post_init__(self) -> None:
        assert_target_blind(self.fields)


@dataclass(frozen=True)
class ContinuationSeed:
    rng: dict
    stream_a: dict
    stream_b: dict
    global_clock: dict
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "digest",
            torch_digest({
                "rng": self.rng,
                "stream_a": self.stream_a,
                "stream_b": self.stream_b,
                "global_clock": self.global_clock,
            }),
        )

    @classmethod
    def from_state(cls, state: FullTrainingStateV2) -> "ContinuationSeed":
        return cls(
            rng=cpu_clone(state.rng),
            stream_a=cpu_clone(state.stream_a),
            stream_b=cpu_clone(state.stream_b),
            global_clock=cpu_clone(state.global_clock),
        )


@dataclass(frozen=True)
class HandoffAuditRecord:
    operator: str
    parent_digest: str
    result_digest: str
    changed_components: ComponentMask
    observation: StateObservation
    identity: bool
    paired_target_access: bool = False

    def __post_init__(self) -> None:
        if self.paired_target_access:
            raise ValueError("handoff audit cannot report paired-target access")


class HandoffOperator(ABC):
    """A target-blind, state-complete handoff transformation."""

    name: str
    component_mask: ComponentMask

    @abstractmethod
    def observe(self, state: FullTrainingStateV2) -> StateObservation:
        raise NotImplementedError

    @abstractmethod
    def transport(
        self, state: FullTrainingStateV2
    ) -> tuple[FullTrainingStateV2, HandoffAuditRecord]:
        raise NotImplementedError


class IdentityHandoff(HandoffOperator):
    name = "identity"
    component_mask = ComponentMask()

    def observe(self, state: FullTrainingStateV2) -> StateObservation:
        return StateObservation({"state_digest": state.digest(), "active": False})

    def transport(
        self, state: FullTrainingStateV2
    ) -> tuple[FullTrainingStateV2, HandoffAuditRecord]:
        result = FullTrainingStateV2(**cpu_clone(state.__dict__))
        observation = self.observe(state)
        record = HandoffAuditRecord(
            operator=self.name,
            parent_digest=state.digest(),
            result_digest=result.digest(),
            changed_components=self.component_mask,
            observation=observation,
            identity=True,
        )
        return result, record


class ShadowNativeAccumulator:
    """Adapter for transactional native G/F Adam-state accumulation."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def accumulate(self, steps: int) -> None:
        self.engine._reconstruct_local_native_moments(int(steps))


class CoStateEquilibrator:
    """Adapter for a G/G-optimizer-frozen D/E/F equilibration exposure."""

    def __init__(self, engine) -> None:
        self.engine = engine

    def step(self) -> None:
        self.engine._hold_step(equilibrate=True)


class ContinuationAuditor:
    """Verify that a serial counterfactual leaves its immutable parent intact."""

    def __init__(self, parent) -> None:
        self.parent = parent
        self.parent_digest = torch_digest(parent)

    def verify(self) -> None:
        if torch_digest(self.parent) != self.parent_digest:
            raise RuntimeError("continuation branch mutated its immutable parent")
