"""Unified multi-dimensional communicator mesh for visual generation models.

VisualGenMapping subclasses DeviceMeshTopologyImpl and overrides build_mesh()
to create a single PyTorch DeviceMesh covering all parallelism axes
(CFG, TP, CP, Ulysses).  The resulting mesh is stored in the shared
DeviceMeshTopologyImpl.device_mesh class variable so that any Mapping object
constructed afterward (e.g. via to_llm_mapping()) can reuse the same
process groups.

Parallelism taxonomy
--------------------
- CFG: splits classifier-free-guidance (positive/negative) prompts.
- TP:  tensor-parallel Linear weight sharding.
- CP:  context parallelism.  A single *size* with a selectable *implementation*:
       * ``cp_impl == "ring"``  -> ring-attention over the CP group.
- Ulysses: all-to-all sequence parallelism over the ulysses mesh dim.

The outer sequence-parallel dimension exposed to models is
``sp = cp * ulysses`` (a flattened ("cp", "ulysses") sub-mesh when both are
active).  Models only need ``sp_size / sp_rank / sp_group`` to shard inputs;
the parallel-attention factory consumes ``cp_impl`` and ``cp_group`` to build
the right attention wrapper internally.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.device_mesh import init_device_mesh

from tensorrt_llm._torch.device_mesh import DeviceMeshTopologyImpl, SingleProcessGroup
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

CpImpl = Literal["ring"]

_VALID_DIM_NAMES = frozenset({"cfg", "tp", "cp", "ulysses"})
DEFAULT_DIM_ORDER = "cfg-tp-cp-ulysses"


class VisualGenMapping(DeviceMeshTopologyImpl):
    """Multi-dimensional communicator mesh for visual generation models.

    Ordering rationale (default ``"cfg-tp-cp-ulysses"``):
    - Ulysses innermost: all-to-all is latency-sensitive, contiguous ranks.
    - CP next (Ring KV streaming): benefits from adjacency.
    - TP next: Linear all-reduce.
    - CFG outermost: independent until the final all-gather.

    The *order* string maps directly to ``init_device_mesh``'s ``mesh_shape``
    tuple (first = outermost / slowest-varying, last = innermost / most
    contiguous).  CP and Ulysses must be adjacent whenever both are > 1 so
    that ``("cp", "ulysses")._flatten()`` yields a valid SP mesh.
    """

    # Flattened ("cp", "ulysses") sub-mesh, cached after build_mesh().
    sp_mesh = None

    def __init__(
        self,
        world_size: int,
        rank: int,
        cfg_size: int = 1,
        tp_size: int = 1,
        cp_size: int = 1,
        cp_impl: Optional[CpImpl] = None,
        ulysses_size: int = 1,
        order: str = DEFAULT_DIM_ORDER,
    ):
        cp_impl = self._canonicalise_cp(cp_size, cp_impl)

        product = cfg_size * tp_size * cp_size * ulysses_size
        if product != world_size:
            raise ValueError(
                f"cfg({cfg_size}) * tp({tp_size}) * cp({cp_size}) * "
                f"ulysses({ulysses_size}) = {product} != world_size({world_size})"
            )

        dims = order.split("-")
        if set(dims) != _VALID_DIM_NAMES or len(dims) != len(_VALID_DIM_NAMES):
            raise ValueError(
                f"order must be a '-'-separated permutation of "
                f"{sorted(_VALID_DIM_NAMES)}, got '{order}'"
            )

        self.world_size = world_size
        self._rank = rank
        self.cfg_size = cfg_size
        self.tp_size = tp_size
        self.cp_size = cp_size
        self.cp_impl = cp_impl
        self.ulysses_size = ulysses_size
        self._order = order
        self._dim_names = tuple(dims)
        self._dim_sizes = {
            "cfg": cfg_size,
            "tp": tp_size,
            "cp": cp_size,
            "ulysses": ulysses_size,
        }

        if dist.is_initialized() and world_size > 1:
            self.build_mesh()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _canonicalise_cp(cp_size: int, cp_impl: Optional[CpImpl]) -> Optional[CpImpl]:
        if cp_size < 1:
            raise ValueError(f"cp_size must be >= 1, got {cp_size}")

        if cp_size == 1:
            if cp_impl is not None:
                raise ValueError(f"cp_impl='{cp_impl}' requires cp_size > 1 (got cp_size=1)")
            return None

        if cp_impl not in ("ring",):
            raise ValueError(f"cp_size > 1 requires cp_impl='ring', got {cp_impl!r}")
        return cp_impl

    # ------------------------------------------------------------------
    # Mesh construction
    # ------------------------------------------------------------------
    def build_mesh(self):
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is not None:
            return

        shape = tuple(self._dim_sizes[d] for d in self._dim_names)
        cls.device_mesh = init_device_mesh(
            "cuda",
            mesh_shape=shape,
            mesh_dim_names=self._dim_names,
        )
        logger.debug(
            f"VisualGenMapping.build_mesh: dims={self._dim_names}, "
            f"shape={shape}, mesh={cls.device_mesh}"
        )

        # SP = cp x ulysses (only needed when both dims > 1; otherwise sp_group
        # aliases whichever single dim is active).
        if self.cp_size > 1 and self.ulysses_size > 1:
            cp_idx = self._dim_names.index("cp")
            uly_idx = self._dim_names.index("ulysses")
            if abs(cp_idx - uly_idx) != 1:
                raise ValueError(
                    "cp and ulysses must be adjacent in mesh order to flatten "
                    f"into sp (got order='{self._order}')"
                )
            VisualGenMapping.sp_mesh = cls.device_mesh["cp", "ulysses"]._flatten(mesh_dim_name="sp")

    # ------------------------------------------------------------------
    # Rank decomposition
    # ------------------------------------------------------------------
    def _local_rank(self, dim: str) -> int:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            return 0
        return cls.device_mesh[dim].get_local_rank()

    @property
    def cfg_rank(self) -> int:
        return self._local_rank("cfg")

    @property
    def tp_rank(self) -> int:
        return self._local_rank("tp")

    @property
    def cp_rank(self) -> int:
        return self._local_rank("cp")

    @property
    def ulysses_rank(self) -> int:
        return self._local_rank("ulysses")

    @property
    def is_cfg_conditional(self) -> bool:
        return self.cfg_rank == 0

    # ------------------------------------------------------------------
    # Process groups (None when size == 1 and mesh was not built)
    # ------------------------------------------------------------------
    def _group(self, dim: str) -> Optional[ProcessGroup]:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            if self.world_size == 1:
                return SingleProcessGroup.get_group()
            return None
        return cls.device_mesh[dim].get_group()

    @property
    def cfg_group(self) -> Optional[ProcessGroup]:
        return self._group("cfg")

    @property
    def tp_group_pg(self) -> Optional[ProcessGroup]:
        return self._group("tp")

    @property
    def cp_group(self) -> Optional[ProcessGroup]:  # type: ignore[override]
        return self._group("cp")

    @property
    def ulysses_group(self) -> Optional[ProcessGroup]:
        return self._group("ulysses")

    # ------------------------------------------------------------------
    # Sequence-parallel view (cp x ulysses)
    # ------------------------------------------------------------------
    @property
    def sp_size(self) -> int:
        return self.cp_size * self.ulysses_size

    @property
    def sp_rank(self) -> int:
        # Innermost-fastest index: sweep ulysses inside cp.
        return self.cp_rank * self.ulysses_size + self.ulysses_rank

    @property
    def sp_group(self) -> Optional[ProcessGroup]:
        cls = DeviceMeshTopologyImpl
        if cls.device_mesh is None:
            if self.world_size == 1:
                return SingleProcessGroup.get_group()
            return None
        if self.cp_size > 1 and self.ulysses_size > 1:
            assert VisualGenMapping.sp_mesh is not None, (
                "sp_mesh should have been built in build_mesh()"
            )
            return VisualGenMapping.sp_mesh.get_group()
        if self.cp_size > 1:
            return self._group("cp")
        if self.ulysses_size > 1:
            return self._group("ulysses")
        # Degenerate (sp_size == 1): return whichever trivial group.
        return self._group("ulysses")

    # ------------------------------------------------------------------
    # Bridge to LLM Mapping (for Linear layers)
    # ------------------------------------------------------------------
    def to_llm_mapping(self) -> Mapping:
        """Return a ``Mapping`` whose TP group is backed by this mesh's TP dim.

        ``build_mesh()`` has already populated
        ``DeviceMeshTopologyImpl.device_mesh``, so the returned ``Mapping``'s
        ``build_mesh()`` is a no-op and ``tp_group_pg`` reads from the shared
        mega-mesh.
        """
        return Mapping(
            world_size=self.tp_size,
            rank=self.tp_rank,
            tp_size=self.tp_size,
        )
