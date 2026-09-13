"""UI-neutral view model for a future Qt/VTK front end."""

from dataclasses import dataclass

from .pipeline import PipelineResult


@dataclass(frozen=True)
class PipelineViewModel:
    """Stable data boundary that Qt/VTK can consume without importing either."""
    point_count: int
    instance_count: int
    report_rows: tuple

    @classmethod
    def from_result(cls, result: PipelineResult) -> "PipelineViewModel":
        rows = tuple({"pipe_id": row.pipe_id, "geometry": row.geometry,
                      "estimated_radius": row.estimated_radius,
                      "nominal_radius": row.nominal_radius, "passed": row.passed}
                     for row in result.compliance)
        return cls(len(result.processed_cloud.points), len(result.instances), rows)
