from __future__ import annotations

from pathlib import Path

from simulation.validation import discover_xosc_files, validate_xosc_file


def _write_minimal_xosc(path: Path, *, non_monotonic: bool = False, bad_root: bool = False) -> None:
	root = "BadRoot" if bad_root else "OpenSCENARIO"
	v0 = "1.0" if non_monotonic else "0.0"
	text = f"""<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<{root}>
  <FileHeader revMajor=\"1\" revMinor=\"0\" date=\"2026-03-16T00:00:00Z\" description=\"test\" author=\"unit\"/>
  <Entities>
    <ScenarioObject name=\"ego\" />
  </Entities>
  <Storyboard>
    <Story name=\"s\">
      <Act name=\"a\">
        <ManeuverGroup name=\"mg\" maximumExecutionCount=\"1\">
          <Maneuver name=\"m\">
            <Event name=\"e\" priority=\"overwrite\">
              <Action name=\"ac\">
                <PrivateAction>
                  <RoutingAction>
                    <FollowTrajectoryAction>
                      <Trajectory name=\"traj\" closed=\"false\">
                        <Shape>
                          <Polyline>
                            <Vertex time=\"{v0}\"><Position><WorldPosition x=\"0\" y=\"0\" z=\"0\" h=\"0\" p=\"0\" r=\"0\"/></Position></Vertex>
                            <Vertex time=\"0.5\"><Position><WorldPosition x=\"1\" y=\"0\" z=\"0\" h=\"0\" p=\"0\" r=\"0\"/></Position></Vertex>
                          </Polyline>
                        </Shape>
                      </Trajectory>
                    </FollowTrajectoryAction>
                  </RoutingAction>
                </PrivateAction>
              </Action>
            </Event>
          </Maneuver>
        </ManeuverGroup>
      </Act>
    </Story>
  </Storyboard>
</{root}>
"""
	path.write_text(text, encoding="utf-8")


def test_discover_xosc_files(tmp_path: Path) -> None:
	good = tmp_path / "a.xosc"
	other = tmp_path / "b.txt"
	_write_minimal_xosc(good)
	other.write_text("noop", encoding="utf-8")
	found = discover_xosc_files(str(tmp_path))
	assert [p.name for p in found] == ["a.xosc"]


def test_validate_xosc_file_pass(tmp_path: Path) -> None:
	path = tmp_path / "good.xosc"
	_write_minimal_xosc(path)
	result = validate_xosc_file(str(path))
	assert result.well_formed is True
	assert result.semantic_valid is True
	assert result.is_valid is True
	assert result.errors == []


def test_validate_xosc_file_bad_root_fails(tmp_path: Path) -> None:
	path = tmp_path / "bad_root.xosc"
	_write_minimal_xosc(path, bad_root=True)
	result = validate_xosc_file(str(path))
	assert result.is_valid is False
	assert any("Root element" in msg for msg in result.errors)


def test_validate_xosc_non_monotonic_vertex_time_fails(tmp_path: Path) -> None:
	path = tmp_path / "bad_time.xosc"
	_write_minimal_xosc(path, non_monotonic=True)
	result = validate_xosc_file(str(path))
	assert result.is_valid is False
	assert any("non-monotonic" in msg for msg in result.errors)
