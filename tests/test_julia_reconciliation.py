"""Regression coverage for the safe Julia behavior ported from PR #560."""

from pathlib import Path

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build
from code_review_graph.parser import CodeParser


def _parse(source: str):
    return CodeParser().parse_bytes(
        Path("/repo/case.jl"),
        source.encode("utf-8"),
    )


def _qualified(node) -> str:
    if node.kind == "File":
        return node.file_path
    if node.parent_name:
        return f"{node.file_path}::{node.parent_name}.{node.name}"
    return f"{node.file_path}::{node.name}"


def test_function_stub_is_a_function():
    nodes, _ = _parse("function hook end")

    assert [(node.kind, node.name) for node in nodes if node.kind != "File"] == [
        ("Function", "hook")
    ]


def test_malformed_qualified_stub_fails_soft():
    nodes, edges = _parse("function A.B.hook end")

    assert [node.kind for node in nodes] == ["File"]
    assert edges == []


@pytest.mark.parametrize(
    ("signature", "expected"),
    [
        ("+(a, b) = a", "+"),
        ("Base.:+(a, b) = a", "+"),
        ("Base.:(==)(a, b) = true", "=="),
    ],
)
def test_operator_definition_uses_operator_name(signature, expected):
    nodes, _ = _parse(signature)

    assert [node.name for node in nodes if node.kind == "Function"] == [expected]


def test_parameterized_const_only_is_a_type():
    nodes, _ = _parse(
        "const FloatVec = Vector{Float64}\n"
        "const PairMap = Dict{String, Tuple{Int, Int}}\n"
        "const MAX_RETRIES = 3\n"
    )

    assert {node.name for node in nodes if node.kind == "Type"} == {"FloatVec", "PairMap"}


def test_import_alias_records_real_dependency():
    _, edges = _parse("import DataFrames as DF\nimport Tables: AbstractColumns as Columns\n")

    assert {edge.target for edge in edges if edge.kind == "IMPORTS_FROM"} == {
        "DataFrames",
        "Tables.AbstractColumns",
    }


def test_qualified_definitions_have_collision_free_identities():
    nodes, edges = _parse(
        "module Demo\n"
        "function show(x)\n"
        "    x\n"
        "end\n"
        "function Base.show(x)\n"
        "    x\n"
        "end\n"
        "Base.length(x) = x\n"
        "Base.:+(a, b) = a\n"
        "function A.B.run(x)\n"
        "    x\n"
        "end\n"
        "function Base()\n"
        "end\n"
        "end\n"
    )

    functions = [node for node in nodes if node.kind == "Function"]
    assert {(node.name, node.parent_name) for node in functions} >= {
        ("show", "Demo"),
        ("show", "Demo.Base"),
        ("length", "Demo.Base"),
        ("+", "Demo.Base"),
        ("run", "Demo.A.B"),
        ("Base", "Demo"),
    }
    assert {
        node.extra.get("julia_module_qualifier")
        for node in functions
        if node.name in {"length", "+", "run"}
    } == {"Base", "A.B"}

    qualifier_refs = [
        edge
        for edge in edges
        if edge.kind == "REFERENCES" and edge.extra.get("julia_qualified_def")
    ]
    assert any(
        edge.source == "/repo/case.jl::Demo.Base.show" and edge.target == "Base"
        for edge in qualifier_refs
    )


def test_short_form_body_call_resolves_to_local_function():
    _, edges = _parse("module Demo\ngreet(x) = x\ndelegate(x) = greet(x)\nend\n")

    assert any(
        edge.kind == "CALLS"
        and edge.source == "/repo/case.jl::Demo.delegate"
        and edge.target == "/repo/case.jl::Demo.greet"
        for edge in edges
    )


def test_module_scope_call_resolves_within_current_module():
    _, edges = _parse("module Demo\ninitialize() = nothing\ninitialize()\nend\n")

    assert any(
        edge.kind == "CALLS"
        and edge.source == "/repo/case.jl::Demo"
        and edge.target == "/repo/case.jl::Demo.initialize"
        for edge in edges
    )


def test_qualified_calls_keep_full_module_and_resolve_collisions():
    _, edges = _parse(
        "module Demo\n"
        "run(x) = x\n"
        "function A.B.run(x)\n"
        "    x\n"
        "end\n"
        "function caller(x)\n"
        "    run(x)\n"
        "    A.B.run(x)\n"
        "    LinearAlgebra.BLAS.gemv(x)\n"
        "end\n"
        "end\n"
    )

    calls = [edge for edge in edges if edge.kind == "CALLS"]
    targets = {edge.target for edge in calls}
    assert "/repo/case.jl::Demo.run" in targets
    assert "/repo/case.jl::Demo.A.B.run" in targets
    assert "LinearAlgebra.BLAS.gemv" in targets
    assert any(
        edge.target == "LinearAlgebra.BLAS.gemv"
        and edge.extra.get("julia_call_module") == "LinearAlgebra.BLAS"
        for edge in calls
    )


def test_nested_modules_and_functions_keep_complete_scope():
    nodes, edges = _parse(
        "module Outer\n"
        "f(x) = x\n"
        "module Inner\n"
        "f(x) = x + 1\n"
        "function wrapper(x)\n"
        "    function leaf(y)\n"
        "        f(y)\n"
        "    end\n"
        "    leaf(x)\n"
        "end\n"
        "end\n"
        "end\n"
    )

    identities = {
        (node.name, node.parent_name) for node in nodes if node.kind in {"Class", "Function"}
    }
    assert ("Inner", "Outer") in identities
    assert ("f", "Outer.Inner") in identities
    assert ("wrapper", "Outer.Inner") in identities
    assert ("leaf", "Outer.Inner.wrapper") in identities
    assert any(
        edge.kind == "CONTAINS"
        and edge.source == "/repo/case.jl::Outer"
        and edge.target == "/repo/case.jl::Outer.Inner"
        for edge in edges
    )
    assert any(
        edge.kind == "CALLS"
        and edge.source == "/repo/case.jl::Outer.Inner.wrapper.leaf"
        and edge.target == "/repo/case.jl::Outer.Inner.f"
        for edge in edges
    )


def test_calls_through_import_aliases_use_real_module_paths():
    _, edges = _parse(
        "module Demo\n"
        "import DataFrames as DF\n"
        "import Tables: AbstractColumns as Columns\n"
        "function caller(x)\n"
        "    DF.transform(x)\n"
        "    Columns(x)\n"
        "end\n"
        "end\n"
        "module Other\n"
        "import OtherFrames as DF\n"
        "function caller(x)\n"
        "    DF.transform(x)\n"
        "end\n"
        "end\n"
    )

    calls = [edge for edge in edges if edge.kind == "CALLS"]
    assert any(
        edge.source == "/repo/case.jl::Demo.caller"
        and edge.target == "DataFrames.transform"
        and edge.extra.get("julia_call_module") == "DataFrames"
        for edge in calls
    )
    assert any(edge.target == "Tables.AbstractColumns" for edge in calls)
    assert any(
        edge.source == "/repo/case.jl::Other.caller"
        and edge.target == "OtherFrames.transform"
        and edge.extra.get("julia_call_module") == "OtherFrames"
        for edge in calls
    )


def test_function_local_testset_and_macros_keep_canonical_scope():
    nodes, edges = _parse(
        "module Demo\n"
        "macro passthrough(ex)\n"
        "    ex\n"
        "end\n"
        "subject(x) = x\n"
        "function wrapper(x)\n"
        '    @testset "nested" begin\n'
        "        @test subject(x) == x\n"
        "    end\n"
        "    @inline subject(x)\n"
        "end\n"
        "end\n"
    )

    assert any(
        node.kind == "Function" and node.name == "passthrough" and node.parent_name == "Demo"
        for node in nodes
    )
    nested_testset = next(
        node for node in nodes if node.kind == "Test" and "testset:nested" in node.name
    )
    assert nested_testset.parent_name == "Demo.wrapper"
    testset_qn = _qualified(nested_testset)
    assert any(
        edge.kind == "CALLS"
        and edge.source == testset_qn
        and edge.target == "/repo/case.jl::Demo.subject"
        for edge in edges
    )
    assert any(
        edge.kind == "CALLS"
        and edge.source == "/repo/case.jl::Demo.wrapper"
        and edge.target == "@inline"
        for edge in edges
    )


def test_full_build_persists_distinct_qualified_nodes_and_callers(tmp_path):
    (tmp_path / ".git").mkdir()
    source_path = tmp_path / "analysis.jl"
    source_path.write_text(
        "module Demo\n"
        "show(x) = x\n"
        "function Base.show(x)\n"
        "    x\n"
        "end\n"
        "invoke(x) = Base.show(x)\n"
        "const FloatVec = Vector{Float64}\n"
        "end\n",
        encoding="utf-8",
    )

    store = GraphStore(tmp_path / "graph.db")
    try:
        result = full_build(tmp_path, store)
        local_qn = f"{source_path}::Demo.show"
        base_qn = f"{source_path}::Demo.Base.show"
        alias_qn = f"{source_path}::Demo.FloatVec"

        local_node = store.get_node(local_qn)
        base_node = store.get_node(base_qn)
        assert local_node is not None
        assert base_node is not None
        assert local_node.id != base_node.id
        assert base_node.extra["julia_module_qualifier"] == "Base"
        assert store.get_node(alias_qn) is not None

        callers = store.get_edges_by_target(base_qn)
        assert any(
            edge.kind == "CALLS" and edge.source_qualified == f"{source_path}::Demo.invoke"
            for edge in callers
        )
        assert result["errors"] == []
    finally:
        store.close()
