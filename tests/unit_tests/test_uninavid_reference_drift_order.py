import ast
from pathlib import Path


def _find_function(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{class_node.name}.{name} was not found.")


def _walk_calls(function_node: ast.FunctionDef, name: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == name:
            calls.append(node)
    return calls


def _walk_method_calls(function_node: ast.FunctionDef, name: str) -> list[ast.Call]:
    calls = []
    for node in ast.walk(function_node):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == name:
            calls.append(node)
    return calls


def test_uninavid_reference_drift_weight_swap_runs_after_backward():
    source = Path("rlinf/workers/actor/fsdp_actor_worker.py").read_text()
    module = ast.parse(source)
    embodied_actor = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "EmbodiedFSDPActor"
    )
    run_training = _find_function(embodied_actor, "run_training")

    cpu_weight_swap_calls = _walk_calls(run_training, "cpu_weight_swap")
    backward_calls = _walk_method_calls(run_training, "backward")

    assert cpu_weight_swap_calls, "reference drift must still use cpu_weight_swap"
    assert backward_calls, "run_training must call backward"
    assert min(call.lineno for call in cpu_weight_swap_calls) > min(
        call.lineno for call in backward_calls
    )
