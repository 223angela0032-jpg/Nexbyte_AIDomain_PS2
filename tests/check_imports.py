"""Static check: every `from <project module> import name` must resolve, and no unused imports."""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
modules = {}
for path in ROOT.rglob("*.py"):
    if "tests" in path.parts:
        continue
    rel = path.relative_to(ROOT).with_suffix("")
    name = ".".join(rel.parts)
    if name.endswith(".__init__"):
        name = name[: -len(".__init__")]
    modules[name] = path

def top_level_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        names.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
    return names

trees = {n: ast.parse(p.read_text()) for n, p in modules.items()}
names = {n: top_level_names(t) for n, t in trees.items()}
problems = []

for mod, tree in trees.items():
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in modules:
            for alias in node.names:
                if alias.name != "*" and alias.name not in names[node.module] \
                        and f"{node.module}.{alias.name}" not in modules:
                    problems.append(f"{mod}: cannot import '{alias.name}' from '{node.module}'")
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name
                if top.split(".")[0] in {m.split(".")[0] for m in modules} and top not in modules:
                    problems.append(f"{mod}: unknown module '{top}'")

    # unused imports
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imported[(a.asname or a.name).split(".")[0]] = node.lineno
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                imported[a.asname or a.name] = node.lineno
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
           {n.value.id for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    for name, line in imported.items():
        if name not in used and not mod.endswith("__init__"):
            problems.append(f"{mod}:{line}: unused import '{name}'")

print("\n".join(problems) or "all project imports resolve; no unused imports")
sys.exit(1 if problems and any("cannot import" in p or "unknown module" in p for p in problems) else 0)
