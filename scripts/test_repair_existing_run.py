from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from c2rust_translator.artifacts import default_run_root, project_root, resolve_project_root
from c2rust_translator.rbench_interface import InterfaceIndex
from c2rust_translator.rust_skeleton_generator import Skeleton
from c2rust_translator.translation_runner import (
    _resolve_rules_path,
    build_arg_parser,
    discover_available_projects,
    load_env_file,
    project_config,
    resolve_run_id,
)
from c2rust_translator.translator import Translator


def build_parser() -> argparse.ArgumentParser:
    base = build_arg_parser()
    parser = argparse.ArgumentParser(
        description="Copy project outputs from an existing run and rerun only project-level test repair."
    )
    parser.add_argument("--source-run", required=True, help="Existing run id under output/runs/ to copy project crates from; legacy output/<run_id> is also supported.")
    for action in base._actions:
        if action.dest == "help":
            continue
        option_strings = list(action.option_strings)
        kwargs = {
            "dest": action.dest,
            "default": action.default,
            "help": action.help,
        }
        if action.nargs is not None:
            kwargs["nargs"] = action.nargs
        if action.choices is not None:
            kwargs["choices"] = action.choices
        if action.type is not None:
            kwargs["type"] = action.type
        if action.const is not None:
            kwargs["const"] = action.const
        if isinstance(action, argparse._StoreTrueAction):
            parser.add_argument(*option_strings, action="store_true", default=action.default, help=action.help)
        elif isinstance(action, argparse._AppendAction):
            parser.add_argument(*option_strings, action="append", default=action.default, help=action.help)
        elif option_strings:
            parser.add_argument(*option_strings, **kwargs)
        else:
            kwargs.pop("dest", None)
            parser.add_argument(action.dest, **kwargs)
    parser.set_defaults(alignment_mode="rule")
    return parser


def split_project_csv(project_csv: str | None) -> list[str]:
    if not project_csv:
        return []
    return [item.strip() for item in project_csv.split(",") if item.strip()]


def selected_projects(args: argparse.Namespace, available: list[str]) -> list[str]:
    project_options = getattr(args, "project_options", None)
    if project_options is None:
        project_options = getattr(args, "project", [])
    project_csv = getattr(args, "project_csv", None)
    requested = [*args.projects, *project_options, *split_project_csv(project_csv)]
    seen: set[str] = set()
    out: list[str] = []
    for project in requested:
        if project and project not in seen:
            seen.add(project)
            out.append(project)
    if out:
        return out
    return available[: max(args.limit, 0)]


def copy_project_output(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)

    def ignore(_dir: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            path = Path(_dir) / name
            if (
                name == "target"
                or name.endswith(".gcda")
                or name.endswith(".gcno")
                or name in {"translation_report.json", "test_output.txt", "test_fix_history.json"}
                or path.is_fifo()
                or path.is_socket()
                or (path.is_file() and not os.access(path, os.R_OK))
            ):
                ignored.add(name)
        return ignored

    shutil.copytree(source, destination, ignore=ignore)


def prepare_translator(root_dir: Path, run_root: Path, project: str, args: argparse.Namespace) -> tuple[Translator, list]:
    config = project_config(root_dir, run_root, project, args)
    config["clean_output_dir"] = False
    config["rules_path"] = _resolve_rules_path(root_dir, getattr(args, "rules_path", ""))
    translator = Translator(config)
    translator.skeleton = Skeleton(str(translator.rbench_path), str(translator.output_dir), clean=False)
    translator.skeleton._index_modules()

    interface_index = InterfaceIndex(str(translator.rbench_path)).build()
    public_callables = interface_index.list_public()
    callables = interface_index.list_unimplemented()
    c_funcs = translator._build_c_function_index()
    ordered, edge_count, cycles = translator._plan(callables, c_funcs)
    translator._prepare_test_repair_callables(public_callables, c_funcs)
    translator.dep_analysis = {
        "total_modules": len(ordered),
        "total_edges": edge_count,
        "circular_dependencies": cycles,
    }
    translator.translated_modules = {translator._callable_id(item) for item in ordered}
    translator.translation_log = [
        {
            "module": item.module,
            "kind": item.kind,
            "type_name": item.type_name,
            "name": item.name,
            "success": True,
        }
        for item in ordered
    ]
    return translator, ordered


def run_project(root_dir: Path, source_run: str, run_root: Path, project: str, args: argparse.Namespace) -> None:
    source = resolve_project_root(root_dir, source_run, project)
    destination = project_root(run_root, project)
    if not source.exists():
        raise FileNotFoundError(f"Missing source project output: {source}")

    print(f"Start test-repair-only project: {project}")
    print(f"Source: {source}")
    print(f"Output: {destination}")
    started = time.time()
    copy_project_output(source, destination)
    translator, ordered = prepare_translator(root_dir, run_root, project, args)
    tests_ok, output, stats = translator._run_test_repair_loop()
    ok = tests_ok and len(translator.translated_modules) == len(ordered)
    translator._save_report(ok)
    print(
        f"Finished test-repair-only project: {project} "
        f"success={ok} passed={stats.get('passed', 0)} failed={stats.get('failed', 0)} "
        f"elapsed={time.time() - started:.2f}s"
    )


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    load_env_file(root_dir / ".env")
    parser = build_parser()
    args = parser.parse_args()
    available = discover_available_projects(root_dir)
    projects = selected_projects(args, available)
    unknown = [project for project in projects if project not in set(available)]
    if unknown:
        parser.error(f"Unknown project(s): {', '.join(unknown)}")
    if not projects:
        parser.error("No projects selected.")

    run_id = resolve_run_id(args.run_id)
    run_root = default_run_root(root_dir, run_id)
    print(f"Selected {len(projects)} project(s): {', '.join(projects)}")
    print(f"Source run: {args.source_run}")
    print(f"Run id: {run_id}")
    print(f"Run root: {run_root}")
    print(f"Project jobs: {max(args.jobs, 1)}")
    print(f"Alignment mode: {args.alignment_mode}")
    print()

    errors: list[str] = []
    jobs = max(args.jobs, 1)
    with ThreadPoolExecutor(max_workers=min(jobs, len(projects))) as executor:
        futures = {
            executor.submit(run_project, root_dir, args.source_run, run_root, project, args): project
            for project in projects
        }
        for future in as_completed(futures):
            project = futures[future]
            try:
                future.result()
            except Exception as exc:
                errors.append(f"{project}: {exc}")
                print(f"Project crashed: {project}: {exc}")

    if errors:
        raise RuntimeError("One or more projects crashed: " + "; ".join(errors))


if __name__ == "__main__":
    main()
