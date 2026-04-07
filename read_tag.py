from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: PyYAML. Install it with `pip install pyyaml`."
    ) from exc


FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)


def ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def load_front_matter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = FRONT_MATTER_RE.match(text)
    if not match:
        raise ValueError(f"Missing front matter: {path}")
    return yaml.safe_load(match.group(1)) or {}


def collect_posts(root_dir: Path) -> tuple[Counter, Counter, dict[str, list[str]], list[str]]:
    tag_counter: Counter = Counter()
    category_counter: Counter = Counter()
    tag_to_files: dict[str, list[str]] = defaultdict(list)
    invalid_files: list[str] = []

    for path in sorted(root_dir.rglob("*.md")):
        try:
            data = load_front_matter(path)
        except ValueError:
            invalid_files.append(str(path))
            continue

        relative_path = str(path.relative_to(root_dir.parent))

        for tag in ensure_list(data.get("tags")):
            tag_counter[tag] += 1
            tag_to_files[tag].append(relative_path)

        for category in ensure_list(data.get("categories")):
            category_counter[category] += 1

    return tag_counter, category_counter, tag_to_files, invalid_files


def print_counter(title: str, counter: Counter, limit: int | None = None) -> None:
    print(title)
    items = counter.most_common(limit)
    for key, count in items:
        print(f"  {count:>3}  {key}")
    print()


def print_possible_collisions(tag_to_files: dict[str, list[str]]) -> None:
    normalized = defaultdict(list)
    for tag in tag_to_files:
        normalized[tag.casefold()].append(tag)

    collisions = {
        key: sorted(set(values))
        for key, values in normalized.items()
        if len(set(values)) > 1
    }

    print("Possible style collisions")
    if not collisions:
        print("  (none)")
        print()
        return

    for _, variants in sorted(collisions.items(), key=lambda item: item[0]):
        print(f"  {' | '.join(variants)}")
    print()


def print_single_use_tags(tag_counter: Counter, limit: int) -> None:
    single_use_tags = sorted(tag for tag, count in tag_counter.items() if count == 1)
    print(f"Single-use tags ({len(single_use_tags)})")
    for tag in single_use_tags[:limit]:
        print(f"  {tag}")
    if len(single_use_tags) > limit:
        print(f"  ... and {len(single_use_tags) - limit} more")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize tags and categories from Hexo posts."
    )
    parser.add_argument(
        "--root",
        default="_posts",
        help="Posts root directory. Defaults to _posts.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many top tags/categories to print. Defaults to 15.",
    )
    parser.add_argument(
        "--single-use-limit",
        type=int,
        default=30,
        help="How many single-use tags to print. Defaults to 30.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root)
    if not root_dir.exists():
        print(f"Directory not found: {root_dir}", file=sys.stderr)
        return 1

    tag_counter, category_counter, tag_to_files, invalid_files = collect_posts(root_dir)
    total_posts = len(list(root_dir.rglob("*.md")))
    total_tag_assignments = sum(tag_counter.values())
    single_use_count = sum(1 for count in tag_counter.values() if count == 1)

    print("Hexo tag summary")
    print(f"  Posts: {total_posts}")
    print(f"  Unique tags: {len(tag_counter)}")
    print(f"  Tag assignments: {total_tag_assignments}")
    print(f"  Unique categories: {len(category_counter)}")
    print(f"  Single-use tags: {single_use_count}")
    print(f"  Invalid front matter files: {len(invalid_files)}")
    print()

    print_counter("Top tags", tag_counter, args.top)
    print_counter("Top categories", category_counter, args.top)
    print_possible_collisions(tag_to_files)
    print_single_use_tags(tag_counter, args.single_use_limit)

    if invalid_files:
        print("Files with invalid front matter")
        for path in invalid_files:
            print(f"  {path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
