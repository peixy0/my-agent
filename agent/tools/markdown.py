import re


def parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body). If no frontmatter is found returns
    ({}, original_content).  Only simple 'key: value' pairs are supported;
    quoting is stripped from values.
    """
    match = re.match(
        r"^---[ \t]*\n(.*?)^---[ \t]*\n(.*)",
        content,
        re.DOTALL | re.MULTILINE,
    )
    if not match:
        return {}, content

    fm_text = match.group(1)
    body = match.group(2).strip()

    fm: dict[str, str] = {}
    for line in fm_text.splitlines():
        m = re.match(r"^(\w+)\s*:\s*(.*?)\s*$", line)
        if m:
            fm[m.group(1)] = m.group(2).strip('"').strip("'")
    return fm, body
