"""SVG minifier for matplotlib output (keeps SVGs small enough to commit as text)."""
import re
import xml.etree.ElementTree as ET

_SVG = "http://www.w3.org/2000/svg"
_XL = "http://www.w3.org/1999/xlink"
ET.register_namespace("", _SVG)
ET.register_namespace("xlink", _XL)


def _r(s: str) -> str:
    return re.sub(r"-?\d+\.\d+", lambda n: (f"{float(n.group(0)):.1f}".rstrip("0").rstrip(".") or "0"), s)


def minify_svg(svg: str) -> str:
    """Shrink matplotlib SVG: drop metadata/clip paths/unused ids, round coords, flatten bare groups."""
    root = ET.fromstring(svg.encode("utf-8"))
    used = set(re.findall(r'href="#([^"]+)"', svg))

    def clean(el):
        new_children = []
        for ch in list(el):
            tag = ch.tag.split("}")[-1]
            if tag in ("metadata", "clipPath"):
                continue
            clean(ch)
            if tag == "defs" and len(ch) == 0 and not ch.attrib:
                continue
            if tag == "g" and not ch.attrib:
                new_children.extend(list(ch))
            else:
                new_children.append(ch)
        for ch in list(el):
            el.remove(ch)
        el.extend(new_children)
        a = el.attrib
        a.pop("clip-path", None)
        if "id" in a and a["id"] not in used:
            del a["id"]
        for k in ("d", "x", "y", "transform", "points", "x1", "x2", "y1", "y2", "width", "height", "cx", "cy", "r"):
            if k in a and el.tag.split("}")[-1] != "svg":
                a[k] = _r(a[k])
        if "d" in a:
            a["d"] = " ".join(a["d"].split())
        if a.get("transform", "").startswith("rotate(-0 "):
            del a["transform"]
        if "style" in a:
            a["style"] = re.sub(r"font-family: [^;]*sans-serif", "font-family: sans-serif", a["style"])
        if el.text and not el.text.strip():
            el.text = None
        if el.tail and not el.tail.strip():
            el.tail = None

    clean(root)
    out = ET.tostring(root, encoding="unicode").replace(" />", "/>").replace("><", ">\n<")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + out + "\n"
