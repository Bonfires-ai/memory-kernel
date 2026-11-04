"""Request-local CL-JSON string adaptation; no global methods or source edits.

Ported from a reference-harness module (unreached top-level definitions pruned).
"""


def repair_cl_json_astral_known(value, allowed):
    """Source-verified repair of CL-JSON's over-width ``\\uXXXXX`` escapes (transport utf8-v4).

    CL-JSON writes a non-BMP character as ``\\u`` plus 5-6 hex digits, so ``\\u1F31F`` may be the star
    U+1F31F or BMP U+1F31 followed by a literal "F". The ambiguity is resolved against ``allowed`` —
    the non-BMP characters present in the request's own source text: a 6- or 5-digit reading is used
    only when it names an allowed character; otherwise the escape is the ordinary 4-digit BMP escape and
    the following characters are left literal. The caller still checks every returned string against
    the source, so a wrong reading cannot pass silently.
    """
    allowed = {ord(c) for c in allowed}
    out, index, in_string = [], 0, False
    while index < len(value):
        char = value[index]
        if in_string and char == "\\":
            if value.startswith("\\u", index):
                end = index + 2
                while end < len(value) and end < index + 8 and value[end] in "0123456789abcdefABCDEF":
                    end += 1
                digits = value[index + 2 : end]
                for width in (6, 5):
                    if len(digits) >= width and int(digits[:width], 16) in allowed:
                        point = int(digits[:width], 16) - 0x10000
                        out.append(f"\\u{0xD800 + (point >> 10):04X}\\u{0xDC00 + (point & 0x3FF):04X}")
                        index += 2 + width
                        break
                else:
                    if len(digits) < 4:
                        raise ValueError("truncated JSON unicode escape")
                    out.append(value[index : index + 6])
                    index += 6
                continue
            out.append(value[index : index + 2])
            index += 2
            continue
        if char == '"':
            in_string = not in_string
        out.append(char)
        index += 1
    return "".join(out)
