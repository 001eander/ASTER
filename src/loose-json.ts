export function parseLooseJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch (first) {
    const repaired = escapeRawControlCharsInStrings(text);
    if (repaired === text) throw first;
    return JSON.parse(repaired);
  }
}

// Models sometimes paste multi-line prose into a JSON string. The raw newline
// makes JSON.parse throw, so escape control characters that appear inside
// strings while leaving already-escaped sequences untouched.
export function escapeRawControlCharsInStrings(text: string): string {
  let out = "";
  let inString = false;
  let escaped = false;
  for (const ch of text) {
    if (!inString) {
      if (ch === '"') inString = true;
      out += ch;
      continue;
    }
    if (escaped) {
      out += ch;
      escaped = false;
      continue;
    }
    if (ch === "\\") {
      out += ch;
      escaped = true;
      continue;
    }
    if (ch === '"') {
      out += ch;
      inString = false;
      continue;
    }
    const code = ch.codePointAt(0) ?? 0;
    if (code < 0x20) {
      if (ch === "\n") out += "\\n";
      else if (ch === "\r") out += "\\r";
      else if (ch === "\t") out += "\\t";
      else out += `\\u${code.toString(16).padStart(4, "0")}`;
      continue;
    }
    out += ch;
  }
  return out;
}
