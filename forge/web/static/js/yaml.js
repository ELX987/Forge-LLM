// Minimal YAML dump/parse sufficient for Forge configs and recipes (the JSON-compatible subset plus
// comments, block/flow collections, quoted strings and | / > block scalars). Not a full YAML 1.2 parser:
// anchors, tags, multi-document streams and complex keys are intentionally unsupported.

export class YamlError extends Error {
  constructor(msg, line) { super(line ? `YAML line ${line}: ${msg}` : msg); this.name = 'YamlError'; this.line = line || 0; }
}

// ------------------------------------------------------------------------------------------ dump
const RESERVED = /^(true|false|null|yes|no|on|off|y|n|~)$/i;
const NUMLIKE = /^[-+]?(\d[\d_]*\.?\d*([eE][-+]?\d+)?|\.\d+([eE][-+]?\d+)?|0x[0-9a-fA-F]+|\.inf|\.nan)$/;
const PLAIN_OK = /^[A-Za-z_./][A-Za-z0-9_./\-+@]*( [A-Za-z0-9_./\-+@]+)*$/;

function dumpString(s) {
  if (s === '') return '""';
  if (PLAIN_OK.test(s) && !RESERVED.test(s) && !NUMLIKE.test(s) && !s.includes(': ') && !s.includes(' #')) return s;
  return JSON.stringify(s);
}
function dumpKey(k) { return dumpString(String(k)); }
function dumpScalar(v) {
  if (v === null || v === undefined) return 'null';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : (Number.isNaN(v) ? '.nan' : (v > 0 ? '.inf' : '-.inf'));
  return dumpString(String(v));
}
const isScalar = (v) => v === null || v === undefined || typeof v !== 'object';

function flowItem(v) {
  if (isScalar(v)) return dumpScalar(v);
  if (Array.isArray(v)) return '[' + v.map(flowItem).join(', ') + ']';
  return '{' + Object.entries(v).map(([k, x]) => `${dumpKey(k)}: ${flowItem(x)}`).join(', ') + '}';
}

function dumpNode(v, indent, lines, prefix) {
  const pad = ' '.repeat(indent);
  if (Array.isArray(v)) {
    if (!v.length) { lines.push(prefix + '[]'); return; }
    const flat = v.every(isScalar) && v.length <= 10 && v.map(flowItem).join(', ').length <= 70;
    if (flat) { lines.push(prefix + '[' + v.map(flowItem).join(', ') + ']'); return; }
    if (prefix) lines.push(prefix.replace(/\s+$/, ''));
    for (const item of v) {
      if (isScalar(item)) lines.push(`${pad}- ${dumpScalar(item)}`);
      else if (Array.isArray(item)) {
        if (!item.length) lines.push(`${pad}- []`);
        else if (item.every(isScalar)) lines.push(`${pad}- ${flowItem(item)}`);
        else { lines.push(`${pad}-`); dumpNode(item, indent + 2, lines, ''); }
      } else {
        const keys = Object.keys(item);
        if (!keys.length) { lines.push(`${pad}- {}`); continue; }
        const sub = []; dumpNode(item, indent + 2, sub, '');
        sub[0] = `${pad}- ` + sub[0].slice(indent + 2);
        lines.push(...sub);
      }
    }
    return;
  }
  if (v && typeof v === 'object') {
    const keys = Object.keys(v);
    if (!keys.length) { lines.push(prefix + '{}'); return; }
    if (prefix) lines.push(prefix.replace(/\s+$/, ''));
    for (const k of keys) {
      const x = v[k];
      if (x === undefined) continue;
      if (isScalar(x)) lines.push(`${pad}${dumpKey(k)}: ${dumpScalar(x)}`);
      else dumpNode(x, indent + 2, lines, `${pad}${dumpKey(k)}: `);
    }
    return;
  }
  lines.push(prefix + dumpScalar(v));
}

/** Serialise a JSON-compatible value to YAML text. */
export function dump(value) {
  if (isScalar(value)) return dumpScalar(value) + '\n';
  const lines = [];
  dumpNode(value, 0, lines, '');
  return lines.join('\n') + '\n';
}

// ------------------------------------------------------------------------------------------ parse
function stripComment(s) {
  let q = null;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === '\\' && q === '"') i++; else if (c === q) q = null; continue; }
    if (c === '"' || c === "'") { if (i === 0 || /[\s,\[{:]/.test(s[i - 1])) q = c; continue; }
    if (c === '#' && (i === 0 || /\s/.test(s[i - 1]))) return s.slice(0, i).replace(/\s+$/, '');
  }
  return s.replace(/\s+$/, '');
}

/** Split "key: rest" at the first top-level ': ' (or trailing ':'). Returns [key, rest] or null. */
function splitKey(text) {
  let i = 0; let key;
  if (text[0] === '"' || text[0] === "'") {
    const q = text[0]; i = 1;
    while (i < text.length) {
      if (text[i] === '\\' && q === '"') { i += 2; continue; }
      if (text[i] === q) { if (q === "'" && text[i + 1] === "'") { i += 2; continue; } break; }
      i++;
    }
    if (i >= text.length) return null;
    const after = text.slice(i + 1);
    if (!(after.startsWith(':') && (after.length === 1 || after[1] === ' '))) return null;
    key = parseScalar(text.slice(0, i + 1));
    return [String(key), after.slice(1).trim()];
  }
  if (text[0] === '[' || text[0] === '{') return null;
  for (i = 0; i < text.length; i++) {
    if (text[i] === ':' && (i === text.length - 1 || text[i + 1] === ' ')) {
      return [text.slice(0, i).trim(), text.slice(i + 1).trim()];
    }
  }
  return null;
}

function parseScalar(s) {
  s = s.trim();
  if (s === '' || s === '~' || /^null$/i.test(s)) return null;
  if (/^true$/i.test(s)) return true;
  if (/^false$/i.test(s)) return false;
  if (s[0] === '"') {
    try { return JSON.parse(s); } catch (_) {
      throw new YamlError(`bad double-quoted string: ${s}`);
    }
  }
  if (s[0] === "'") {
    if (s.length < 2 || s[s.length - 1] !== "'") throw new YamlError(`unterminated single-quoted string: ${s}`);
    return s.slice(1, -1).replace(/''/g, "'");
  }
  if (/^[-+]?\d[\d_]*$/.test(s)) return parseInt(s.replace(/_/g, ''), 10);
  if (/^[-+]?(\d[\d_]*\.\d*|\.\d+|\d[\d_]*)([eE][-+]?\d+)?$/.test(s)) return parseFloat(s.replace(/_/g, ''));
  if (/^0x[0-9a-fA-F]+$/.test(s)) return parseInt(s, 16);
  if (/^[-+]?\.inf$/i.test(s)) return s[0] === '-' ? -Infinity : Infinity;
  if (/^\.nan$/i.test(s)) return NaN;
  return s;
}

/** Parse a flow collection starting at s[pos]. Returns [value, nextPos]. */
function parseFlow(s, pos, lineNo) {
  const skip = () => { while (pos < s.length && /\s/.test(s[pos])) pos++; };
  const parseVal = () => {
    skip();
    if (s[pos] === '[') {
      pos++; const arr = [];
      for (;;) {
        skip();
        if (s[pos] === ']') { pos++; return arr; }
        if (pos >= s.length) throw new YamlError('unterminated [ ]', lineNo);
        arr.push(parseVal()); skip();
        if (s[pos] === ',') pos++; else if (s[pos] !== ']') throw new YamlError(`expected , or ] near "${s.slice(pos, pos + 12)}"`, lineNo);
      }
    }
    if (s[pos] === '{') {
      pos++; const obj = {};
      for (;;) {
        skip();
        if (s[pos] === '}') { pos++; return obj; }
        if (pos >= s.length) throw new YamlError('unterminated { }', lineNo);
        const k = parseKeyTok(); skip();
        if (s[pos] !== ':') throw new YamlError(`expected : after key "${k}"`, lineNo);
        pos++;
        skip();
        obj[k] = (s[pos] === ',' || s[pos] === '}') ? null : parseVal();
        skip();
        if (s[pos] === ',') pos++; else if (s[pos] !== '}') throw new YamlError('expected , or }', lineNo);
      }
    }
    return parseTok(false);
  };
  const quoted = () => {
    const q = s[pos]; let j = pos + 1;
    while (j < s.length) {
      if (s[j] === '\\' && q === '"') { j += 2; continue; }
      if (s[j] === q) { if (q === "'" && s[j + 1] === "'") { j += 2; continue; } break; }
      j++;
    }
    if (j >= s.length) throw new YamlError('unterminated string', lineNo);
    const tok = s.slice(pos, j + 1); pos = j + 1; return parseScalar(tok);
  };
  const parseTok = (isKey) => {
    skip();
    if (s[pos] === '"' || s[pos] === "'") return quoted();
    let j = pos;
    while (j < s.length) {
      const c = s[j];
      if (c === ',' || c === ']' || c === '}') break;
      if (isKey && c === ':') break;
      if (!isKey && c === ':' && (s[j + 1] === ' ')) break;
      j++;
    }
    const tok = s.slice(pos, j); pos = j;
    return isKey ? tok.trim() : parseScalar(tok);
  };
  const parseKeyTok = () => String(parseTok(true));
  const v = parseVal();
  return [v, pos];
}

/** Parse YAML text to a JS value. Throws YamlError. */
export function parse(text) {
  const raw = String(text).replace(/\r\n?/g, '\n').split('\n');
  const lines = []; // {indent, text, no, raw}
  raw.forEach((r, i) => {
    if (/^\s*$/.test(r)) { lines.push({ blank: true, no: i + 1, raw: r }); return; }
    if (/^\t/.test(r)) throw new YamlError('tabs are not allowed for indentation', i + 1);
    const indent = r.match(/^ */)[0].length;
    const body = r.slice(indent);
    if (indent === 0 && (body === '---' || body.startsWith('--- '))) { lines.push({ blank: true, no: i + 1, raw: '' }); return; }
    if (indent === 0 && body === '...') { lines.push({ blank: true, no: i + 1, raw: '' }); return; }
    const t = stripComment(body);
    if (t === '') { lines.push({ blank: true, no: i + 1, raw: r }); return; }
    lines.push({ indent, text: t, no: i + 1, raw: r });
  });
  let p = 0;
  const skipBlank = () => { while (p < lines.length && lines[p].blank) p++; };

  function blockScalar(header, parentIndent) {
    // header like | |- |+ > >- ; consumes following lines indented deeper than parentIndent
    const style = header[0]; const chomp = header.includes('-') ? 'strip' : (header.includes('+') ? 'keep' : 'clip');
    const body = []; let blockIndent = null;
    while (p < lines.length) {
      const L = lines[p];
      if (L.blank) { body.push(''); p++; continue; }
      if (L.indent <= parentIndent) break;
      if (blockIndent == null) blockIndent = L.indent;
      body.push(L.raw.slice(Math.min(blockIndent, L.indent)));
      p++;
    }
    while (body.length && body[body.length - 1] === '') { if (chomp === 'keep') break; body.pop(); }
    let out;
    if (style === '|') out = body.join('\n');
    else {
      out = ''; // folded: join non-empty lines with spaces, blank lines -> newline
      for (let i = 0; i < body.length; i++) {
        if (body[i] === '') out += '\n';
        else out += (out && !out.endsWith('\n') ? ' ' : '') + body[i];
      }
    }
    return chomp === 'strip' ? out : out + '\n';
  }

  function inlineValue(rest, L, parentIndent) {
    if (rest === '') return undefined;
    if (rest[0] === '|' || rest[0] === '>') return blockScalar(rest, parentIndent);
    if (rest[0] === '[' || rest[0] === '{') {
      let s = rest;
      const balanced = (x) => {
        let d = 0; let q = null;
        for (let i = 0; i < x.length; i++) {
          const c = x[i];
          if (q) { if (c === '\\' && q === '"') i++; else if (c === q) q = null; continue; }
          if (c === '"' || c === "'") q = c; else if (c === '[' || c === '{') d++; else if (c === ']' || c === '}') d--;
        }
        return d <= 0;
      };
      while (!balanced(s) && p < lines.length) {
        const N = lines[p++];
        if (!N.blank) s += ' ' + N.text.trim();
      }
      const [v, end] = parseFlow(s, 0, L.no);
      if (s.slice(end).trim() !== '') throw new YamlError(`unexpected text after flow collection: ${s.slice(end).trim()}`, L.no);
      return v;
    }
    return parseScalar(rest);
  }

  function parseBlock(indent) {
    skipBlank();
    const L = lines[p];
    if (!L) return null;
    if (L.text === '-' || L.text.startsWith('- ')) return parseSeq(L.indent);
    if (splitKey(L.text)) return parseMap(L.indent);
    p++;
    return inlineValue(L.text, L, indent - 1);
  }

  function parseMap(indent) {
    const obj = {};
    for (;;) {
      skipBlank();
      const L = lines[p];
      if (!L || L.indent < indent) break;
      if (L.indent > indent) throw new YamlError('bad indentation', L.no);
      if (L.text === '-' || L.text.startsWith('- ')) break;
      const kv = splitKey(L.text);
      if (!kv) throw new YamlError(`expected "key: value" but got "${L.text}"`, L.no);
      const [key, rest] = kv;
      p++;
      if (Object.prototype.hasOwnProperty.call(obj, key)) throw new YamlError(`duplicate key "${key}"`, L.no);
      let val = inlineValue(rest, L, indent);
      if (val === undefined) {
        skipBlank();
        const N = lines[p];
        if (N && (N.indent > indent || (N.indent === indent && (N.text === '-' || N.text.startsWith('- '))))) val = parseBlock(N.indent);
        else val = null;
      }
      obj[key] = val;
    }
    return obj;
  }

  function parseSeq(indent) {
    const arr = [];
    for (;;) {
      skipBlank();
      const L = lines[p];
      if (!L || L.indent < indent) break;
      if (L.indent > indent) throw new YamlError('bad indentation in list', L.no);
      if (!(L.text === '-' || L.text.startsWith('- '))) break;
      const rest = L.text === '-' ? '' : L.text.slice(2).replace(/^\s+/, '');
      if (rest === '') {
        p++; skipBlank();
        const N = lines[p];
        arr.push(N && N.indent > indent ? parseBlock(N.indent) : null);
      } else if (rest === '-' || rest.startsWith('- ') || splitKey(rest)) {
        const off = L.text.length - rest.length;
        lines[p] = { ...L, indent: indent + off, text: rest, raw: ' '.repeat(indent + off) + rest };
        arr.push(parseBlock(indent + off));
      } else {
        p++;
        arr.push(inlineValue(rest, L, indent));
      }
    }
    return arr;
  }

  skipBlank();
  if (p >= lines.length) return null;
  const first = lines[p];
  const v = parseBlock(first.indent);
  skipBlank();
  if (p < lines.length) throw new YamlError(`unexpected content "${lines[p].text}"`, lines[p].no);
  return v;
}

/** Parse JSON or YAML (tries JSON first when the text starts with { or [). */
export function parseAny(text) {
  const t = String(text).trim();
  if (t.startsWith('{') || t.startsWith('[')) {
    try { return JSON.parse(t); } catch (_) { /* maybe flow YAML */ }
  }
  return parse(text);
}
