#!/usr/bin/env python3
"""
audit_recovery.py — what survived the Lexar failure, and what is gone for good?

    python3 audit_recovery.py "/Volumes/Elements/PlantLeaf-copy/Audio Experiments"
    python3 audit_recovery.py --deep "/Volumes/.../Audio Experiments"   # + read every byte

Answers, in order of how much they cost to be wrong about:

  1. INTEGRITY — is each recovered .paudio actually readable? The default pass
     checks the header and the footer chain. `--deep` additionally streams every
     byte, which is the only way to surface the bad sectors a failing disk leaves
     behind: a file can have a perfect header and still throw EIO in the middle.
  2. COVERAGE — which recordings are missing? The expected set is derived from
     the labelled corpus on the internal disk plus the 30 sessions the deployed
     v6 model was fitted on. Missing-and-labelled is the expensive kind.
  3. WHAT IS RECOVERABLE — features, candidates, screenshots and predictions all
     come back by re-running the export. LABELS do not: they are human judgements
     that lived in the candidate CSVs, and only the newest recordings carry them
     inside the .paudio itself (the EVTR footer).

WHAT IS *NOT* CORRUPTION
    A body that is not a whole number of frames. audio_save_worker appends the
    body bin-by-bin (5 bytes at a time) across repeated saves, so a recording
    stopped mid-frame leaves a partial one. It is reported as a note, never as
    damage — calling it truncation would condemn ~30 % of a healthy corpus.
"""
from __future__ import annotations
import argparse, csv, json, struct, sys, zlib
from collections import defaultdict
from pathlib import Path

HEADER_SIZE, FRAME_BYTES = 128, 770
MARKERS   = (b'CLCK', b'EVNT', b'EVTR')
TAIL_SCAN = 128 << 20
CHUNK     = 8 << 20


def footer_chain(path: Path, size: int):
    """(body_end, {marker: (payload_off, len)}) by walking the chain BACKWARDS.

    Footers are appended, contiguous, and the last ends exactly at EOF. So a
    block is real only if its declared length lands precisely on the end of the
    next one — a chance b'EVTR' inside body data will not satisfy that. Only the
    tail is scanned: these recordings reach 2 GB and the footers are never far
    from the end.
    """
    span, out = min(size, TAIL_SCAN), {}
    base = size - span
    with open(path, 'rb') as f:
        f.seek(base)
        tail = f.read(span)
    by_end = {}
    for mk in MARKERS:
        pos = tail.find(mk)
        while pos >= 0:
            if pos + 8 <= len(tail):
                (ln,) = struct.unpack_from('<I', tail, pos + 4)
                end = base + pos + 8 + ln
                if end <= size:
                    by_end[end] = (mk, base + pos, base + pos + 8, ln)
            pos = tail.find(mk, pos + 1)
    cursor = size
    while cursor in by_end:
        mk, start, poff, ln = by_end[cursor]
        if mk in out:
            break
        out[mk] = (poff, ln)
        cursor = start
    return (cursor if out else size), out


def probe(path: Path, deep: bool = False) -> dict:
    r = {'path': path, 'size': 0, 'ok': False, 'why': '', 'note': '',
         'duration_s': 0.0, 'version': None, 'footers': '-',
         'n_rows': 0, 'n_labels': 0, 'n_clicks': 0, 'n_ambig': 0}
    try:
        size = r['size'] = path.stat().st_size
        with open(path, 'rb') as f:
            head = f.read(HEADER_SIZE)
    except OSError as e:
        r['why'] = f'unreadable ({e.strerror})'
        return r

    if len(head) < 78:
        r['why'] = f'only {len(head)} header bytes — stub, not a recording'
        return r
    if head[0:10].rstrip(b'\x00') != b'PLANTAUDIO':
        r['why'] = 'bad magic — header destroyed or not a .paudio'
        return r
    (ver,) = struct.unpack_from('<f', head, 10)
    fs   = struct.unpack_from('<I', head, 34)[0]
    nfft = struct.unpack_from('<I', head, 38)[0]
    r['version'] = round(ver, 1)
    if fs == 0 or nfft == 0:
        r['why'] = f'zero fs/fft_size (fs={fs}, fft={nfft}) — header corrupt'
        return r

    body_end, chain = footer_chain(path, size)
    r['footers'] = ','.join(m.decode() for m in MARKERS if m in chain) or '-'
    body = body_end - HEADER_SIZE
    if body <= 0:
        r['why'] = 'header only, no frame data'
        return r
    n_frames, rem = divmod(body, FRAME_BYTES)
    r['duration_s'] = n_frames * nfft / fs
    if rem:
        r['note'] = f'partial final frame ({rem}/{FRAME_BYTES} B) — normal'

    if b'EVTR' in chain:
        poff, ln = chain[b'EVTR']
        try:
            with open(path, 'rb') as f:
                f.seek(poff)
                rows = json.loads(zlib.decompress(f.read(ln)).decode('utf-8'))
            if isinstance(rows, list):
                r['n_rows'] = len(rows)
                for row in rows:
                    v = str(row.get('label', '')).strip()
                    if v in ('', 'None', 'nan'):
                        continue
                    r['n_labels'] += 1
                    r['n_clicks'] += v in ('1', '1.0')
                    r['n_ambig']  += v in ('2', '2.0')
        except Exception as e:                                  # noqa: BLE001
            r['why'] = f'EVTR present but undecodable ({type(e).__name__})'
            return r

    if deep:
        try:
            read = 0
            with open(path, 'rb') as f:
                while True:
                    b = f.read(CHUNK)
                    if not b:
                        break
                    read += len(b)
            if read != size:
                r['why'] = f'short read: {read:,} of {size:,} bytes — bad sectors'
                return r
        except OSError as e:
            r['why'] = f'I/O error at ~{read:,} B ({e.strerror}) — bad sectors'
            return r
    r['ok'] = True
    return r


def expected():
    """(everything we have a record of, the 30 model sessions, {stem: labels})."""
    home = Path.home() / 'PlantLeaf_dev'
    known, labels = set(), defaultdict(int)
    for p in home.rglob('*_candidates.csv'):
        if p.name.startswith('._'):
            continue
        known.add(p.name.replace('_candidates.csv', ''))
        try:
            rows = list(csv.DictReader(open(p, encoding='utf-8-sig')))
        except OSError:
            continue
        stem = p.name.replace('_candidates.csv', '')
        n = sum(1 for r in rows if (r.get('label') or '').strip())
        if n:
            labels[stem] += n
        for r in rows:
            if r.get('file'):
                known.add(r['file'].strip())
    sessions = set()
    try:
        import joblib
        m = joblib.load(home / 'PlantLeaf-Desktop-App/src/ml/v6/plantleaf_svm_v6_DEPLOYED.pkl')
        sessions = set(m['sessions_train']) | set(m['sessions_test'])
    except Exception as e:                                      # noqa: BLE001
        print(f'  (deployed model unreadable: {e})')
    return known | sessions, sessions, dict(labels)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('roots', nargs='+', type=Path)
    ap.add_argument('--deep', action='store_true',
                    help='read every byte of every file — surfaces bad sectors. Slow.')
    ap.add_argument('--csv', type=Path, default=None, help='write the per-file table here')
    a = ap.parse_args()

    files = []
    for r in a.roots:
        try:
            if not r.exists():
                print(f'ERROR: {r} does not exist'); return 2
            files += [p for p in r.rglob('*.paudio') if not p.name.startswith('._')]
        except OSError as e:
            print(f'ERROR: cannot read {r} ({e.strerror}).')
            print('  If this is "Operation not permitted", grant Full Disk Access to')
            print('  your terminal in System Settings > Privacy & Security.')
            return 2

    print(f'\n{len(files)} .paudio file(s) under {len(a.roots)} root(s)'
          + ('   [deep read]' if a.deep else ''))
    probes = []
    for i, p in enumerate(sorted(files), 1):
        if a.deep:
            print(f'\r  reading {i}/{len(files)} ...', end='', flush=True)
        probes.append(probe(p, a.deep))
    if a.deep:
        print('\r' + ' ' * 40 + '\r', end='')

    good = [x for x in probes if x['ok']]
    bad  = [x for x in probes if not x['ok']]
    hours = sum(x['duration_s'] for x in good) / 3600

    print('\n1. INTEGRITY')
    print(f'   readable        : {len(good)}')
    print(f'   DAMAGED         : {len(bad)}')
    print(f'   audio recovered : {hours:.1f} h')
    if not a.deep:
        print('   (header+footer only — re-run with --deep to catch bad sectors)')
    for x in bad:
        print(f'     ✗ {x["why"]:<56} {x["path"].name}')
    partial = [x for x in good if x['note']]
    if partial:
        print(f'   {len(partial)} file(s) end on a partial frame — normal, not damage')

    print('\n2. BY FOLDER')
    by_dir = defaultdict(lambda: [0, 0.0])
    for x in good:
        d = x['path'].parent
        by_dir[d][0] += 1
        by_dir[d][1] += x['duration_s']
    for d in sorted(by_dir, key=str):
        try:
            rel = d.relative_to(a.roots[0])
        except ValueError:
            rel = d
        n, secs = by_dir[d]
        print(f'   {n:4}  {secs/3600:7.1f} h  {rel}')

    have = {x['path'].stem for x in good}
    known, sessions, labels = expected()
    missing = sorted(known - have)
    extra   = sorted(have - known)

    print('\n3. COVERAGE')
    print(f'   on the drive                  : {len(have)}')
    print(f'   we have records of            : {len(known)}')
    print(f'   present                       : {len(known & have)}')
    print(f'   MISSING                       : {len(missing)}')
    print(f'   on the drive but unrecorded   : {len(extra)}')

    ms = sorted(s for s in missing if s in sessions)
    ml = sorted(((s, labels[s]) for s in missing if s in labels), key=lambda x: -x[1])
    if ms:
        print(f'\n   ⚠️  {len(ms)}/30 deployed-model session(s) have NO .paudio:')
        for s in ms:
            print(f'        {s}')
    if ml:
        print(f'\n   ⚠️  {len(ml)} missing recording(s) carry labelling effort:')
        for s, n in ml:
            print(f'        {n:5} labelled rows   {s}')
    rest = [s for s in missing if s not in labels and s not in sessions]
    if rest:
        print(f'\n   {len(rest)} other missing, unlabelled recording(s):')
        for s in rest[:30]:
            print(f'        {s}')
        if len(rest) > 30:
            print(f'        ... and {len(rest)-30} more')

    with_lab = [x for x in good if x['n_labels']]
    print('\n4. LABELS INSIDE THE AUDIO (EVTR footer)')
    if with_lab:
        tot = sum(x['n_labels'] for x in with_lab)
        print(f'   {len(with_lab)} file(s) carry {tot} label(s) in their EVTR footer —')
        print('   these ARE recoverable from the audio alone:')
        for x in sorted(with_lab, key=lambda y: -y['n_labels']):
            print(f'        {x["n_labels"]:5} labels ({x["n_clicks"]} clicks, '
                  f'{x["n_ambig"]} ambiguous)  {x["path"].name}')
    else:
        print('   None. The EVTR footer post-dates this corpus, so no label in any')
        print('   of these files can be recovered from the audio — they live only in')
        print('   the candidate CSVs.')

    print('\n5. WHAT THIS MEANS')
    print(f'   REGENERABLE from the {len(good)} readable file(s): every candidate CSV,')
    print('     every v5/v6 feature, screenshot and prediction. Re-run the Data')
    print('     Collection export — none of it was unique to the Lexar.')
    n_rows = sum(labels.values())
    print(f'   SAFE on the internal disk: {n_rows:,} labelled rows over '
          f'{len(labels)} recording(s).')
    print('   AT RISK: any labelling done only on the Lexar. Labels cannot be')
    print('     recomputed from audio. Compare the v6 exhaustive pass against what')
    print('     the internal disk holds before assuming it survived.')

    if a.csv:
        with open(a.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['file', 'path', 'size_bytes', 'ok', 'problem', 'note',
                        'duration_s', 'version', 'footers', 'evtr_rows', 'evtr_labels'])
            for x in sorted(probes, key=lambda y: str(y['path'])):
                w.writerow([x['path'].stem, x['path'], x['size'], int(x['ok']),
                            x['why'], x['note'], round(x['duration_s'], 3),
                            x['version'], x['footers'], x['n_rows'], x['n_labels']])
        print(f'\n   per-file table -> {a.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
