#!/usr/bin/env python3
"""Confirmed-cut timeline, caption mapping and lossless PCM editing (60 fps).

This tool does not decide what was spoken. Cuts refer to ORIGINAL narration.
Each cut requires start_frame/end_frame/reason/evidence. Optional
ramp_safe={"start":true,"end":true} explicitly confirms word-free retained
audio immediately before/after that cut. Missing flags mean NO ramp.
No insertion support: apply later project edits through a separate remap.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import wave

FPS = 60
RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2
SAMPLES_PER_FRAME = 800
BYTES_PER_SAMPLE_FRAME = 4


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}.')
    return value


def _evidence(value):
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return bool(value) and any(_evidence(v) for v in value.values())
    if isinstance(value, list):
        return bool(value) and any(_evidence(v) for v in value)
    return False


def build_map(source_frames, cuts, fps=FPS):
    """Return an edit map from confirmed original-clock cuts; never sort them."""
    _integer(source_frames, 'source_frames', 1)
    if type(fps) is not int or fps != FPS:
        raise ValueError('This PCM-exact preset requires fps=60.')
    if not isinstance(cuts, list):
        raise ValueError('cuts must be a list (an empty list is valid).')
    normalized = []
    prior_end = 0
    for i, cut in enumerate(cuts):
        if not isinstance(cut, dict):
            raise ValueError(f'Cut {i}: expected an object.')
        a = _integer(cut.get('start_frame'), f'Cut {i} start_frame')
        b = _integer(cut.get('end_frame'), f'Cut {i} end_frame', 1)
        if not (a < b <= source_frames) or a < prior_end:
            raise ValueError(f'Cut {i}: unsorted, overlapping, empty or out-of-range interval.')
        reason = cut.get('reason')
        if not isinstance(reason, str) or not reason.strip() or not _evidence(cut.get('evidence')):
            raise ValueError(f'Cut {i}: nonempty reason and evidence are required.')
        safe = cut.get('ramp_safe', {})
        if not isinstance(safe, dict) or set(safe) - {'start', 'end'}:
            raise ValueError(f'Cut {i}: ramp_safe accepts only start/end boolean flags.')
        if any(type(v) is not bool for v in safe.values()):
            raise ValueError(f'Cut {i}: ramp_safe flags must be booleans.')
        normalized.append(dict(start_frame=a, end_frame=b, reason=reason,
                               evidence=cut['evidence'],
                               ramp_safe=dict(start=safe.get('start', False), end=safe.get('end', False))))
        prior_end = b
    keeps = []
    source_cursor = output_cursor = 0
    previous_cut = None
    for i in range(len(normalized) + 1):
        next_cut = normalized[i] if i < len(normalized) else None
        end = next_cut['start_frame'] if next_cut else source_frames
        if end > source_cursor:
            count = end - source_cursor
            keeps.append(dict(source_start_frame=source_cursor, source_end_frame=end,
                              output_start_frame=output_cursor, output_end_frame=output_cursor + count,
                              frames=count,
                              ramp_in_safe=bool(previous_cut and previous_cut['ramp_safe']['end']),
                              ramp_out_safe=bool(next_cut and next_cut['ramp_safe']['start'])))
            output_cursor += count
        if next_cut:
            source_cursor = next_cut['end_frame']
            previous_cut = next_cut
    if not output_cursor:
        raise ValueError('Cuts remove the entire narration; at least one frame must remain.')
    return dict(schema_version=1, operation='remove_confirmed_intervals',
                source_clock='original_narration', output_clock='edited_narration',
                fps=fps, source_frames=source_frames, frames=output_cursor,
                source_duration=source_frames/fps, duration=output_cursor/fps,
                removed_frames=source_frames-output_cursor, contains_insertions=False,
                cuts=normalized, keeps=keeps,
                edit_map=[dict({key:keep[key] for key in ('source_start_frame', 'source_end_frame',
                          'output_start_frame', 'output_end_frame', 'frames')}, fps=fps) for keep in keeps])


def _validated_map(edit_map):
    if not isinstance(edit_map, dict):
        raise ValueError('Edit map must be an object.')
    if edit_map.get('contains_insertions') is not False or edit_map.get('inserts'):
        raise ValueError('Insertions are unsupported; do not reapply old cuts after later edits.')
    rebuilt = build_map(edit_map.get('source_frames'), edit_map.get('cuts'), edit_map.get('fps'))
    for key in rebuilt:
        if edit_map.get(key) != rebuilt[key]:
            raise ValueError(f'Edit map field {key} disagrees with its original-clock cuts.')
    return rebuilt


def map_captions(edit_map, captions):
    """Intersect source cues with keeps; preserve text/IDs and flag clipped text.

    A cue spanning removed source time becomes ONE output cue with multiple
    source_segments. Its source_gaps remain explicit and require text review.
    No words are invented, removed, or inferred from relative duration.
    """
    m = _validated_map(edit_map)
    if isinstance(captions, dict):
        if captions.get('clock', 'original_narration') != 'original_narration':
            raise ValueError('Captions must use original narration clock, not an already edited timeline.')
        captions = captions.get('captions')
    if not isinstance(captions, list):
        raise ValueError('captions must be a list or an original-clock captions object.')
    output, audit = [], []
    prior_end = 0
    ids = set()
    for index, cue in enumerate(captions):
        if not isinstance(cue, dict):
            raise ValueError(f'Caption {index}: expected an object.')
        a = _integer(cue.get('start_frame'), f'Caption {index} start_frame')
        b = _integer(cue.get('end_frame'), f'Caption {index} end_frame', 1)
        text = cue.get('text')
        if not (a < b <= m['source_frames']) or a < prior_end:
            raise ValueError(f'Caption {index}: unsorted, overlapping or out-of-range interval.')
        if not isinstance(text, str) or not text.strip() or '\x00' in text:
            raise ValueError(f'Caption {index}: nonempty text is required.')
        cue_id = cue.get('id', f'source-cue-{index+1:04d}')
        if type(cue_id) not in (str, int) or cue_id == '':
            raise ValueError(f'Caption {index}: id must be a nonempty string or integer.')
        identity = (type(cue_id).__name__, cue_id)
        if identity in ids:
            raise ValueError(f'Caption {index}: duplicate source cue id {cue_id!r}.')
        ids.add(identity)
        segments, gaps = [], []
        for keep in m['keeps']:
            lo, hi = max(a, keep['source_start_frame']), min(b, keep['source_end_frame'])
            if lo < hi:
                offset = keep['output_start_frame'] - keep['source_start_frame']
                segments.append(dict(source_start_frame=lo, source_end_frame=hi,
                                     output_start_frame=lo+offset, output_end_frame=hi+offset))
        for cut_index, cut in enumerate(m['cuts']):
            lo, hi = max(a, cut['start_frame']), min(b, cut['end_frame'])
            if lo < hi:
                kind = 'entire' if (lo, hi) == (a, b) else ('leading' if lo == a else ('trailing' if hi == b else 'internal'))
                gaps.append(dict(source_start_frame=lo, source_end_frame=hi,
                                 kind=kind, cut_index=cut_index, reason=cut['reason']))
        retained_frames = sum(s['source_end_frame']-s['source_start_frame'] for s in segments)
        status = 'omitted' if not segments else ('clipped' if gaps else 'retained')
        record = dict(source_cue_id=cue_id, source_index=index,
                      source_start_frame=a, source_end_frame=b, text=text, status=status,
                      retained_frames=retained_frames, removed_frames=b-a-retained_frames,
                      source_segments=segments, source_gaps=gaps,
                      needs_text_review=bool(segments and gaps))
        if segments:
            for left, right in zip(segments, segments[1:]):
                if left['output_end_frame'] != right['output_start_frame']:
                    raise ValueError('A caption crosses a non-removal discontinuity; separate remapping is needed.')
            start, end = segments[0]['output_start_frame'], segments[-1]['output_end_frame']
            if output and start < output[-1]['end_frame']:
                raise ValueError('Mapped captions overlap.')
            output.append(dict(id=cue_id, source_cue_id=cue_id, source_index=index,
                               start_frame=start, end_frame=end, start=start/FPS, end=end/FPS,
                               text=text, source_segments=segments, source_gaps=gaps,
                               needs_text_review=bool(gaps)))
            record.update(output_start_frame=start, output_end_frame=end)
        audit.append(record)
        prior_end = b
    return dict(schema_version=1, clock='edited_narration', fps=FPS, frames=m['frames'],
                duration=m['duration'], captions=output, source_cues=audit,
                qa=dict(status='REVIEW_REQUIRED' if any(c['needs_text_review'] for c in output) else 'PASS',
                        source_count=len(captions), mapped_count=len(output),
                        retained_count=sum(c['status']=='retained' for c in audit),
                        clipped_count=sum(c['status']=='clipped' for c in audit),
                        omitted_count=sum(c['status']=='omitted' for c in audit),
                        source_text_and_ids_preserved_in_audit=True,
                        review_source_ids=[c['source_cue_id'] for c in output if c['needs_text_review']],
                        note='Clipped text is unchanged and requires listening review; source gaps are not word timings.'))


def _srt(mapped):
    def stamp(frame):
        ms = (frame*1000 + FPS//2)//FPS
        h, ms = divmod(ms, 3600000)
        minutes, ms = divmod(ms, 60000)
        seconds, ms = divmod(ms, 1000)
        return f'{h:02d}:{minutes:02d}:{seconds:02d},{ms:03d}'
    blocks = []
    for n, cue in enumerate(mapped['captions'], 1):
        if any(not line.strip() for line in cue['text'].splitlines()):
            raise ValueError('A source cue contains a blank line; review before SRT export rather than changing its text silently.')
        blocks.append(f"{n}\n{stamp(cue['start_frame'])} --> {stamp(cue['end_frame'])}\n{cue['text']}\n")
    return '\n'.join(blocks)


def _fingerprint(path):
    p = Path(path).resolve()
    stat = p.stat()
    return dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                device=stat.st_dev, inode=stat.st_ino)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _check_output(path, protected=(), force=False):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Refusing a symlink output: {path}')
    for original in protected:
        original = Path(original)
        if path.resolve() == original.resolve() or (path.exists() and original.exists() and path.samefile(original)):
            raise ValueError(f'Output would replace an input or protected file: {path}')
    if path.exists() and (not force or not path.is_file()):
        raise FileExistsError(f'Output already exists; use --force only for replaceable outputs: {path}')


def _publish(temp, destination, force):
    # Both operations publish a complete file atomically. link() also closes the
    # check-then-write race when overwriting was not authorized.
    if force:
        os.replace(temp, destination)
    else:
        os.link(temp, destination)
        os.unlink(temp)


def _atomic_text(path, text, force=False, protected=()):
    path = Path(path)
    _check_output(path, protected, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(name, path, force)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _ramped(data, offset, total, fade_in, fade_out):
    if not ((fade_in and offset < fade_in) or (fade_out and offset+len(data)//4 > total-fade_out)):
        return data
    samples = array.array('h')
    samples.frombytes(data)
    if sys.byteorder != 'little':
        samples.byteswap()
    for local in range(len(samples)//2):
        position = offset+local
        gain = 1.0
        if fade_in and position < fade_in:
            gain = position/(fade_in-1) if fade_in > 1 else 0.0
        if fade_out and position >= total-fade_out:
            gain *= (total-1-position)/(fade_out-1) if fade_out > 1 else 0.0
        if gain != 1.0:
            for channel in (0, 1):
                i = local*2+channel
                samples[i] = round(samples[i]*gain)
    if sys.byteorder != 'little':
        samples.byteswap()
    return samples.tobytes()


def cut_pcm(edit_map, input_path, output_path, ramp_ms=4.0, force=False):
    """Write exact 48k stereo s16 keeps, with explicitly permitted <=4 ms ramps.

    Returns detailed byte-fidelity QA. The CLI additionally saves this as a
    sidecar. The source is never opened for writing; an existing output is not
    replaced unless force=True. Media publication follows full validation.
    """
    m = _validated_map(edit_map)
    source, destination = Path(input_path), Path(output_path)
    if isinstance(ramp_ms, bool) or not math.isfinite(ramp_ms) or not 0 <= ramp_ms <= 4:
        raise ValueError('ramp_ms must be between 0 and 4; only confirmed word-free boundaries may ramp.')
    _check_output(destination, (source,), force)
    before = _fingerprint(source)
    source_hash = _sha256(source)
    ramp_samples = round(ramp_ms*RATE/1000)
    spans = []
    with wave.open(str(source), 'rb') as reader:
        fmt = (reader.getframerate(), reader.getnchannels(), reader.getsampwidth(), reader.getcomptype())
        if fmt != (RATE, CHANNELS, SAMPLE_WIDTH, 'NONE'):
            raise ValueError(f'Expected 48k stereo signed-16-bit uncompressed WAV, got {fmt}.')
        if reader.getnframes() != m['source_frames']*SAMPLES_PER_FRAME:
            raise ValueError('Input sample count must equal source_frames × 800 exactly; this may be already edited audio.')
        for keep in m['keeps']:
            start, end = keep['source_start_frame']*800, keep['source_end_frame']*800
            count = end-start
            fade_in = ramp_samples if keep['ramp_in_safe'] else 0
            fade_out = ramp_samples if keep['ramp_out_safe'] else 0
            if fade_in+fade_out > count:
                raise ValueError('Approved ramps overlap in a short keep; revise edit decisions.')
            spans.append(dict(source_start_sample=start, source_end_sample=end,
                              output_start_sample=keep['output_start_frame']*800,
                              output_end_sample=keep['output_end_frame']*800,
                              fade_in_samples=fade_in, fade_out_samples=fade_out))
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temp_name = tempfile.mkstemp(prefix='.'+destination.name+'.', suffix='.tmp', dir=destination.parent)
        os.close(descriptor)
        try:
            with wave.open(temp_name, 'wb') as writer:
                writer.setparams((CHANNELS, SAMPLE_WIDTH, RATE, 0, 'NONE', 'not compressed'))
                for span in spans:
                    reader.setpos(span['source_start_sample'])
                    total = span['source_end_sample']-span['source_start_sample']
                    offset = 0
                    while offset < total:
                        take = min(RATE, total-offset)
                        data = reader.readframes(take)
                        if len(data) != take*4:
                            raise ValueError('Source ended before the declared keep.')
                        writer.writeframesraw(_ramped(data, offset, total, span['fade_in_samples'], span['fade_out_samples']))
                        offset += take
            compared = exact_bytes = allowed_ramp_bytes = different_unapproved = incorrect_ramp_bytes = 0
            with wave.open(temp_name, 'rb') as candidate:
                if candidate.getnframes() != m['frames']*800:
                    raise ValueError('Output sample count does not match the mapped frame count.')
                for span in spans:
                    reader.setpos(span['source_start_sample'])
                    candidate.setpos(span['output_start_sample'])
                    total = span['source_end_sample']-span['source_start_sample']
                    offset = 0
                    while offset < total:
                        take = min(RATE, total-offset)
                        original, actual = reader.readframes(take), candidate.readframes(take)
                        if len(original) != take*4 or len(actual) != take*4:
                            raise ValueError('Incomplete PCM during fidelity validation.')
                        expected = _ramped(original, offset, total, span['fade_in_samples'], span['fade_out_samples'])
                        plain_start = min(take, max(0, span['fade_in_samples']-offset))
                        plain_end = min(take, max(0, total-span['fade_out_samples']-offset))
                        if plain_end < plain_start:
                            raise ValueError('Invalid non-ramp comparison range.')
                        lo, hi = plain_start*4, plain_end*4
                        exact_bytes += hi-lo
                        allowed_ramp_bytes += take*4-(hi-lo)
                        if original[lo:hi] != actual[lo:hi]:
                            different_unapproved += sum(a != b for a, b in zip(original[lo:hi], actual[lo:hi]))
                        if expected[:lo] != actual[:lo] or expected[hi:] != actual[hi:]:
                            incorrect_ramp_bytes += sum(a != b for a, b in zip(expected[:lo]+expected[hi:], actual[:lo]+actual[hi:]))
                        compared += take*4
                        offset += take
            if different_unapproved or incorrect_ramp_bytes:
                raise ValueError('PCM byte-fidelity verification failed; output not published.')
            if _fingerprint(source) != before or _sha256(source) != source_hash:
                raise ValueError('Input changed during editing; output not published.')
            output_hash = _sha256(temp_name)
            with open(temp_name, 'rb') as stream:
                os.fsync(stream.fileno())
            _publish(temp_name, destination, force)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return dict(status='PASS', source=before, source_sha256=source_hash,
                output=str(destination.resolve()), output_sha256=output_hash,
                fps=FPS, sample_rate=RATE, channels=CHANNELS, sample_width=SAMPLE_WIDTH,
                source_frames=m['source_frames'], frames=m['frames'], duration=m['duration'],
                output_sample_frames=m['frames']*800, sample_frames_per_video_frame=800,
                compared_retained_pcm_bytes=compared, exact_unmodified_pcm_bytes=exact_bytes,
                approved_ramp_pcm_bytes=allowed_ramp_bytes, differing_unapproved_bytes=different_unapproved,
                incorrect_ramp_bytes=incorrect_ramp_bytes, source_unchanged=True,
                configured_ramp_ms=ramp_ms, spans=spans,
                note='Only explicitly approved word-free edges use linear ramps. No resampling, normalization, insertion or voice-speed change.')


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2)+'\n'


def selftest():
    """Temporary synthetic PCM and fictional captions only; no real speech."""
    def rejects(function):
        try:
            function()
        except (ValueError, FileExistsError):
            return
        raise AssertionError('Invalid input unexpectedly accepted.')
    cut = dict(start_frame=20, end_frame=40, reason='Synthetic test cut', evidence='Artificial fixture; no speech')
    m = build_map(90, [cut])
    assert m['frames'] == 70 and m['keeps'][1]['output_start_frame'] == 20
    assert build_map(90, [])['frames'] == 90
    rejects(lambda:build_map(90, [cut, dict(cut, start_frame=39, end_frame=50)]))
    rejects(lambda:build_map(90, [dict(cut, start_frame=50, end_frame=60), cut]))
    rejects(lambda:build_map(90, [dict(cut, reason='')]))
    rejects(lambda:build_map(90, [dict(cut, evidence=None)]))
    rejects(lambda:build_map(90, [dict(cut, start_frame=0, end_frame=90)]))
    rejects(lambda:build_map(90, [dict(cut, start_frame=20.5)]))
    cues = [dict(id='before', start_frame=0, end_frame=10, text='虚构说明'),
            dict(id='cross', start_frame=15, end_frame=45, text='跨剪口原文不自动改写'),
            dict(id='after', start_frame=45, end_frame=60, text='实际结果')]
    mapped = map_captions(m, cues)
    assert [(c['start_frame'],c['end_frame']) for c in mapped['captions']] == [(0,10),(15,25),(25,40)]
    cross = mapped['captions'][1]
    assert cross['text'] == cues[1]['text'] and len(cross['source_segments']) == 2
    assert cross['source_gaps'][0]['kind'] == 'internal' and cross['needs_text_review']
    omitted = map_captions(m, [dict(id=42, start_frame=20, end_frame=40, text='完全位于剪区')])
    assert omitted['captions'] == [] and omitted['source_cues'][0]['source_cue_id'] == 42
    assert omitted['qa']['omitted_count'] == 1
    edges = map_captions(m, [dict(id='left', start_frame=10,end_frame=20,text='左边界'),
                             dict(id='right',start_frame=40,end_frame=50,text='右边界')])
    assert [(c['start_frame'],c['end_frame']) for c in edges['captions']] == [(10,20),(20,30)]
    assert not any(c['needs_text_review'] for c in edges['captions'])
    assert '00:00:00,250 --> 00:00:00,417' in _srt(mapped)
    rejects(lambda:map_captions(m, [cues[0],dict(cues[1],start_frame=9)]))
    rejects(lambda:map_captions(m,dict(clock='edited_narration',captions=cues)))
    with tempfile.TemporaryDirectory(prefix='course-timeline-selftest-') as tmp:
        root = Path(tmp); source = root/'synthetic.wav'
        values = array.array('h')
        for n in range(90*800):
            # Deterministic tones contain no voices or private recordings.
            values.extend((round(10000*math.sin(n*.031)),round(8000*math.cos(n*.047))))
        if sys.byteorder != 'little':
            values.byteswap()
        raw = values.tobytes()
        with wave.open(str(source),'wb') as w:
            w.setparams((2,2,48000,0,'NONE','not compressed'));w.writeframes(raw)
        original_hash = _sha256(source)
        q = cut_pcm(m, source, root/'plain.wav')
        assert q['approved_ramp_pcm_bytes'] == 0 and q['exact_unmodified_pcm_bytes'] == 70*800*4
        with wave.open(str(root/'plain.wav'),'rb') as w:
            actual = w.readframes(w.getnframes())
        assert actual == raw[:20*800*4]+raw[40*800*4:]
        safe = build_map(90,[dict(cut,ramp_safe=dict(start=True,end=True))])
        q2 = cut_pcm(safe,source,root/'ramped.wav')
        assert q2['approved_ramp_pcm_bytes'] == 2*192*4 and q2['differing_unapproved_bytes'] == 0
        with wave.open(str(root/'ramped.wav'),'rb') as w:
            w.setpos(20*800-1);assert w.readframes(2) == b'\0'*8
        rejects(lambda:cut_pcm(m,source,source,force=True))
        rejects(lambda:cut_pcm(m,source,root/'plain.wav'))
        rejects(lambda:cut_pcm(build_map(89,[]),source,root/'bad.wav'))
        link = root/'hardlink.wav';os.link(source,link)
        rejects(lambda:cut_pcm(m,source,link,force=True))
        cut_pcm(m,source,root/'plain.wav',force=True)
        identity = cut_pcm(build_map(90,[]),source,root/'identity.wav')
        assert identity['exact_unmodified_pcm_bytes'] == len(raw)
        assert _sha256(source) == original_hash
    return dict(status='PASS', fixtures='Temporary synthetic tones and fictional captions; no human speech.',
                tested=['Integer-frame/sample mapping','Empty cuts identity','Strict sorted nonoverlap and evidence rejection',
                        'Caption shift, exact boundaries, preserved IDs/text and internal gaps','Fully omitted caption audit',
                        'Unapproved ramps disabled','Approved 4 ms ramp edges','Full retained PCM byte identity',
                        'Input/alias protection and overwrite refusal','Exact source sample count and atomic publication'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest',action='store_true',help='Run synthetic temporary fixtures only.')
    sub = parser.add_subparsers(dest='command')
    build = sub.add_parser('build-map',help='Build a 60fps map from confirmed original-clock cuts.')
    build.add_argument('--frames',type=int,required=True);build.add_argument('--cuts',type=Path,required=True)
    build.add_argument('--out',type=Path,required=True);build.add_argument('--force',action='store_true')
    captions = sub.add_parser('map-captions',help='Intersect source cues; clipped text is flagged, never rewritten.')
    captions.add_argument('--map',type=Path,required=True);captions.add_argument('--captions',type=Path,required=True)
    captions.add_argument('--out',type=Path,required=True);captions.add_argument('--srt',type=Path,required=True)
    captions.add_argument('--force',action='store_true')
    pcm = sub.add_parser('cut-pcm',help='Edit exact 48k stereo16 PCM; default ramp 4ms only at approved edges.')
    pcm.add_argument('--map',type=Path,required=True);pcm.add_argument('--input',type=Path,required=True)
    pcm.add_argument('--out',type=Path,required=True);pcm.add_argument('--ramp-ms',type=float,default=4.0)
    pcm.add_argument('--force',action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.selftest:
            if args.command:
                raise ValueError('--selftest cannot be combined with a production command.')
            print(_json(selftest()),end='');return 0
        if not args.command:
            parser.error('Choose build-map, map-captions, cut-pcm, or --selftest.')
        if args.command == 'build-map':
            value = build_map(args.frames,_read_json(args.cuts))
            _atomic_text(args.out,_json(value),args.force,(args.cuts,))
            summary = dict(status='PASS',output=str(args.out.resolve()),source_frames=value['source_frames'],frames=value['frames'])
        elif args.command == 'map-captions':
            _check_output(args.out,(args.map,args.captions,args.srt),args.force)
            _check_output(args.srt,(args.map,args.captions,args.out),args.force)
            value = map_captions(_read_json(args.map),_read_json(args.captions))
            srt = _srt(value)
            _atomic_text(args.out,_json(value),args.force,(args.map,args.captions,args.srt))
            _atomic_text(args.srt,srt,args.force,(args.map,args.captions,args.out))
            summary = dict(**value['qa'],output=str(args.out.resolve()),srt=str(args.srt.resolve()))
        else:
            qa_path = args.out.with_name(args.out.name+'.qa.json')
            _check_output(args.out,(args.input,args.map,qa_path),args.force)
            _check_output(qa_path,(args.input,args.map,args.out),args.force)
            value = cut_pcm(_read_json(args.map),args.input,args.out,args.ramp_ms,args.force)
            _atomic_text(qa_path,_json(value),args.force,(args.input,args.map,args.out))
            summary = dict(status=value['status'],output=value['output'],frames=value['frames'],qa=str(qa_path.resolve()))
        print(_json(summary),end='');return 0
    except (ValueError,TypeError,KeyError,OSError,wave.Error,json.JSONDecodeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 2


if __name__ == '__main__':
    sys.exit(main())
