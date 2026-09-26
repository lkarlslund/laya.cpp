"""Isolated baseline adapter used exclusively by validation and benchmark tools."""
import sys
from pathlib import Path
import torch


class Oracle:
    def __init__(self, source, model, fp32=False, fp16=False, device='cuda'):
        if fp32 and fp16: raise ValueError('Choose one baseline precision')
        if device not in ('cuda', 'cpu'): raise ValueError('Unknown baseline device')
        if device == 'cpu' and not fp32: raise ValueError('CPU baseline currently requires FP32')
        sys.path.insert(0, str(Path(source).resolve()))
        import laya
        from laya.common import build_sequence, collate_items, QTYPES
        self.build_sequence, self.collate, self.types = build_sequence, collate_items, QTYPES
        self.agent = laya.load(str(Path(model).resolve()), device=device)
        if self.agent.device.type != device:
            raise RuntimeError(f'Baseline did not load on {device}')
        self.device = device
        self.fp32 = fp32
        if fp16: self.agent.dtype = torch.float16
        if not fp32 and self.agent.dtype != (torch.float16 if fp16 else torch.bfloat16):
            raise RuntimeError('Baseline selected a different autocast dtype')
        if fp32 and device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def prepare(self, requests):
        items = []
        for request in requests:
            for q in request['questions'].values():
                q = self.agent._to_internal(q)
                ids, markers = self.build_sequence(self.agent.tok, request['state'], q,
                                                   self.agent.cfg.get('max_len', 512), self.agent.cfg.get('head_max_len', 192))
                items.append(dict(ids=ids, markers=markers, qtype=self.types[q['t']]))
        batch = self.collate([items], self.agent.tok.pad_token_id)
        tensors = tuple(batch[key] for key in ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype'))
        return tensors

    @torch.inference_mode()
    def forward(self, inputs):
        with torch.autocast(self.device, dtype=self.agent.dtype, enabled=not self.fp32):
            return self.agent.model(*(t.to(self.device) for t in inputs))

    def expected_inputs(self, tensors):
        ids, mask, markers, valid, types = tensors
        batch, length = ids.shape
        flat_markers = markers + torch.arange(batch)[:, None]*length
        return dict(batch=batch, length=length, options=markers.shape[1], ids=ids.flatten().tolist(),
                    lengths=mask.sum(-1).tolist(), markers=flat_markers.flatten().tolist(),
                    counts=valid.sum(-1).tolist(), types=types.tolist())

    def format(self, requests, outputs):
        # Reuse the public output formatter with precomputed batched logits. Model
        # evaluation is bypassed here; this is outside all performance timing.
        original = self.agent.model
        results, offset = [], 0
        try:
            for request in requests:
                count = len(request['questions'])
                selected = tuple(t[offset:offset+count] for t in outputs)
                self.agent.model = lambda *args, values=selected: values
                results.append(self.agent.predict(request['state'], request['questions']))
                offset += count
        finally:
            self.agent.model = original
        return results

    def finish(self, requests, inputs, outputs):
        """Batched host formatting, checked against the public formatter in validation."""
        import math
        import numpy as np
        logits = outputs[0].float().cpu().numpy()
        actions = torch.softmax(outputs[1].float(), -1).cpu().numpy()
        results, row = [], 0
        for request in requests:
            answers, tokens = {}, 0
            for name, definition in request['questions'].items():
                q = self.agent._to_internal(definition)
                kind = q['t']
                count = int(inputs[3][row].sum())
                bucket = '2' if count <= 2 else '3-5' if count <= 5 else '6-10' if count <= 10 else '11+'
                temperature = self.agent.temperature_by_options.get(kind+':'+bucket, self.agent.temperature[self.types[kind]])
                z = logits[row, :count] / max(0.001, float(temperature))
                probabilities = np.exp(z-z.max()); probabilities /= probabilities.sum()
                entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum()
                answer = dict(type=kind, confidence=round(float(np.clip(1-entropy/math.log(count), 0, 1)), 4),
                              action={'act_probability': round(float(actions[row, 0]), 4)})
                if kind == 'choice':
                    names = list(q['crit'])
                    answer['choice'] = names[int(probabilities.argmax())]
                    answer['probabilities'] = {k: round(float(v), 4) for k,v in zip(names, probabilities)}
                elif kind == 'score':
                    answer['score'] = round(float((np.arange(count)*probabilities).sum()), 4)
                    answer['legend'] = {str(i): v for i,v in enumerate(q['crit'])}
                    answer['probabilities'] = {str(i): round(float(v), 4) for i,v in enumerate(probabilities)}
                else:
                    value = float(probabilities[1])
                    answer['noul'] = round(value, 4)
                    answer['confidence'] = round(max(value, 1-value), 4)
                tokens += int(inputs[1][row].sum())
                answers[name] = answer
                row += 1
            results.append(dict(model='laya-rl-agent', answers=answers, usage=dict(input_tokens=tokens, output_tokens=0)))
        return results

    def predict_batch(self, requests):
        inputs = self.prepare(requests)
        return self.finish(requests, inputs, self.forward(inputs))
