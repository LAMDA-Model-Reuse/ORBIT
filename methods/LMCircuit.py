"""Circuits of LLM Classifiers (Khalaf et al., arXiv:2610.30076), in ORBIT.

Classification retains a latent-type belief, a class head and exact finite-state
backward induction. Non-classification adds a quality head selecting an observed
answer; its finite response codebook is an approximation, not the paper's exact
open-text optimum. No answer generation, paid judge or upstream package is used.

The belief equations and DP follow hskhalaf/lmcircuit@24a3ac3 (MIT). This is an
independent ORBIT implementation for one recorded call per model, without the
upstream pairwise candidate-swap extension. All algorithm code lives here.
"""
from __future__ import annotations

import ast
from copy import deepcopy
from fnmatch import fnmatchcase
import json
import logging
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupShuffleSplit

from methods.base import BaseRouter
from utils.embedding import Embedder


def _text(value):
    """Preserve ordered turns/messages, distinguishing an empty call from no record."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Missing recorded response; supply circuit fields, never impute a score as an answer.")
    original = value
    if isinstance(value, str) and value.lstrip().startswith(("[", "{")):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if isinstance(value, (list, dict, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            # Code snippets can be Python literals containing sets or mixed
            # dictionary keys; they remain the original recorded text.
            return str(original)
    return str(value)


def _class_index(value, classes, invalid=False):
    """Parse an explicit class/index or a final answer marker, never a gold hint."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if 0 <= int(value) < len(classes):
            return int(value)
    try:
        text = _text(value).strip()
    except ValueError:
        text = ""
    if text.startswith("["):
        try:
            parts = json.loads(text)
            if isinstance(parts, list) and parts:
                text = str(parts[-1]).strip()
        except (ValueError, TypeError):
            pass
    normalized = [str(c).strip().casefold() for c in classes]
    candidates = [text, text.strip("().[] $*`\n")]
    text = text.replace(r"\n", "\n").replace(r"\r", "\r")
    text = text.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{\s*([A-Z])\s*\}", r"\1", text)
    # Use a final verdict when present, including common math/Markdown output.
    # Do not turn an explicitly multi-answer verdict (e.g. K and L) into K.
    markers = list(re.finditer(
        r"(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|正确答案|答案)"
        r"(?:\s+(?:is\b|would\s+be\b)|\s*[:：]|\s*[为是])[\s:：]*", text, re.I))
    verdict = text[markers[-1].end():].strip() if markers else text
    multi = re.match(r"^[\s*$`(]*[A-Z][)\s*$`]*(?:and|or|&|/|,|、|和|及)\s*[(*$`]*[A-Z]\b", verdict)
    lines = re.findall(r"(?:^|\n)\s*\**\(?([A-Z])\)?[.):：]\s", verdict)
    if not multi:
        boxed = re.findall(r"\\boxed\s*\{\s*\(?([A-Z])\)?\s*\}", verdict)
        if boxed:
            candidates.insert(0, boxed[-1])
        short = re.match(r"^[\s*$`]*\(?([A-Z])\)?(?:[.)：:]|\s|$)", verdict)
        if short and (markers or len(set(lines)) <= 1):
            candidates.insert(0 if not boxed else 1, short.group(1))
        # A final standalone label is also used after an explanation or a value.
        final = re.search(r"(?:^|\n|[.!?:]\s+)\s*[$(]*([A-Z])[)$]*[.!]?\s*$", verdict)
        if final:
            candidates.append(final.group(1))
        # Some releases retain a label embedded in a short explanatory sentence.
        inline = re.findall(r"\b([A-Z])[.)]\s+", verdict)
        if len(set(inline)) == 1:
            candidates.append(inline[-1])
    for candidate in candidates:
        if candidate.casefold() in normalized:
            return normalized.index(candidate.casefold())
    if invalid:
        return len(classes)  # a recorded refusal/invalid verdict is an observable outcome
    raise ValueError(f"Cannot parse target {text[:80]!r} in classes {classes}; supply verified labels.")


def _choice_options(prompt):
    """Read explicit options from the input, without consulting its gold answer."""
    if "Choices:" not in prompt:
        return {}
    block = prompt.rsplit("Choices:", 1)[-1]
    matches = list(re.finditer(r"(?:^|\n)\s*\(?([A-Z])(?:\)|\.)[ \t]+", block))
    return {match.group(1): block[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(block)].strip()
            for i, match in enumerate(matches)}


def _target_distribution(row, classes):
    """Represent equivalent gold options; an out-of-alphabet gold earns no hit.

    Some MMRBench choice prompts have a free-value gold, and duplicate option
    texts can share the gold value. Keep these rows, never infer their task type
    from the gold, and do not invent a class label from a recorded quality score.
    """
    value = row.get("answer")
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Classification requires a verified, nonmissing gold answer.")
    text = str(value).strip()
    normalize = lambda s: " ".join(str(s).split()).casefold()
    matches = [label for label, option in _choice_options(str(row.prompt)).items()
               if normalize(option) == normalize(text)]
    if normalize(text) in [normalize(c) for c in classes]:
        matches = [text]
    distribution = np.zeros(len(classes), dtype=np.float32)
    if matches:
        for label in matches:
            distribution[_class_index(label, classes)] = 1
    else:
        index = _class_index(value, classes, invalid=True)
        if index < len(classes):
            distribution[index] = 1
    return distribution / max(float(distribution.sum()), 1)


class _StateSpace:
    """Digit A denotes unasked; other digits are the recorded answer/code."""

    def __init__(self, models, answers, max_states=6561):
        self.M, self.A = int(models), int(answers)
        self.size = (self.A + 1) ** self.M
        if self.M < 1 or self.A < 1 or self.size > max_states:
            raise ValueError(f"Circuit needs {self.size} states; reduce pool_size/answer_codes (limit {max_states}).")
        self.powers = (self.A + 1) ** np.arange(self.M, dtype=np.int64)
        self.digits = (np.arange(self.size)[:, None] // self.powers) % (self.A + 1)
        self.root = self.size - 1
        self.depth = (self.digits != self.A).sum(axis=1)


def _solve_dp(space, stop_reward, next_probs, costs, lambdas, tol=1e-10):
    """Exact on the supplied finite belief: stop wins ties, then lowest model ID.

    stop_reward (S,), next_probs (S,M,A), costs (M,), result actions (L,S).
    Costs are *planning* costs, never the current test row's realized costs.
    """
    stop = np.asarray(stop_reward, dtype=np.float64)
    probs, costs, prices = map(lambda a: np.asarray(a, dtype=np.float64), (next_probs, costs, lambdas))
    if stop.shape != (space.size,) or probs.shape != (space.size, space.M, space.A):
        raise ValueError("Circuit DP has inconsistent state/probability dimensions.")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(-1), 1, atol=1e-5):
        raise ValueError("Circuit transition probabilities must be finite distributions.")
    if costs.shape != (space.M,) or not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Circuit planning costs must be finite and nonnegative.")
    if (np.isnan(prices) | (prices < 0)).any():
        raise ValueError("Circuit prices must be nonnegative.")
    values = np.broadcast_to(stop, (len(prices), space.size)).copy()
    actions = np.full(values.shape, -1, dtype=np.int32)
    with np.errstate(invalid="ignore"):
        penalties = prices[:, None] * costs[None]
    penalties[np.isinf(prices)] = np.inf  # infinity means no query, even for a free call
    for depth in range(space.M - 1, -1, -1):
        for j in range(space.M):
            rows = np.flatnonzero((space.depth == depth) & (space.digits[:, j] == space.A))
            children = rows[:, None] + (np.arange(space.A) - space.A) * space.powers[j]
            expected = (values[:, children] * probs[rows, j][None]).sum(-1) - penalties[:, j, None]
            better = expected > values[:, rows] + tol
            values[:, rows] = np.where(better, expected, values[:, rows])
            actions[:, rows] = np.where(better, j, actions[:, rows])
    return values, actions


def _solve_dp_batch(space, stop_reward, next_probs, costs, lambdas, tol=1e-10):
    """Batched torch version of the same backward induction, on CPU or GPU.

    The leading dimension contains independent queries; no beliefs or policies
    are shared between them. Float64 and the same update order preserve ties.
    """
    stop = stop_reward.to(torch.float64)
    probs = next_probs.to(torch.float64)
    device = stop.device
    prices = torch.as_tensor(lambdas, dtype=torch.float64, device=device)
    costs = torch.as_tensor(costs, dtype=torch.float64, device=device)
    values = stop[:, None].expand(-1, len(prices), -1).clone()
    actions = torch.full(values.shape, -1, dtype=torch.int32, device=device)
    penalties = prices[:, None] * costs[None]
    penalties[torch.isinf(prices)] = torch.inf
    for depth in range(space.M - 1, -1, -1):
        for j in range(space.M):
            rows = np.flatnonzero((space.depth == depth) & (space.digits[:, j] == space.A))
            children = rows[:, None] + (np.arange(space.A) - space.A) * space.powers[j]
            rows = torch.as_tensor(rows, device=device)
            children = torch.as_tensor(children, device=device)
            expected = (values[:, :, children] * probs[:, None, rows, j]).sum(-1) - penalties[None, :, j, None]
            current = values[:, :, rows]
            better = expected > current + tol
            values[:, :, rows] = torch.where(better, expected, current)
            actions[:, :, rows] = torch.where(better, j, actions[:, :, rows])
    return values, actions


def _replay(space, actions, terminal, observe):
    """Only invoke observe(j) for a paid query; terminal is a class or pool index."""
    state, trajectory, seen = space.root, [], set()
    for _ in range(space.M + 1):
        j = int(actions[state])
        if j < 0:
            return int(terminal[state]), trajectory, state
        if j in seen or j >= space.M:
            raise RuntimeError("Circuit attempted an invalid/repeated query.")
        answer = int(observe(j))
        if not 0 <= answer < space.A:
            raise ValueError("Observed response is outside the trained alphabet.")
        trajectory.append(j)
        seen.add(j)
        state += (answer - space.A) * space.powers[j]
    raise RuntimeError("Circuit did not terminate.")


class _BeliefNet(nn.Module):
    """Latent-type Bayes update and permutation-invariant observation tokens."""

    def __init__(self, features, models, answers, outputs, mode, types=4, width=64, hidden=128):
        super().__init__()
        self.M, self.A, self.T, self.mode = models, answers, types, mode
        self.text = nn.Linear(features, width)
        self.type_head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, types))
        self.response_head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, types * models * answers))
        self.model_emb, self.answer_emb, self.type_emb = nn.Embedding(models, width), nn.Embedding(answers, width), nn.Embedding(types, width)
        self.token = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width))
        self.trunk = nn.Sequential(nn.Linear(2 * width + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.output_head = nn.Linear(hidden, outputs)
        self.register_buffer("temperature", torch.ones(2))

    def forward(self, x, digits, calibrated=True):
        h = self.text(x)
        log_pi = self.type_head(h).log_softmax(-1)
        log_q = self.response_head(h).view(-1, self.T, self.M, self.A).log_softmax(-1)
        observed = digits != self.A
        answers = digits.clamp(max=self.A - 1)
        tokens = self.model_emb.weight[None] + self.answer_emb(answers)
        pooled = (self.token(tokens[:, None] + self.type_emb.weight[None, :, None]) * observed[:, None, :, None]).sum(2)
        share = observed.float().mean(1)[:, None, None].expand(-1, self.T, 1)
        latent = self.trunk(torch.cat((h[:, None] + self.type_emb.weight[None], pooled, share), -1))
        seen_log_q = log_q.gather(-1, answers[:, None, :, None].expand(-1, self.T, -1, 1)).squeeze(-1)
        log_w = (log_pi + (seen_log_q * observed[:, None]).sum(-1)).log_softmax(-1)
        raw = self.output_head(latent)
        if self.mode == "classification":
            output = torch.logsumexp(log_w[..., None] + raw.log_softmax(-1), 1)
            output = (output / (self.temperature[0] if calibrated else 1)).log_softmax(-1)
        else:
            output = (log_w.exp()[..., None] * raw.sigmoid()).sum(1)
        response = torch.logsumexp(log_w[..., None, None] + log_q, 1)
        response = (response / (self.temperature[1] if calibrated else 1)).log_softmax(-1)
        return output, response


def _sample_states(answers, alphabet, rng):
    n, m = answers.shape
    ranks = rng.random((n, m)).argsort(1).argsort(1)
    counts = rng.integers(0, m + 1, size=n)
    return np.where(ranks < counts[:, None], answers, alphabet)


def _losses(net, x, states, answers, targets, calibrated=False):
    output, response = net(x, states, calibrated)
    if net.mode == "classification":
        if targets.ndim == 2:
            valid = targets.sum(-1) > 0
            primary = -(targets[valid] * output[valid]).sum(-1).mean() if valid.any() else output.sum() * 0
        else:
            primary = F.nll_loss(output, targets.long())
    else:
        # All quality labels are training supervision, not inference observations.
        primary = F.mse_loss(output, targets)
    per = F.nll_loss(response.flatten(0, 1), answers.flatten(), reduction="none").view_as(answers)
    unasked = states == net.A
    count = unasked.sum(1)
    active = count > 0
    secondary = ((per * unasked).sum(1)[active] / count[active]).mean() if active.any() else response.sum() * 0
    return primary, secondary


def _fit_belief(x, answers, targets, train, val, mode, outputs, alphabet, cfg, seed, device):
    """Train type candidates; validation only selects epochs/types/temperatures."""
    tx = torch.as_tensor(x, dtype=torch.float32, device=device)
    ta = torch.as_tensor(answers, dtype=torch.long, device=device)
    ty = torch.as_tensor(targets, dtype=torch.float32, device=device)
    epochs, batch = int(cfg.get("epochs", 300)), int(cfg.get("batch_size", 256))
    if epochs < 1 or batch < 1:
        raise ValueError("Circuit epochs and batch_size must be positive.")
    fixed = torch.as_tensor(_sample_states(answers[val], alphabet, np.random.default_rng(seed + 100)), device=device)
    best_net, best_score, record = None, np.inf, []
    for types in cfg.get("types", [1, 2, 4, 8]):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = _BeliefNet(x.shape[1], answers.shape[1], alphabet, outputs, mode, int(types),
                         int(cfg.get("width", 64)), int(cfg.get("hidden", 128))).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.get("lr", 0.002)), weight_decay=float(cfg.get("weight_decay", 1e-4)))
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        best, state, stale = np.inf, None, 0
        for epoch in range(epochs):
            net.train()
            for ids in np.array_split(rng.permutation(train), max(1, int(np.ceil(len(train) / batch)))):
                states = torch.as_tensor(_sample_states(answers[ids], alphabet, rng), device=device)
                second = torch.as_tensor(_sample_states(answers[ids], alphabet, rng), device=device)
                primary, _ = _losses(net, tx[ids], states, ta[ids], ty[ids])
                _, response_loss = _losses(net, tx[ids], second, ta[ids], ty[ids])
                loss = primary + float(cfg.get("response_weight", 1)) * response_loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite circuit training loss.")
                opt.zero_grad()
                loss.backward()
                opt.step()
            schedule.step()
            net.eval()
            with torch.no_grad():
                main, other = _losses(net, tx[val], fixed, ta[val], ty[val])
            if main.item() < best:
                best, state, stale = main.item(), deepcopy(net.state_dict()), 0
            else:
                stale += 1
            if stale >= int(cfg.get("patience", 30)):
                break
        net.load_state_dict(state)
        with torch.no_grad():
            main, other = _losses(net, tx[val], fixed, ta[val], ty[val])
        score = (main + float(cfg.get("response_weight", 1)) * other).item()
        record.append({"types": int(types), "epochs": epoch + 1, "validation_loss": score})
        if score < best_score:
            best_score, best_net = score, net
    if best_net is None:
        raise ValueError("Circuit types must contain at least one positive type count.")
    # Calibration on fixed held-out states, without changing the fitted network.
    rng = np.random.default_rng(seed + 101)
    rows = np.tile(val, int(cfg.get("calibration_states", 10)))
    states = torch.as_tensor(_sample_states(answers[rows], alphabet, rng), device=device)
    with torch.no_grad():
        out, resp = best_net(tx[rows], states, False)
    log_temp = torch.zeros(2, device=device, requires_grad=True)
    opt = torch.optim.Adam([log_temp], lr=0.05)
    for _ in range(int(cfg.get("calibration_steps", 60))):
        temp = log_temp.clamp(-3, 3).exp()
        loss = out.sum() * 0
        if mode == "classification":
            calibrated = (out / temp[0]).log_softmax(-1)
            if ty.ndim == 2:
                valid = ty[rows].sum(-1) > 0
                loss = -(ty[rows][valid] * calibrated[valid]).sum(-1).mean() if valid.any() else calibrated.sum() * 0
            else:
                loss = F.nll_loss(calibrated, ty[rows].long())
        per = F.nll_loss((resp / temp[1]).log_softmax(-1).flatten(0, 1), ta[rows].flatten(), reduction="none").view_as(ta[rows])
        mask = states == alphabet
        loss = loss + (per * mask).sum() / mask.sum().clamp_min(1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    best_net.temperature.copy_(log_temp.detach().clamp(-3, 3).exp())
    best_net.eval()
    return best_net, record


def _choose_pool(performance, costs, size):
    """Training-only frontier coverage, greedy additions followed by improving swaps."""
    means, prices = np.mean(performance, 0), np.mean(costs, 0)
    positive = prices[prices > 0]
    grid = np.geomspace(max(float(positive.min()) if len(positive) else 1e-8, 1e-8), max(float(prices.max()), 1e-8), 64)
    frontier = np.where(prices[:, None] <= grid, means[:, None], 0)
    def score(pool):
        return float(frontier[pool].max(0).mean()) if pool else -np.inf
    pool = []
    while len(pool) < min(size, len(means)):
        pool.append(max((j for j in range(len(means)) if j not in pool), key=lambda j: (score(pool + [j]), -prices[j], -j)))
    changed = True
    while changed:
        changed = False
        for old in list(pool):
            for new in range(len(means)):
                candidate = [new if j == old else j for j in pool]
                if new not in pool and score(candidate) > score(pool) + 1e-10:
                    pool, changed = candidate, True
    return np.asarray(sorted(pool, key=lambda j: (prices[j], j)), dtype=int)


class LMCircuitRouter(BaseRouter):
    """ORBIT router with task-specific terminal heads and a shared DP engine.

    predict(X) retains the standard (N,M) prior-quality/cost contract. Sequential
    evaluation explicitly replays policies; it never feeds those matrices into
    BaseRouter's one-model evaluator or reads test quality to choose an answer.
    """

    supported_benchmarks = {"Routerbench", "Mixinstruct", "RouterEval", "MMRBench", "LLMRouterBench"}

    def __init__(self, args):
        super().__init__(args)
        self.cfg = dict(args.get("lmcircuit", {}))
        self.groups = {}
        self._response_embedder = None
        self.prices = np.asarray(self.cfg.get("lambdas", [0.0] + np.geomspace(1e-8, 100, 201).tolist() + [float("inf")]), dtype=float)
        self.budgets = np.asarray(self.cfg.get("budgets", [0.0] + np.geomspace(0.003, 1, 50).tolist()
            + np.geomspace(1.01, 3, 30).tolist()), dtype=float)
        if not len(self.prices) or not len(self.budgets) or not np.isfinite(self.budgets).all() or (self.budgets < 0).any():
            raise ValueError("Circuit requires nonnegative finite budgets and a nonempty lambda grid.")

    def _spec(self, row):
        task = str(row.get("eval_name", "default"))
        rules = self.cfg.get("tasks", {}).get(self.args["dataset"]["name"], {})
        spec = {"mode": "selection"}
        for pattern, options in rules.items():
            if fnmatchcase(task, pattern):
                spec.update(options)
        mode = spec["mode"]
        if mode == "choice_or_selection":
            # HLE's released single-choice format has an explicit options block.
            # Letters enumerating statements in an open/multi-answer question
            # are not evidence of a single-choice task. Never inspect test gold.
            prompt = str(row.prompt)
            options = _choice_options(prompt)
            explicit_letter = "correct option letter" in prompt.casefold()
            mode = "classification" if len(options) >= 2 or explicit_letter else "selection"
            # A common A-Z head shares support across rare option counts.
            # Feasible terminal labels are still constrained by each input.
            spec.setdefault("classes", list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        if mode not in {"classification", "selection"}:
            raise ValueError(f"Unknown circuit terminal mode: {mode}")
        class_values = spec.get("classes", [])
        if class_values == "from_data":
            class_values = row.get("classes")
            if not isinstance(class_values, (list, tuple)):
                raise ValueError(f"{task}: verified classes metadata is required in the label sidecar.")
        classes = tuple(str(v) for v in class_values) if mode == "classification" else ()
        if mode == "classification" and (len(classes) < 2 or len(set(classes)) != len(classes)):
            raise ValueError(f"Explicit, distinct classes are required for classification task {task}.")
        # Mixed benchmarks contain tiny source tags (e.g. individual ShareGPT IDs).
        # Share a fitted head across compatible tasks; the query still conditions it.
        group_name = task if self.cfg.get("separate_tasks", False) else mode
        return group_name, mode, classes

    def _embed_queries(self, frame):
        modality = self.args["modality"].split("+")
        x = self.embedder.run_embed(texts=frame.prompt.astype(str).tolist() if "text" in modality else None,
                                    images=frame.image_path.tolist() if "image" in modality else None)
        return np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=np.float32)

    @staticmethod
    def _input_groups(frame):
        if "input_group" in frame:
            return frame.input_group
        groups = frame.prompt.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        if "image_path" in frame:
            # Identical generic instructions with different images are different
            # inputs, as in MathVerse. Repeated prompt/image pairs stay grouped.
            groups = groups + "\nimage=" + frame.image_path.astype(str)
        return groups

    @staticmethod
    def _class_mask(frame, classes):
        mask = np.ones((len(frame), len(classes)), dtype=bool)
        for i, prompt in enumerate(frame.prompt.astype(str)):
            options = _choice_options(prompt)
            if len(options) >= 2:
                mask[i] = [c in options for c in classes]
        return mask

    def _embed_responses(self, texts):
        if self._response_embedder is None:
            args = deepcopy(self.args)
            config = dict(args["embeddings"])
            config.update(self.cfg.get("response_embeddings", {}))
            config.update(image_model=None, joint_model=None, training=False, out_dim=None)
            if not config.get("text_model"):
                raise ValueError("Non-classification requires lmcircuit.response_embeddings.text_model.")
            args["embeddings"] = config
            self._response_embedder = Embedder(args)
        x = self._response_embedder.run_embed(texts=texts, images=None)
        return np.asarray(x.detach().cpu() if isinstance(x, torch.Tensor) else x, dtype=np.float32)

    def _matrices(self, frame):
        m = len(self.model_list)
        perf = frame[[f"model_{j}_performance" for j in range(m)]].to_numpy(dtype=np.float32)
        cost = frame[[f"model_{j}_cost" for j in range(m)]].to_numpy(dtype=np.float32)
        if not np.isfinite(perf).all() or not np.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("Circuit replay needs complete finite score/cost records; missing calls are not zero calls.")
        return perf, cost

    def _encode_response(self, row, global_model, group):
        classes = group["classes"]
        if group["mode"] == "classification":
            value = row.get(f"model_{global_model}_prediction")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                value = row.get(f"model_{global_model}_response")
            if value is None or (isinstance(value, float) and np.isnan(value)):
                raise ValueError("Missing recorded classification response; enable circuit fields.")
            return _class_index(value, classes, invalid=True)
        text = self._selection_text(row, global_model)
        if not text or text in {"[]", "{}"}:
            return group["empty_code"]
        vector = self._embed_responses([text])[0]
        return int(np.argmin(((group["centers"] - vector) ** 2).sum(1)))

    def _selection_text(self, row, global_model):
        if row.get(f"model_{global_model}_response_kind") == "choice_scores":
            if not any(fnmatchcase(str(row.get("eval_name", "")), pattern) for pattern in self.cfg.get("score_vector_tasks", [])):
                raise ValueError("Choice likelihoods require classification labels or explicit score-vector selection.")
        return _text(row.get(f"model_{global_model}_response"))

    def train(self):
        self.groups = {}
        frame = self.train_df.reset_index(drop=True)
        x = self._embed_queries(frame)
        perf, cost = self._matrices(frame)
        self._prior_quality, self._prior_cost = perf.mean(0), cost.mean(0)
        keys = [self._spec(row) for _, row in frame.iterrows()]
        max_states = int(self.cfg.get("max_states", 6561))
        for key in dict.fromkeys(keys):
            ids = np.asarray([i for i, k in enumerate(keys) if k == key])
            part = frame.iloc[ids]
            groups = self._input_groups(part)
            if groups.nunique() < 3:
                raise ValueError(f"Task {key[0]} needs at least three distinct input groups for circuit validation.")
            train, val = next(GroupShuffleSplit(n_splits=1, test_size=float(self.cfg.get("validation_fraction", 0.2)),
                                               random_state=self.seed).split(part, groups=groups))
            task, mode, classes = key
            alphabet = len(classes) + 1 if mode == "classification" else int(self.cfg.get("answer_codes", 4)) + 1
            size = min(int(self.cfg.get("pool_size", 4)), len(self.model_list))
            while size > 0 and (alphabet + 1) ** size > max_states:
                size -= 1
            pool = _choose_pool(perf[ids[train]], cost[ids[train]], size)
            space = _StateSpace(len(pool), alphabet, max_states)
            group = {"key": key, "mode": mode, "classes": classes, "pool": pool, "space": space}
            group["source_tasks"] = sorted(set(part.get("eval_name", pd.Series(["default"])).astype(str)))
            group["training_support"] = len(train)
            answers = np.empty((len(ids), len(pool)), dtype=np.int64)
            if mode == "selection":
                texts = []
                positions = []
                empty = []
                for i, (_, row) in enumerate(part.iterrows()):
                    for j, mid in enumerate(pool):
                        text = self._selection_text(row, mid)
                        if text and text not in {"[]", "{}"}:
                            texts.append(text); positions.append((i, j))
                        else:
                            empty.append((i, j))
                if not texts:
                    raise ValueError(f"{task}: no nonempty responses for the answer codebook.")
                vectors = self._embed_responses(texts)
                training_ids = set(train)
                training = np.asarray([i in training_ids for i, _ in positions])
                if not training.any():
                    raise ValueError(f"{task}: no nonempty training responses for the answer codebook.")
                count = min(alphabet - 1, len(np.unique(vectors[training], axis=0)))
                book = KMeans(n_clusters=count, random_state=self.seed, n_init=10).fit(vectors[training])
                # Do not plan through extra code IDs that have no fitted centroid.
                alphabet = count + 1
                group["space"] = space = _StateSpace(len(pool), alphabet, max_states)
                group["centers"], group["empty_code"] = book.cluster_centers_, count
                for (i, j), code in zip(positions, book.predict(vectors)):
                    answers[i, j] = code
                for i, j in empty:
                    answers[i, j] = group["empty_code"]
                targets = perf[ids][:, pool]
            else:
                if "answer" not in part:
                    raise ValueError(f"{task}: classification needs verified answer labels; enable circuit fields/label sidecar.")
                targets = np.asarray([_target_distribution(row, classes) for _, row in part.iterrows()])
                unmapped = int((targets.sum(-1) == 0).sum())
                if unmapped:
                    logging.warning("[LMCircuit] %d choice rows have out-of-alphabet golds; retained with zero class accuracy.", unmapped)
                for i, (_, row) in enumerate(part.iterrows()):
                    for j, mid in enumerate(pool):
                        answers[i, j] = self._encode_response(row, mid, group)
            group["cost_scale"] = max(float(cost[ids[train]].mean(0).max()), 1e-12)
            group["plan_cost"] = cost[ids[train]][:, pool].mean(0) / group["cost_scale"]
            # A training-only affine normalization keeps the query representation stable.
            group["x_mean"] = x[ids[train]].mean(0)
            group["x_std"] = np.maximum(x[ids[train]].std(0), 1e-4)
            normalized = (x[ids] - group["x_mean"]) / group["x_std"]
            group["weights"] = np.ones(len(classes), dtype=np.float32)
            objective = self.cfg.get("classification_objective", self.cfg.get("objective", "accuracy"))
            if objective not in {"accuracy", "balanced_accuracy"}:
                raise ValueError("Circuit classification objective must be accuracy or balanced_accuracy.")
            if mode == "classification" and objective == "balanced_accuracy":
                frequencies = targets[train].mean(0)
                group["weights"] = 1 / (len(classes) * np.maximum(frequencies, 1 / len(train)))
            training_cfg = dict(self.cfg.get("belief", {}))
            training_cfg.update(self.args.get("training", {}))
            fitting_targets = targets
            if mode == "selection":
                # BERTScore and other native benchmark rewards need not be in
                # [0,1]. Fit the bounded head in a training-only affine scale;
                # planning and reported scores retain the original units.
                group["quality_offset"] = float(targets[train].min())
                group["quality_scale"] = max(float(targets[train].max()) - group["quality_offset"], 1e-8)
                fitting_targets = (targets - group["quality_offset"]) / group["quality_scale"]
            group["net"], group["fit"] = _fit_belief(normalized, answers, fitting_targets, train, val, mode,
                len(classes) if mode == "classification" else len(pool), alphabet, training_cfg, self.seed, self.device)
            self.groups[key] = group
            # Only held-out training rows choose budget operating points.
            group["val_cost"], group["val_value"] = self._validation(group, normalized[val], answers[val], targets[val], cost[ids[val]][:, pool],
                self._class_mask(part.iloc[val], classes) if mode == "classification" else None)
            group["points"] = []
            for budget in self.budgets:
                eligible = np.flatnonzero(group["val_cost"] <= budget + 1e-12)
                group["points"].append(None if not len(eligible) else min(eligible, key=lambda k: (-group["val_value"][k], group["val_cost"][k], k)))
            logging.info("[LMCircuit] %s %s: %d train/%d val, pool=%s, states=%d", task, mode, len(train), len(val),
                         [self.model_list[j] for j in pool], space.size)

    @torch.no_grad()
    def _tables_batch(self, group, x, class_mask=None):
        space, net = group["space"], group["net"]
        outputs, transitions = [], []
        chunk = int(self.cfg.get("state_batch_size", 512))
        if chunk < 1:
            raise ValueError("state_batch_size must be positive.")
        features = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        digits = torch.as_tensor(space.digits, device=self.device)
        for start in range(0, len(x) * space.size, chunk):
            ids = torch.arange(start, min(start + chunk, len(x) * space.size), device=self.device)
            out, nxt = net(features[ids // space.size], digits[ids % space.size])
            outputs.append(out.exp() if group["mode"] == "classification" else out)
            transitions.append(nxt.exp())
        out = torch.cat(outputs).view(len(x), space.size, -1)
        if group["mode"] == "selection":
            out = out * group["quality_scale"] + group["quality_offset"]
        nxt = torch.cat(transitions).view(len(x), space.size, space.M, space.A)
        if group["mode"] == "classification":
            weighted = out * torch.as_tensor(group["weights"], device=self.device)
            allowed = weighted
            if class_mask is not None:
                allowed = weighted.masked_fill(~torch.as_tensor(class_mask, device=self.device)[:, None], -torch.inf)
            reward, terminal = allowed.max(-1)
            # Negative weighted risk, not just max(w*p), for imbalanced objectives.
            reward = reward - weighted.sum(-1)
        else:
            available = out.masked_fill((digits == space.A)[None], -torch.inf)
            reward, terminal = available.max(-1)
        _, actions = _solve_dp_batch(space, reward, nxt, group["plan_cost"], self.prices)
        return out.cpu().numpy(), terminal.cpu().numpy(), actions.cpu().numpy()

    def _tables(self, group, x):
        return tuple(v[0] for v in self._tables_batch(group, np.asarray(x)[None]))

    def _table_rows(self, group, x, class_mask=None):
        batch = int(self.cfg.get("query_batch_size", 16))
        if batch < 1:
            raise ValueError("query_batch_size must be positive.")
        for start in range(0, len(x), batch):
            tables = self._tables_batch(group, x[start:start + batch], None if class_mask is None else class_mask[start:start + batch])
            for i in range(len(tables[0])):
                yield tuple(v[i] for v in tables)

    def _validation(self, group, x, answers, targets, costs, class_mask=None):
        values = np.zeros(len(self.prices)); spent = np.zeros_like(values)
        for i, (_, terminal, actions) in enumerate(self._table_rows(group, x, class_mask)):
            for k in range(len(self.prices)):
                pred, asked, _ = _replay(group["space"], actions[k], terminal, lambda j: answers[i, j])
                if group["mode"] == "classification":
                    correct = targets[i, pred] > 0 if targets.ndim == 2 else pred == targets[i]
                    values[k] += float(correct) * group["weights"][pred]
                elif asked:
                    values[k] += targets[i, pred]
                else:
                    # lambda=inf cannot produce an answer in the selection branch.
                    spent[k] = np.inf
                spent[k] += costs[i, asked].sum() / group["cost_scale"]
        return spent / len(x), values / len(x)

    def _evaluation_rows(self, frame, x):
        """Batch only responses that a replayed policy actually requests.

        Restarting a short replay while awaiting an observation keeps the same
        policies and query order, while embedding requested text in batches.
        Unqueried model responses never enter the response encoder.
        """
        class AwaitObservation(Exception):
            pass

        keys = [self._spec(row) for _, row in frame.iterrows()]
        batch = int(self.cfg.get("query_batch_size", 16))
        for key in dict.fromkeys(keys):
            if key not in self.groups:
                raise ValueError(f"No training support for task {key}; out-of-domain transfer is not assumed.")
            group = self.groups[key]
            ids = [i for i, k in enumerate(keys) if k == key]
            for start in range(0, len(ids), batch):
                selected = ids[start:start + batch]
                part = frame.iloc[selected]
                normalized = (x[selected] - group["x_mean"]) / group["x_std"]
                tables = self._tables_batch(group, normalized,
                    self._class_mask(part, group["classes"]) if group["mode"] == "classification" else None)
                rows = [row for _, row in part.iterrows()]
                caches = [{} for _ in rows]
                points = sorted({p for p in group["points"] if p is not None})
                for _ in range(group["space"].M):
                    pending = set()
                    for i, row in enumerate(rows):
                        def observe(j):
                            if j not in caches[i]:
                                pending.add((i, j))
                                raise AwaitObservation
                            return caches[i][j]
                        for point in points:
                            try:
                                _replay(group["space"], tables[2][i, point], tables[1][i], observe)
                            except AwaitObservation:
                                pass
                    if not pending:
                        break
                    positions, texts = [], []
                    for i, j in sorted(pending):
                        mid = int(group["pool"][j])
                        if group["mode"] == "classification":
                            caches[i][j] = self._encode_response(rows[i], mid, group)
                        else:
                            text = self._selection_text(rows[i], mid)
                            if not text or text in {"[]", "{}"}:
                                caches[i][j] = group["empty_code"]
                            else:
                                positions.append((i, j)); texts.append(text)
                    if texts:
                        vectors = self._embed_responses(texts)
                        codes = ((vectors[:, None] - group["centers"][None]) ** 2).sum(-1).argmin(-1)
                        for (i, j), code in zip(positions, codes):
                            caches[i][j] = int(code)
                for i, index in enumerate(selected):
                    yield index, group, tuple(v[i] for v in tables), caches[i]

    @torch.no_grad()
    def predict(self, test_embedding):
        """Input-only reference predictions; evaluate() uses the sequential policy.

        Task-compatible heads are marginalised by training support because this
        legacy interface carries embeddings, not task IDs. Pool estimates use
        learned heads; models outside trained pools retain their training mean.
        No answers/test_df are read; actual policies are evaluated separately.
        """
        x = np.asarray(test_embedding.detach().cpu() if isinstance(test_embedding, torch.Tensor) else test_embedding, dtype=np.float32)
        quality = np.zeros((len(x), len(self.model_list)), dtype=np.float32)
        costs = np.zeros_like(quality)
        for group in self.groups.values():
            weight = group["training_support"] / sum(g["training_support"] for g in self.groups.values())
            part = np.broadcast_to(self._prior_quality, quality.shape).copy()
            price = np.broadcast_to(self._prior_cost, quality.shape).copy()
            net, space = group["net"], group["space"]
            features = torch.as_tensor((x - group["x_mean"]) / group["x_std"], device=self.device)
            states = torch.full((len(x), space.M), space.A, dtype=torch.long, device=self.device)
            out, response = net(features, states)
            for j, mid in enumerate(group["pool"]):
                if group["mode"] == "selection":
                    part[:, mid] = out[:, j].cpu().numpy() * group["quality_scale"] + group["quality_offset"]
                else:
                    expected = torch.zeros(len(x), device=self.device)
                    for answer in range(space.A):
                        observed = states.clone(); observed[:, j] = answer
                        posterior, _ = net(features, observed)
                        expected += response[:, j, answer].exp() * posterior.exp().max(-1).values
                    part[:, mid] = expected.cpu().numpy()
                price[:, mid] = group["plan_cost"][j] * group["cost_scale"]
            quality += weight * part
            costs += weight * price
        return quality, costs

    def evaluate(self):
        """Replay fixed validation-selected policies, then score their final outputs."""
        if not self.groups:
            raise RuntimeError("Train LMCircuit before evaluating.")
        frame = self.test_df.reset_index(drop=True)
        x = self._embed_queries(frame)
        records = [[] for _ in self.budgets]
        for position, (i, group, (belief, terminal, actions), cache) in enumerate(self._evaluation_rows(frame, x)):
            row = frame.iloc[i]
            def observe(j):
                if j not in cache:
                    raise RuntimeError("Batched replay did not resolve a requested observation.")
                return cache[j]
            for b, point in enumerate(group["points"]):
                if point is None:
                    records[b].append(None)
                    continue
                pred, asked, state = _replay(group["space"], actions[point], terminal, observe)
                selected = None
                if group["mode"] == "classification":
                    target_distribution = _target_distribution(row, group["classes"])
                    target = int(target_distribution.argmax()) if target_distribution.any() else -1
                    performance = float(target_distribution[pred] > 0)
                    prediction = group["classes"][pred]
                else:
                    if pred not in asked:
                        raise RuntimeError("Selection head chose an unobserved answer.")
                    selected = int(group["pool"][pred])
                    performance = float(row[f"model_{selected}_performance"])
                    prediction = _text(row[f"model_{selected}_response"]) if self.cfg.get("save_trajectories", True) else None
                models = [int(group["pool"][j]) for j in asked]
                cost = sum(float(row[f"model_{j}_cost"]) for j in models)
                if not np.isfinite(cost) or cost < 0 or not np.isfinite(performance):
                    raise ValueError("Non-finite circuit replay outcome.")
                record = {"sample_id": str(row.get("sample_id", i)), "task": str(row.get("eval_name", "default")), "mode": group["mode"],
                          "prediction": prediction, "selected_model": selected, "queried_models": models,
                          "cost": cost, "normalized_cost": cost / group["cost_scale"], "performance": performance,
                          "lambda_index": int(point)}
                if group["mode"] == "classification":
                    record["belief"] = belief[state].tolist()
                    record["target_index"] = target
                    record["classes"] = list(group["classes"])
                    record["brier"] = float(np.square(belief[state] - target_distribution).sum()) if target >= 0 else None
                    record["accepted_classes"] = [c for c, p in zip(group["classes"], target_distribution) if p > 0]
                records[b].append(record)
            if (position + 1) % 4096 == 0:
                logging.info("[LMCircuit] Replayed %d/%d test queries", position + 1, len(frame))
        curves = []
        for budget, rows in zip(self.budgets, records):
            if any(row is None for row in rows):
                continue  # explicitly infeasible for at least one task, never a fictitious free answer
            point = {"budget": float(budget), "cost": float(np.mean([r["cost"] for r in rows])),
                     "normalized_cost": float(np.mean([r["normalized_cost"] for r in rows])),
                     "performance": float(np.mean([r["performance"] for r in rows])),
                     "calls": float(np.mean([len(r["queried_models"]) for r in rows]))}
            classified = [r for r in rows if r["mode"] == "classification"]
            if classified:
                strata = {}
                for record in classified:
                    if record["target_index"] >= 0:
                        strata.setdefault((tuple(record["classes"]), record["target_index"]), []).append(record["performance"])
                point["classification_accuracy"] = float(np.mean([r["performance"] for r in classified]))
                point["classification_balanced_accuracy"] = float(np.mean([np.mean(v) for v in strata.values()])) if strata else None
                brier = [r["brier"] for r in classified if r["brier"] is not None]
                point["classification_brier"] = float(np.mean(brier)) if brier else None
                point["unmapped_class_targets"] = sum(r["target_index"] < 0 for r in classified)
            curves.append(point)
        if not curves:
            raise ValueError("No feasible circuit budget for all tasks; extend the configured budget range.")
        # Feed realized episode quality and the sum of all paid calls into the
        # unchanged ORBIT metric implementation, including its shared anchor,
        # benchmark cost interval, Peak Score and quality-neutral cost ratio.
        self.cal_metrics([{"cost": p["cost"], "performance": p["performance"]} for p in curves])
        output = Path(self.cfg.get("output_dir", f'./outputs/{self.args["dataset"]["name"]}/{self.args["dataset"]["split"]["mode"]}'))
        output.mkdir(parents=True, exist_ok=True)
        run = f"LMCircuit_{time.time_ns()}"
        report = {"method": "LMCircuit", "seed": self.seed, "curve": curves,
                  "infeasible_budgets": [float(b) for b, rows in zip(self.budgets, records) if any(r is None for r in rows)],
                  "cost_protocol": "sum of original ORBIT model_i_cost; planning means and scale fitted on training only",
                  "metric_protocol": "BaseRouter.cal_metrics, unchanged shared minimum-cost anchor and benchmark cost interval",
                  "evaluation_cost_bounds": list(self._evaluation_cost_bounds()),
                  "tasks": [{"group": k[0], "source_tasks": g["source_tasks"], "mode": k[1], "classes": list(k[2]), "pool": [self.model_list[j] for j in g["pool"]],
                             "states": g["space"].size, "cost_scale": g["cost_scale"], "fit": g["fit"],
                             "approximate_text_planning": k[1] == "selection"} for k, g in self.groups.items()]}
        (output / f"{run}_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        if self.cfg.get("save_trajectories", True):
            with (output / f"{run}_trajectories.jsonl").open("w") as handle:
                for budget, rows in zip(self.budgets, records):
                    for record in rows:
                        if record is not None:
                            handle.write(json.dumps(dict(budget=float(budget), **record), ensure_ascii=False) + "\n")
        self.last_report = report
        logging.info("[LMCircuit] Saved curves, protocol and trajectories: %s/%s", output, run)
        return report
