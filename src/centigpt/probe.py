"""Измерительная оснастка: смотреть, что обученная модель делает с позициями.

Мотив. Про обученные головы легко рассуждать и трудно угадать: какой масштаб
логитов достигается на самом деле, бывают ли головы «предыдущий токен» и
насколько остро они пикуют, сколько массы уходит на нулевую позицию. Здесь
всё это меряется на чекпойнтах из runs/.

`model.py` не трогается. Паттерны достаются подклассом, который подменяется
на живом инстансе — веса и буферы остаются те же самые объекты.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from centigpt.config import Config
from centigpt.model import GPT, CausalMultiheadAttention


class EagerAttention(CausalMultiheadAttention):
    """То же внимание, но с явным softmax и сохранённым паттерном.

    `F.scaled_dot_product_attention` веса внимания наружу не отдаёт — она
    вправе уйти в слитое ядро, где промежуточной матрицы вообще не
    существует. Поэтому здесь та же математика выписана руками.

    Совпадение с оригиналом не предполагается, а проверяется:
    `tests/test_probe.py::test_eager_attention_gives_the_same_logits_as_sdpa`.
    Разойдись они — все дальнейшие числа описывали бы другую модель.
    """

    last_pattern: Tensor
    """(B, H, T, T) после softmax, с последнего прогона."""

    last_scores: Tensor
    """(B, H, T, T) до softmax и без маски: запас пика над соперником виден
    только здесь."""

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.size()
        assert d_model == self.d_model

        shape = (batch_size, seq_len, self.n_heads, self.d_head)
        Q = self.apply_rope(self.W_Q(x).view(shape).transpose(1, 2))
        K = self.apply_rope(self.W_K(x).view(shape).transpose(1, 2))
        V = self.W_V(x).view(shape).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) * self.d_head**-0.5
        future = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)
        pattern = torch.softmax(scores.masked_fill(future, -torch.inf), dim=-1)

        self.last_scores = scores.detach()
        self.last_pattern = pattern.detach()

        out = (pattern @ V).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.W_O(out)


def capture_attention(model: GPT) -> GPT:
    """Включить наблюдаемое внимание, ничего не пересоздавая.

    Подмена `__class__` на живом инстансе меняет только таблицу методов:
    параметры и буфер `freqs_cis` остаются теми же объектами, никакого
    копирования и никакой переинициализации.
    """
    for module in model.modules():
        if isinstance(module, CausalMultiheadAttention):
            module.__class__ = EagerAttention
    return model


def observed_heads(model: GPT) -> list[EagerAttention]:
    """Наблюдаемые головы по слоям — или внятная ошибка, если их нет.

    Молчаливый пустой список читался бы как «модель никуда не смотрит», а это
    крайне запутанный способ узнать, что забыт вызов.
    """
    heads: list[EagerAttention] = []
    for block in model.blocks:
        attn = block.attn
        # TRY004 предлагает здесь TypeError, потому что видит isinstance. Но это
        # не проверка типа аргумента, а проверка состояния: тип правильный,
        # просто модель ещё не переведена в наблюдаемый режим.
        if not isinstance(attn, EagerAttention):
            raise RuntimeError(  # noqa: TRY004
                "паттерны не записываются: вызови capture_attention(model) до прогона"
            )
        heads.append(attn)
    return heads


def attention_patterns(model: GPT) -> list[Tensor]:
    """Паттерны последнего прогона, по слою на элемент."""
    return [head.last_pattern for head in observed_heads(model)]


def attention_scores(model: GPT) -> list[Tensor]:
    """Сырые скорры последнего прогона, по слою на элемент."""
    return [head.last_scores for head in observed_heads(model)]


def load_gpt(checkpoint_path: str | Path, cfg: Config | None = None) -> GPT:
    """Достать GPT из lightning-чекпойнта.

    Lightning хранит GPT внутри своего модуля, поэтому каждый ключ приезжает
    как `model.<имя>`. Градиенты выключаются: здесь бывает только forward.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
    }

    model = GPT(cfg if cfg is not None else Config())
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)
    return model


def token_windows(
    tokens_path: str | Path, n_seqs: int, seq_len: int, seed: int
) -> Tensor:
    """Непрерывные куски настоящего текста — (n_seqs, seq_len) int64.

    Именно непрерывные: перемешанные токены разрушили бы ровно ту
    позиционную структуру, которую мы собрались измерять.
    """
    tokens = np.load(tokens_path, mmap_mode="r")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - seq_len, size=n_seqs)
    rows = np.stack(
        [np.asarray(tokens[s : s + seq_len], dtype=np.int64) for s in starts]
    )
    return torch.from_numpy(rows)


def offset_mass(pattern: Tensor, offset: int) -> Tensor:
    """Средняя масса на диагонали `offset`, по голове на элемент — (H,).

    `offset=-1` — «предыдущий токен». Позиции, у которых такого соседа нет,
    в среднее не попадают: у нулевой позиции предшественника не существует,
    и штрафовать за него голову не за что.
    """
    diagonal = pattern.diagonal(offset=offset, dim1=-2, dim2=-1)
    return diagonal.mean(dim=(0, -1))


def sink_mass(pattern: Tensor) -> Tensor:
    """Средняя масса на нулевой позиции — (H,).

    Нулевая строка исключена: позиция 0 видит только саму себя, её единица
    вынуждена маской и ни о чём не говорит.
    """
    return pattern[:, :, 1:, 0].mean(dim=(0, -1))


def uniform_pattern(seq_len: int, n_heads: int = 1) -> Tensor:
    """Паттерн головы, которая ничего не делает: масса поровну по префиксу.

    Базовая линия для чтения любых измерений. Равномерное внимание само по
    себе кладёт на диагональ `j−1` заметную массу — просто потому, что на
    ранних позициях кандидатов мало. Без этого числа измеренные 0.05
    невозможно отличить от «голова что-то делает».
    """
    allowed = torch.ones(seq_len, seq_len).tril()
    pattern = allowed / allowed.sum(dim=-1, keepdim=True)
    return pattern.expand(1, n_heads, seq_len, seq_len).clone()


def top_margin(scores: Tensor) -> Tensor:
    """Средний зазор между победителем и вторым кандидатом — (H,).

    Та самая величина, от которой зависит острота пика: софтмакс смотрит на
    разности скорров, а не на их абсолютную величину. Нулевая строка не
    учитывается — там кандидат один и соперничать не с кем.
    """
    seq_len = scores.shape[-1]
    future = torch.ones(seq_len, seq_len, dtype=torch.bool).triu(diagonal=1)
    best_two = scores.masked_fill(future, -torch.inf)[:, :, 1:].topk(2, dim=-1).values
    return (best_two[..., 0] - best_two[..., 1]).mean(dim=(0, -1))


@dataclass(frozen=True)
class HeadRow:
    """Одна голова, измеренная со всех сторон, которые нас занимают."""

    layer: int
    head: int
    prev_token: float
    """Масса на j−1. Сравнивать с равномерной базовой линией, не с нулём."""
    two_back: float
    """Масса на j−2: отличает «эта голова про позиции» от «именно про соседа»."""
    self_mass: float
    """Масса на самой себе."""
    sink: float
    """Масса на нулевой позиции — выключатель, если он есть."""
    margin: float
    """Средний зазор победителя над вторым кандидатом, до softmax."""
    max_score: float
    """Наибольший скорр по модулю: до какого масштаба вообще дошло обучение."""


def head_rows(model: GPT, tokens: Tensor, rerun: bool = True) -> list[HeadRow]:
    """Прогнать текст через модель и померить каждую голову.

    `rerun=False` — не считать заново, взять то, что уже лежит в модулях;
    нужно, чтобы можно было подложить паттерн руками и проверить измерялку.
    """
    if rerun:
        capture_attention(model)
        with torch.no_grad():
            model(tokens)

    rows: list[HeadRow] = []
    for layer, head_module in enumerate(observed_heads(model)):
        pattern, scores = head_module.last_pattern, head_module.last_scores
        measured = (
            offset_mass(pattern, -1),
            offset_mass(pattern, -2),
            offset_mass(pattern, 0),
            sink_mass(pattern),
            top_margin(scores),
            scores.abs().amax(dim=(0, 2, 3)),
        )
        rows += [
            HeadRow(layer, head, *(float(value[head]) for value in measured))
            for head in range(pattern.shape[1])
        ]
    return rows
