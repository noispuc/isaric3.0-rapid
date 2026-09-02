"""
Controle de estados dos pipelines RAPID.

Implementa os estados `is_fitted`, `is_decided` e `is_validated` previstos no
contrato do pacote (HUB-BR-005-01, §7.1) e detalhados na revisão de
alinhamento do MVP preditivo (Reqs 3-18).

Racional: métodos como `fit`, `decide` e `validation` modificam o estado do
objeto; `summary`, `report` e `save` apenas leem resultados. Explicitar os
estados evita que o usuário chame um método antes de o objeto estar pronto —
e, no caso de `save`, evita persistir um modelo que ainda não foi escolhido
deliberadamente pelo pesquisador.
"""

from functools import wraps


class RAPIDStateError(RuntimeError):
    """Erro de sequência: um método foi chamado antes do estado necessário."""


#: Mensagens de orientação por estado ausente. O texto de `is_decided` segue
#: a redação especificada no Req 9.
_STATE_HINTS = {
    "is_fitted": "the model was not yet trained. Please call the `fit` method first",
    "is_decided": (
        "the choice on the model was not yet called. "
        "Please specify it before with `decide` method"
    ),
    "is_validated": (
        "the model was not yet validated. "
        "Please run the `validation` method before"
    ),
}


class RAPID_StateMixin:
    """
    Mixin que adiciona controle de estados a um pipeline RAPID.

    Os estados são expostos como propriedades somente-leitura: só mudam pelos
    métodos internos `_mark_*`, chamados ao final de uma execução bem-sucedida.

    A ordem `validation` → `decide` **não** é forçada: o contrato deixa essa
    dinâmica em aberto, e forçá-la impediria o refinamento iterativo que a
    metodologia encoraja. Quem precisar da garantia mais estrita usa
    `save(require_validation=True)`.
    """

    _STATE_NAMES = ("is_fitted", "is_decided", "is_validated")

    def _init_states(self):
        self._states = dict.fromkeys(self._STATE_NAMES, False)

    @property
    def _state_dict(self):
        if not hasattr(self, "_states"):
            self._init_states()
        return self._states

    @property
    def is_fitted(self) -> bool:
        """True depois de `fit()` concluir com sucesso."""
        return self._state_dict["is_fitted"]

    @property
    def is_decided(self) -> bool:
        """True depois de `decide()` concluir com sucesso."""
        return self._state_dict["is_decided"]

    @property
    def is_validated(self) -> bool:
        """True depois de `validation()` concluir com sucesso."""
        return self._state_dict["is_validated"]

    @property
    def state(self) -> dict:
        """Cópia do dicionário de estados, para inspeção e para o relatório."""
        return dict(self._state_dict)

    # ------------------------------------------------------------------
    # Transições
    # ------------------------------------------------------------------

    def _mark_fitted(self):
        """
        Marca o objeto como treinado e **reverte os estados a jusante**.

        Re-treinar invalida qualquer escolha e qualquer validação anteriores:
        elas se referiam a outro modelo. A metodologia encoraja voltar a
        etapas anteriores, então o refit é permitido — mas não pode deixar
        para trás um `is_decided` que não corresponde mais ao modelo atual.
        """
        self._state_dict["is_fitted"] = True
        self._state_dict["is_decided"] = False
        self._state_dict["is_validated"] = False

    def _mark_decided(self):
        self._state_dict["is_decided"] = True

    def _mark_validated(self):
        self._state_dict["is_validated"] = True


def requires_state(*states, mode: str = "any"):
    """
    Exige que o objeto esteja em determinado(s) estado(s) antes de executar.

    Args:
        *states: Nomes de estado exigidos (ex.: 'is_fitted').
        mode (str): 'any' — basta um dos estados (usado por `summary`/`report`,
            que podem rodar em qualquer ponto após o treino); 'all' — todos são
            necessários (usado por `save` quando a validação é exigida).

    Raises:
        RAPIDStateError: com orientação sobre qual método chamar antes.
    """
    if mode not in ("any", "all"):
        raise ValueError(f"mode deve ser 'any' ou 'all', recebido '{mode}'.")

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            satisfied = [bool(getattr(self, state)) for state in states]
            ok = any(satisfied) if mode == "any" else all(satisfied)
            if not ok:
                missing = [s for s, done in zip(states, satisfied) if not done]
                hints = "; ".join(_STATE_HINTS.get(s, s) for s in missing)
                raise RAPIDStateError(
                    f"`{func.__name__}` requires {mode} of {list(states)}: {hints}."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
