"""
Persistência de modelos treinados no formato `.rapid`.

Implementa FR012 e §7.8 do contrato do pacote (HUB-BR-005-01): modelo treinado
e metadados são gravados juntos, num único arquivo com nome padronizado
`MODELTYPE-YYYYMMDD-HHMMSS.rapid`, permitindo recuperação posterior para
comparação ou re-execução.

O arquivo é um container ZIP com dois membros:

- `metadata.json` — nome, tipo, timestamp de criação, versão do RAPID,
  versões das bibliotecas usadas no treino e métricas principais. O registro
  de versões existe para detectar incompatibilidade ao carregar o modelo.
- `model.pkl` — o objeto treinado, serializado.

Privacidade (NFR008): o objeto serializado **não** inclui dados a nível de
paciente. Os pipelines preditivos implementam `__getstate__` excluindo o
dataset e os blocos de treino/teste, e `assert_no_patient_data` verifica o
payload antes da gravação.
"""

import io
import json
import pickle
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

RAPID_SUFFIX = ".rapid"

#: Bibliotecas cujas versões são registradas no header, por serem as que
#: podem quebrar o carregamento de um modelo treinado noutra máquina.
_TRACKED_LIBRARIES = ("scikit-learn", "numpy", "pandas", "xgboost", "shap", "imbalanced-learn")


def _library_versions() -> dict:
    versions = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _rapid_version() -> str:
    try:
        return importlib_metadata.version("isaric")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class ModelMetadata:
    """
    Informação essencial de um modelo salvo (contrato §7.10, Tabela 13).

    Attributes:
        name (str): Nome do modelo salvo — por padrão, o nome do arquivo.
        model_type (str): Tipo do modelo, como recebido.
        created_at (str): Data e hora de criação, em ISO 8601.
        metrics (dict): Métricas principais do modelo.
        rapid_version (str): Versão do pacote no momento do treino.
        library_versions (dict): Versões das bibliotecas relevantes.
        threshold (float | None): Ponto de corte de classificação adotado.
        best_params (dict): Hiperparâmetros escolhidos na busca.
        state (dict): Estados do objeto no momento da gravação.
    """

    name: str
    model_type: str
    created_at: str
    metrics: dict = field(default_factory=dict)
    rapid_version: str = ""
    library_versions: dict = field(default_factory=dict)
    threshold: float = None
    best_params: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


def _json_safe(value):
    """Converte numpy/pandas para tipos nativos, para o metadata.json."""
    if hasattr(value, "item") and getattr(value, "size", 1) == 1:
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _contains_frame(value, depth=1):
    """Procura DataFrame/Series no valor, descendo `depth` níveis em dict/list."""
    import pandas as pd

    if isinstance(value, (pd.DataFrame, pd.Series)):
        return True
    if depth <= 0:
        return False
    if isinstance(value, dict):
        return any(_contains_frame(v, depth - 1) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_frame(v, depth - 1) for v in value)
    return False


def assert_no_patient_data(obj):
    """
    Verifica que o payload a ser serializado não carrega dados de paciente.

    Percorre o estado que será efetivamente gravado (`__getstate__`, quando
    definido) procurando DataFrames e Series. Um modelo treinado não precisa
    deles para funcionar, e gravá-los faria o `.rapid` — arquivo feito para ser
    arquivado e compartilhado — transportar dados individuais, contrariando o
    NFR008.

    Atributos que são reconhecidamente **agregados** (tabelas de métricas,
    SHAP agregado, relatório de colinearidade) devem ser declarados em
    `_AGGREGATE_ATTRS` na classe. A checagem é deliberadamente conservadora:
    qualquer DataFrame novo que não esteja declarado faz a gravação falhar,
    forçando uma decisão explícita sobre a natureza daquele dado.

    Raises:
        ValueError: nomeando os atributos que precisam ser excluídos ou
            declarados como agregados.
    """
    state = obj.__getstate__() if hasattr(obj, "__getstate__") else vars(obj)
    if not isinstance(state, dict):
        return

    allowed = set(getattr(obj, "_AGGREGATE_ATTRS", ()))
    offending = [
        name for name, value in state.items()
        if name not in allowed and _contains_frame(value)
    ]
    if offending:
        raise ValueError(
            "O modelo não pode ser salvo: os atributos "
            f"{sorted(offending)} contêm dados tabulares não declarados como "
            "agregados (NFR008). Exclua-os em __getstate__ ou, se forem "
            "agregados, declare-os em _AGGREGATE_ATTRS."
        )


def build_metadata(model, name: str, metrics: dict = None) -> ModelMetadata:
    """Monta o header de metadados a partir de um pipeline RAPID treinado."""
    return ModelMetadata(
        name=name,
        model_type=type(model).__name__,
        created_at=datetime.now().isoformat(timespec="seconds"),
        metrics=_json_safe(metrics or {}),
        rapid_version=_rapid_version(),
        library_versions=_library_versions(),
        threshold=_json_safe(getattr(model, "threshold_", None)),
        best_params=_json_safe(getattr(model, "best_params_", {}) or {}),
        state=getattr(model, "state", {}),
    )


def default_filename(model_type: str, moment: datetime = None) -> str:
    """Nome padronizado do contrato: MODELTYPE-YYYYMMDD-HHMMSS.rapid"""
    moment = moment or datetime.now()
    return f"{model_type}-{moment:%Y%m%d-%H%M%S}{RAPID_SUFFIX}"


def save_model(model, directory=".", name: str = None, metrics: dict = None) -> str:
    """
    Grava um modelo treinado no formato `.rapid`.

    Args:
        model: Pipeline RAPID treinado.
        directory (str or Path): Diretório de destino.
        name (str, optional): Nome do arquivo. Se omitido, usa a convenção
            `MODELTYPE-YYYYMMDD-HHMMSS.rapid`.
        metrics (dict, optional): Métricas principais gravadas no header.

    Returns:
        str: Caminho do arquivo gravado.

    Raises:
        ValueError: se o payload contiver dados a nível de paciente.
        FileExistsError: se o arquivo de destino já existir.
    """
    assert_no_patient_data(model)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    filename = name or default_filename(type(model).__name__)
    if not filename.endswith(RAPID_SUFFIX):
        filename += RAPID_SUFFIX
    path = directory / filename

    if path.exists():
        raise FileExistsError(f"Já existe um modelo salvo em {path}.")

    metadata = build_metadata(model, name=filename, metrics=metrics)
    payload = pickle.dumps(model)

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", json.dumps(metadata.to_dict(), indent=2, ensure_ascii=False))
        archive.writestr("model.pkl", payload)

    return str(path)


def read_metadata(path) -> ModelMetadata:
    """Lê apenas o header de um `.rapid`, sem desserializar o modelo."""
    with zipfile.ZipFile(path) as archive:
        return ModelMetadata.from_dict(json.loads(archive.read("metadata.json")))


def load_model(path):
    """
    Carrega um modelo salvo.

    Returns:
        tuple: (modelo, ModelMetadata).

    Nota: o objeto carregado não traz os dados de treino/teste — por desenho
    (NFR008). Serve para inferência, comparação e auditoria dos metadados, não
    para re-executar validações que dependam do dataset original.
    """
    with zipfile.ZipFile(path) as archive:
        metadata = ModelMetadata.from_dict(json.loads(archive.read("metadata.json")))
        model = pickle.load(io.BytesIO(archive.read("model.pkl")))
    return model, metadata


class RAPID_Decide:
    """
    Seleção de quais modelos treinados compõem o relatório final
    (contrato §7.10, Tabela 12).

    Importante: `Decide` **não** escolhe o melhor modelo por métrica. A escolha
    é do pesquisador — a classe apenas lista o que foi salvo e registra a
    seleção. Comparar automaticamente algoritmos de tipos diferentes é
    justamente o que o FR013 restringe.

    Args:
        directory (str or Path): Diretório onde estão os arquivos `.rapid`.
    """

    def __init__(self, directory="."):
        self.directory = Path(directory)
        self.selected = []

    def _load_metadata(self):
        return [read_metadata(p) for p in sorted(self.directory.glob(f"*{RAPID_SUFFIX}"))]

    def list(self):
        """Lista os metadados de todos os modelos salvos no diretório."""
        return self._load_metadata()

    def _validate_params(self, include, exclude, available):
        if include and exclude:
            raise ValueError("Use `include` ou `exclude`, não os dois ao mesmo tempo.")
        unknown = sorted(set(include or []) | set(exclude or []) - set(available))
        unknown = [n for n in unknown if n not in available]
        if unknown:
            raise ValueError(f"Modelo(s) não encontrado(s) em {self.directory}: {unknown}")

    def execute(self, include=None, exclude=None):
        """
        Registra quais modelos entram no relatório.

        Args:
            include (list, optional): Nomes a incluir. Sem `include` nem
                `exclude`, todos os modelos salvos são selecionados.
            exclude (list, optional): Nomes a remover da seleção.

        Returns:
            list: Metadados dos modelos selecionados.
        """
        metadata = self._load_metadata()
        available = [m.name for m in metadata]
        self._validate_params(include, exclude, available)

        if include:
            self.selected = [m for m in metadata if m.name in set(include)]
        elif exclude:
            self.selected = [m for m in metadata if m.name not in set(exclude)]
        else:
            self.selected = metadata

        return self.selected
