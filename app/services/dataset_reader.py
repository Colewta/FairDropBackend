"""Leitura pelo conteúdo, com correções estruturais transparentes e sem descartar registros inválidos."""
import csv
import re
from io import BytesIO, StringIO
from pathlib import Path
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile, ZipFile

import pandas as pd
from fastapi import UploadFile

from app.core.config import MAX_COLUMNS, MAX_ROWS, MAX_UPLOAD_BYTES
from app.schemas.dataset import DatasetError


def _header_row(rows: list[list]) -> int:
    for index, row in enumerate(rows[:10]):
        values = [str(v).strip() for v in row if pd.notna(v) and str(v).strip()]
        if not values:
            continue
        # Exportação UCI: linha X1..Xn/Y seguida por nomes reais das variáveis.
        if sum(bool(re.fullmatch(r"[XY]\d*", value)) for value in values) >= max(2, len(values) * .8):
            if index + 1 < len(rows) and all(pd.notna(v) and str(v).strip() for v in rows[index + 1]):
                return index + 1
        return index
    return 0


def _normalize_header(header: list, warnings: list[str]) -> list[str]:
    names = []
    used = set()
    for index, value in enumerate(header):
        base = str(value).replace("\ufeff", "").strip() if pd.notna(value) else ""
        name = base or f"coluna_{index + 1}"
        suffix = 2
        while name in used:
            name = f"{base or f'coluna_{index + 1}'}_{suffix}"
            suffix += 1
        if name != base:
            warnings.append(f"Cabeçalho vazio ou repetido na posição {index + 1}: renomeado para {name}.")
        names.append(name)
        used.add(name)
    return names


def _decode(content: bytes) -> tuple[str, str]:
    encodings = ["utf-8-sig", "cp1252", "latin1"]
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        encodings = ["utf-16"]
    elif content[:1000].count(0) > len(content[:1000]) * .2:
        encodings = ["utf-16-le", "utf-16-be"]
    for encoding in encodings:
        try:
            text = content.decode(encoding)
            if "\x00" not in text:
                return text, encoding
        except UnicodeDecodeError:
            continue
    raise DatasetError("INVALID_ENCODING", "Não conseguimos reconhecer o texto. Exporte a base como CSV ou planilha Excel.")


def _read_csv(content: bytes, warnings: list[str]) -> tuple[pd.DataFrame, dict]:
    text, encoding = _decode(content)
    text = text.lstrip("\r\n")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    if re.fullmatch(r"sep=([,;\t|])", first_line, re.IGNORECASE):
        delimiter = first_line[-1]
        text = text[len(first_line):].lstrip("\r\n")
    else:
        try:
            delimiter = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
        except csv.Error:
            counts = {sep: len(next(csv.reader([first_line], delimiter=sep))) for sep in (",", ";", "\t", "|")}
            delimiter = max(counts, key=counts.get)
    reader = csv.reader(StringIO(text), delimiter=delimiter, strict=True, skipinitialspace=True)
    rows = []
    for row in reader:
        if not row or not any(str(v).strip() for v in row):
            continue
        if len(rows) > MAX_ROWS + 10:
            raise DatasetError("TOO_MANY_ROWS", f"O limite é de {MAX_ROWS} linhas.", 413)
        rows.append(row)
    if not rows:
        raise DatasetError("EMPTY_DATASET", "O arquivo não contém dados.")
    header_index = _header_row(rows)
    header = rows[header_index]
    names = _normalize_header(header, warnings)
    records = rows[header_index + 1:]
    for index, row in enumerate(records, start=header_index + 2):
        if len(row) != len(names):
            raise DatasetError("INVALID_CSV", f"A linha {index} tem {len(row)} campos; o cabeçalho tem {len(names)}. Confira separadores e aspas nesta linha.")
    if header_index:
        warnings.append(f"Cabeçalho reconhecido na linha {header_index + 1}; linhas de apresentação anteriores foram ignoradas.")
    frame = pd.DataFrame(records, columns=names, dtype="string")
    return frame, {"format": "CSV", "encoding": encoding, "delimiter": "tab" if delimiter == "\t" else delimiter, "header_row": header_index + 1}


def read_dataset(content: bytes, filename: str) -> pd.DataFrame:
    if len(content) > MAX_UPLOAD_BYTES:
        raise DatasetError("FILE_TOO_LARGE", f"O limite é de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)
    if not content:
        raise DatasetError("EMPTY_DATASET", "O arquivo está vazio.")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls", ".txt", ".tsv", ".data"}:
        raise DatasetError("INVALID_FILE_TYPE", "Envie CSV, TXT, TSV ou uma planilha Excel.")
    warnings = []
    actual = ".xls" if content.startswith(bytes.fromhex("d0cf11e0a1b11ae1")) else ".xlsx" if content.startswith(b"PK\x03\x04") else ".csv"
    if actual != suffix and (actual != ".csv" or suffix in {".xlsx", ".xls"}):
        warnings.append(f"O conteúdo foi reconhecido como {actual[1:].upper()}, embora o nome termine em {suffix}. A leitura foi ajustada automaticamente.")
    try:
        if actual == ".csv":
            # Extensão Excel com conteúdo inválido não deve virar uma coluna de texto acidentalmente.
            if suffix in {".xls", ".xlsx"} and not any(separator in content[:4096] for separator in (b",", b";", b"\t")):
                raise DatasetError("INVALID_FILE", "A planilha está inválida ou corrompida.")
            df, info = _read_csv(content, warnings)
        else:
            if actual == ".xlsx":
                with ZipFile(BytesIO(content)) as archive:
                    if sum(item.file_size for item in archive.infolist()) > MAX_UPLOAD_BYTES * 10:
                        raise DatasetError("FILE_TOO_LARGE", "A planilha descompactada excede o limite.", 413)
            engine = "openpyxl" if actual == ".xlsx" else "xlrd"
            with pd.ExcelFile(BytesIO(content), engine=engine) as workbook:
                raw = pd.DataFrame()
                sheet = workbook.sheet_names[0]
                for sheet in workbook.sheet_names:
                    raw = pd.read_excel(workbook, sheet_name=sheet, header=None, nrows=MAX_ROWS + 12, dtype=object).dropna(how="all")
                    if not raw.empty:
                        break
            if raw.empty:
                raise DatasetError("EMPTY_DATASET", "A planilha não contém registros.")
            header_index = _header_row(raw.head(10).values.tolist())
            df = raw.iloc[header_index + 1:].copy()
            df.columns = _normalize_header(raw.iloc[header_index].tolist(), warnings)
            info = {"format": actual[1:].upper(), "sheet": sheet, "header_row": header_index + 1}
            if header_index:
                warnings.append(f"Cabeçalho reconhecido na linha {header_index + 1}; linhas de apresentação anteriores foram ignoradas.")
    except DatasetError:
        raise
    except ImportError as exc:
        raise DatasetError("EXCEL_ENGINE_MISSING", "O leitor Excel não está instalado no servidor. Reinstale as dependências do backend.", 503) from exc
    except Exception as exc:
        if isinstance(exc, (ValueError, OSError, BadZipFile, ParseError, csv.Error, pd.errors.ParserError)) or type(exc).__module__.startswith(("xlrd.", "openpyxl.")):
            raise DatasetError("INVALID_FILE", "Não foi possível ler o arquivo. Confira a estrutura ou exporte novamente como CSV ou Excel.") from exc
        raise
    if df.empty:
        raise DatasetError("EMPTY_DATASET", "A base não contém registros abaixo do cabeçalho.")
    if len(df) > MAX_ROWS or len(df.columns) > MAX_COLUMNS:
        raise DatasetError("DATASET_TOO_LARGE", f"Limites: {MAX_ROWS} linhas e {MAX_COLUMNS} colunas.", 413)
    df = df.replace(r"(?i)^\s*(?:nan|null|none|n/a|na|\?)?\s*$", pd.NA, regex=True).reset_index(drop=True)
    empty_columns = [c for c in df.columns if c.startswith("coluna_") and df[c].isna().all()]
    if empty_columns:
        df = df.drop(columns=empty_columns)
        warnings.append(f"{len(empty_columns)} colunas vazias sem nome foram ignoradas.")
    df.attrs.update(import_info=info, import_warnings=warnings)
    return df


def read_upload(file: UploadFile) -> pd.DataFrame:
    return read_dataset(file.file.read(MAX_UPLOAD_BYTES + 1), file.filename or "")
