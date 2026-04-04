from __future__ import annotations

import json
import secrets
import time
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
from xml.etree import ElementTree as ET


XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}

_PREVIEW_TTL_SECONDS = 60 * 30
_PREVIEW_STORE: dict[str, dict] = {}

# Excel epoch: January 1, 1900 (with the off-by-one Lotus 1-2-3 bug)
_EXCEL_EPOCH = datetime(1899, 12, 30)


def _excel_serial_to_date(value: str) -> str:
    """Convert an Excel serial date number to ISO date string (YYYY-MM-DD).
    If the value is already a date string or not a valid serial, return as-is."""
    if not value:
        return value
    try:
        serial = float(value)
        # Valid Excel date serials are roughly 1 (1900-01-01) to 2958465 (9999-12-31)
        if 1 <= serial <= 2958465:
            return (_EXCEL_EPOCH + timedelta(days=serial)).strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        pass
    return value


def _excel_serial_to_time(value: str) -> str:
    """Convert an Excel fractional serial number to a time string (e.g. '6:05 AM').
    Excel stores times as fractions of a day (0.25 = 6:00 AM, 0.75 = 6:00 PM).
    If the value is already a time string or not a valid fraction, return as-is."""
    if not value:
        return value
    try:
        fraction = float(value)
        if 0 <= fraction < 1:
            total_minutes = round(fraction * 24 * 60)
            hours = total_minutes // 60
            minutes = total_minutes % 60
            period = "AM" if hours < 12 else "PM"
            display_hour = hours % 12 or 12
            return f"{display_hour}:{minutes:02d} {period}"
    except (ValueError, TypeError):
        pass
    return value


def _format_humidity(value: str) -> str:
    """Convert Excel decimal humidity (0-1) to percentage string.
    E.g. '0.6' → '60%', '0.75' → '75%'. Already-formatted values pass through."""
    if not value or "%" in value:
        return value
    try:
        num = float(value)
        if 0 < num <= 1:
            pct = round(num * 100)
            return f"{pct}%"
    except (ValueError, TypeError):
        pass
    return value


def _format_temperature(value: str) -> str:
    """Replace 'OC' with '°C' in temperature strings.
    E.g. '26/34OC' → '26/34°C'."""
    if not value:
        return value
    import re
    return re.sub(r'(?i)(\d)\s*OC\b', r'\1°C', value)


def _cleanup_preview_store() -> None:
    now = time.time()
    expired = [token for token, item in _PREVIEW_STORE.items() if now - item["created_at"] > _PREVIEW_TTL_SECONDS]
    for token in expired:
        _PREVIEW_STORE.pop(token, None)


def store_preview_payload(payload: dict) -> str:
    _cleanup_preview_store()
    token = secrets.token_urlsafe(24)
    _PREVIEW_STORE[token] = {"created_at": time.time(), "payload": payload}
    return token


def get_preview_payload(token: str) -> dict | None:
    _cleanup_preview_store()
    item = _PREVIEW_STORE.get(token)
    return item["payload"] if item else None


def discard_preview_payload(token: str) -> None:
    _PREVIEW_STORE.pop(token, None)


def parse_xlsx_workbook(contents: bytes) -> dict:
    try:
        archive = zipfile.ZipFile(BytesIO(contents))
    except zipfile.BadZipFile as exc:
        raise ValueError("Only .xlsx workbook parsing is supported by the backend preview flow.") from exc

    shared_strings = _parse_shared_strings(archive)
    fills = _parse_fill_colors(archive)
    sheet_refs = _resolve_sheet_refs(archive)
    sheets = [_parse_sheet(archive, shared_strings, fills, ref) for ref in sheet_refs]
    return {
        "sheetCount": len(sheets),
        "sheetNames": [sheet["name"] for sheet in sheets],
        "sheets": sheets,
    }


def build_advisory_preview_payload(contents: bytes, metadata: dict, advisory_type: str) -> dict:
    workbook = parse_xlsx_workbook(contents)
    parser = _build_poultry_preview if advisory_type == "poultry-advisory" else _build_agromet_preview
    extracted = parser(workbook["sheets"], metadata)
    parse_token = store_preview_payload(
        {
            "entityType": advisory_type,
            "metadata": metadata,
            "workbook": workbook,
            "extracted": extracted,
        }
    )
    return {
        "parseToken": parse_token,
        "entityType": advisory_type,
        "workbookMeta": {
            "sheetCount": workbook["sheetCount"],
            "sheetNames": workbook["sheetNames"],
        },
        "parsedSheets": [_sheet_summary(sheet) for sheet in workbook["sheets"]],
        "extracted": extracted,
        "warnings": extracted["warnings"],
        "errors": extracted["errors"],
    }


def build_calendar_preview_payload(contents: bytes, metadata: dict, calendar_type: str) -> dict:
    workbook = parse_xlsx_workbook(contents)
    extracted = _build_poultry_calendar_preview(workbook["sheets"], metadata) if calendar_type == "poultry-calendar" else _build_crop_calendar_preview(workbook["sheets"], metadata)
    parse_token = store_preview_payload(
        {
            "entityType": calendar_type,
            "metadata": metadata,
            "workbook": workbook,
            "extracted": extracted,
        }
    )
    return {
        "parseToken": parse_token,
        "entityType": calendar_type,
        "workbookMeta": {
            "sheetCount": workbook["sheetCount"],
            "sheetNames": workbook["sheetNames"],
        },
        "parsedSheets": [_sheet_summary(sheet) for sheet in workbook["sheets"]],
        "extracted": extracted,
        "warnings": extracted["warnings"],
        "errors": extracted["errors"],
    }


def build_committed_calendar_payload(preview_payload: dict) -> dict:
    return json.loads(json.dumps(preview_payload["extracted"]))


def build_committed_advisory_payload(preview_payload: dict, selected_sheets: list[str] | None = None) -> dict:
    extracted = json.loads(json.dumps(preview_payload["extracted"]))
    allowed = set(selected_sheets or [])
    if allowed:
        extracted["sheets"] = [sheet for sheet in extracted["sheets"] if sheet["name"] in allowed]
        extracted["activities"] = [item for item in extracted["activities"] if item["sheetName"] in allowed]
        extracted["advisories"] = [item for item in extracted["advisories"] if item["sheetName"] in allowed]
        if "parsedActivities" in extracted:
            extracted["parsedActivities"] = [a for a in extracted["parsedActivities"] if a["activity"] in allowed]
        extracted["selectedSheets"] = list(allowed)
    else:
        extracted["selectedSheets"] = [sheet["name"] for sheet in extracted["sheets"]]

    extracted["totalRecords"] = sum(sheet["totalRows"] for sheet in extracted["sheets"])
    return extracted


def _sheet_summary(sheet: dict) -> dict:
    return {
        "name": sheet["name"],
        "headers": sheet["headers"],
        "totalRows": sheet["totalRows"],
        "summary": {
            "totalRows": sheet["totalRows"],
            "columns": len(sheet["headers"]),
            "filledCells": sheet["filledCells"],
            "colorsDetected": sorted({color for row in sheet["colors"] for color in row if color}),
            "sections": sheet["sections"],
        },
        "sampleData": sheet["rows"][:3],
    }


def _parse_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: list[str] = []
    for item in root.findall("main:si", XML_NS):
        text_parts = [node.text or "" for node in item.findall(".//main:t", XML_NS)]
        values.append("".join(text_parts))
    return values


def _parse_theme_colors(archive: zipfile.ZipFile) -> list[str]:
    """Extract theme colors from xl/theme/theme1.xml.

    Returns a list indexed by Excel's theme color index:
      0=lt1, 1=dk1, 2=lt2, 3=dk2, 4-9=accent1-6, 10=hlink, 11=folHlink.
    """
    theme_path = "xl/theme/theme1.xml"
    if theme_path not in archive.namelist():
        return []
    draw_ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    root = ET.fromstring(archive.read(theme_path))
    clr_scheme = root.find(f".//{{{draw_ns}}}clrScheme")
    if clr_scheme is None:
        return []

    # XML order: dk1, lt1, dk2, lt2, accent1-6, hlink, folHlink
    xml_order = ["dk1", "lt1", "dk2", "lt2",
                 "accent1", "accent2", "accent3", "accent4", "accent5", "accent6",
                 "hlink", "folHlink"]
    xml_colors: dict[str, str] = {}
    for name in xml_order:
        el = clr_scheme.find(f"{{{draw_ns}}}{name}")
        if el is None:
            continue
        srgb = el.find(f"{{{draw_ns}}}srgbClr")
        sys_clr = el.find(f"{{{draw_ns}}}sysClr")
        if srgb is not None:
            xml_colors[name] = srgb.attrib.get("val", "")
        elif sys_clr is not None:
            xml_colors[name] = sys_clr.attrib.get("lastClr", "")

    # Excel theme index: 0=lt1, 1=dk1, 2=lt2, 3=dk2, 4+=accent1...
    excel_order = ["lt1", "dk1", "lt2", "dk2",
                   "accent1", "accent2", "accent3", "accent4", "accent5", "accent6",
                   "hlink", "folHlink"]
    return [xml_colors.get(name, "") for name in excel_order]


def _resolve_color_element(el: ET.Element | None, theme_colors: list[str]) -> str | None:
    """Resolve a fgColor/bgColor element to a hex color string, handling rgb, theme, and indexed attrs."""
    if el is None:
        return None
    # Direct rgb attribute
    rgb = el.attrib.get("rgb")
    if rgb:
        return f"#{rgb[-6:]}"
    # Theme reference
    theme_idx = el.attrib.get("theme")
    if theme_idx is not None:
        try:
            idx = int(theme_idx)
            if 0 <= idx < len(theme_colors) and theme_colors[idx]:
                return f"#{theme_colors[idx]}"
        except (ValueError, IndexError):
            pass
    return None


_WHITE_COLORS = {"#FFFFFF", "#ffffff"}


def _parse_fill_colors(archive: zipfile.ZipFile) -> dict[int, str | None]:
    """Build a mapping from cell style index (the 's' attribute) to fill color.

    Excel indirection: cell @s -> cellXfs/xf @fillId -> fills/fill -> color.
    Resolves theme colors and falls back to bgColor when fgColor is white.
    """
    if "xl/styles.xml" not in archive.namelist():
        return {}
    root = ET.fromstring(archive.read("xl/styles.xml"))
    theme_colors = _parse_theme_colors(archive)

    # Step 1: parse fill index -> color
    fill_colors: dict[int, str | None] = {}
    fills_el = root.find("main:fills", XML_NS)
    if fills_el is not None:
        for index, fill in enumerate(fills_el.findall("main:fill", XML_NS)):
            pattern = fill.find("main:patternFill", XML_NS)
            if pattern is None:
                fill_colors[index] = None
                continue
            pattern_type = pattern.attrib.get("patternType", "")
            if pattern_type in ("none", "gray125"):
                fill_colors[index] = None
                continue
            fg = _resolve_color_element(pattern.find("main:fgColor", XML_NS), theme_colors)
            # For solid fills the fgColor fully covers the cell; bgColor is
            # only a residual value that is never visible, so ignore it.
            if pattern_type == "solid":
                fill_colors[index] = fg if (fg and fg not in _WHITE_COLORS) else None
            else:
                bg = _resolve_color_element(pattern.find("main:bgColor", XML_NS), theme_colors)
                if fg and fg not in _WHITE_COLORS:
                    fill_colors[index] = fg
                elif bg and bg not in _WHITE_COLORS:
                    fill_colors[index] = bg
                else:
                    fill_colors[index] = None

    # Step 2: map style index -> fill color via cellXfs
    style_to_color: dict[int, str | None] = {}
    cell_xfs = root.find("main:cellXfs", XML_NS)
    if cell_xfs is not None:
        for xf_index, xf in enumerate(cell_xfs.findall("main:xf", XML_NS)):
            fill_id = int(xf.attrib.get("fillId", "0") or "0")
            style_to_color[xf_index] = fill_colors.get(fill_id)
    else:
        style_to_color = fill_colors

    return style_to_color


def _resolve_sheet_refs(archive: zipfile.ZipFile) -> list[dict]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map: dict[str, str] = {}
    for rel in rels.findall("pkgrel:Relationship", XML_NS):
        rel_map[rel.attrib["Id"]] = rel.attrib["Target"]

    refs: list[dict] = []
    for sheet in workbook.findall("main:sheets/main:sheet", XML_NS):
        rel_id = sheet.attrib.get(f"{{{XML_NS['rel']}}}id")
        target = rel_map.get(rel_id, "")
        target = target.lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"
        refs.append({"name": sheet.attrib.get("name", "Sheet"), "path": target})
    return refs


def _parse_sheet(archive: zipfile.ZipFile, shared_strings: list[str], fills: dict[int, str | None], ref: dict) -> dict:
    root = ET.fromstring(archive.read(ref["path"]))
    rows: list[list[str]] = []
    colors: list[list[str | None]] = []
    max_cols = 0
    filled_cells = 0

    for row in root.findall("main:sheetData/main:row", XML_NS):
        row_cells: list[str] = []
        row_colors: list[str | None] = []
        current_col = 0
        for cell in row.findall("main:c", XML_NS):
            cell_ref = cell.attrib.get("r", "")
            col_index = _column_index_from_ref(cell_ref)
            while current_col < col_index:
                row_cells.append("")
                row_colors.append(None)
                current_col += 1
            value = _read_cell_value(cell, shared_strings).strip()
            style_index = int(cell.attrib.get("s", "0") or "0")
            fill_color = fills.get(style_index)
            row_cells.append(value)
            row_colors.append(fill_color)
            current_col += 1
            if value:
                filled_cells += 1
        max_cols = max(max_cols, len(row_cells))
        rows.append(row_cells)
        colors.append(row_colors)

    normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
    normalized_colors = [row + [None] * (max_cols - len(row)) for row in colors]
    non_empty_rows = [row for row in normalized_rows if any(str(value).strip() for value in row)]
    headers = non_empty_rows[0] if non_empty_rows else []
    sections = _detect_sections(non_empty_rows)
    return {
        "name": ref["name"],
        "headers": headers,
        "rows": non_empty_rows[1:] if len(non_empty_rows) > 1 else [],
        "rawRows": non_empty_rows,
        "colors": normalized_colors,
        "filledCells": filled_cells,
        "totalRows": max(0, len(non_empty_rows) - 1),
        "sections": sections,
    }


def _column_index_from_ref(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - 64)
    return max(index - 1, 0)


def _read_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        texts = [node.text or "" for node in cell.findall(".//main:t", XML_NS)]
        return "".join(texts)

    value_node = cell.find("main:v", XML_NS)
    if value_node is None or value_node.text is None:
        return ""
    raw = value_node.text
    if cell_type == "s":
        index = int(raw)
        return shared_strings[index] if 0 <= index < len(shared_strings) else ""
    return raw


def _detect_sections(rows: list[list[str]]) -> list[str]:
    sections: list[str] = []
    for row in rows:
        for cell in row:
            text = str(cell or "").strip()
            if text.startswith("[") and text.endswith("]") and len(text) > 2:
                sections.append(text.strip("[]"))
    return sections


def _parse_agromet_sheet(rows: list[list[str]], sheet_name: str) -> dict:
    """Parse a single agromet advisory sheet matching the Excel template structure.

    Expected row structure:
      Row 1: [ZONE], [REGION], [DISTRICT], [MONTH/YEAR], [WEEK], [START DATE], [END DATE], [CROP]
      Row 2: metadata values
      Row 3: (empty)
      Row 4: (empty), [RAINFALL], [TEMP], [HUMIDITY], [SOIL MOISTURE], [SOIL TEMP], [SUNSHINE INTENSITY], [SUNRISE], [SUNSET], [EVAPO-TRANSP.]
      Row 5: [FORECAST], values...
      Row 6: [IMPLICATION], values...
      Row 7: [ADVISORY], values...
      Row 8: [FORECAST AND ADVISORY TITLE], title text
      Row 9: [FORECAST AND ADVISORY BODY], body text
      Row 10: [SMS/TEXT], sms text
    """
    result = {
        "activity": sheet_name,
        "metadata": {},
        "weatherParameters": [],
        "forecast": {},
        "implication": {},
        "advisory": {},
        "summaryTitle": "",
        "summaryBody": "",
        "smsText": "",
    }

    # Build a lookup of rows by their [LABEL] in column A
    label_rows: dict[str, list[str]] = {}
    param_header_row: list[str] = []

    for row in rows:
        if not row:
            continue
        col_a = str(row[0]).strip()
        col_a_clean = col_a.strip("[] ").upper()

        # Detect the parameter header row (Row 4): column A is empty or has no label, columns B+ have [PARAM] names
        if not col_a or col_a_clean == "":
            has_params = any(
                str(cell).strip().startswith("[") and str(cell).strip().endswith("]")
                for cell in row[1:] if str(cell).strip()
            )
            if has_params and not param_header_row:
                param_header_row = [str(cell).strip().strip("[] ") for cell in row[1:] if str(cell).strip()]

        # Map labeled rows
        if col_a.startswith("[") and col_a.endswith("]"):
            label_rows[col_a_clean] = row

    # Extract metadata from Row 1 (headers) and Row 2 (values)
    header_labels = ["ZONE", "REGION", "DISTRICT", "MONTH/YEAR", "WEEK", "START DATE", "END DATE", "CROP"]
    meta_row = None
    for row in rows:
        if not row:
            continue
        col_a = str(row[0]).strip().strip("[] ").upper()
        if col_a == "ZONE":
            # This is the header row, next non-empty row is the values
            idx = rows.index(row)
            if idx + 1 < len(rows):
                meta_row = rows[idx + 1]
            break

    if meta_row:
        for i, label in enumerate(header_labels):
            if i < len(meta_row):
                val = str(meta_row[i]).strip()
                # Convert Excel serial dates for date fields
                if label in ("START DATE", "END DATE"):
                    val = _excel_serial_to_date(val)
                result["metadata"][label.lower().replace("/", "_").replace(" ", "_")] = val

    # Extract weather parameters from header row
    result["weatherParameters"] = param_header_row

    # Extract forecast, implication, advisory values
    for label_key, field_name in [("FORECAST", "forecast"), ("IMPLICATION", "implication"), ("ADVISORY", "advisory")]:
        row = label_rows.get(label_key, [])
        values = [str(cell).strip() for cell in row[1:]] if len(row) > 1 else []
        for i, param in enumerate(param_header_row):
            val = values[i] if i < len(values) else "-"
            param_upper = param.upper()
            if label_key == "FORECAST":
                if param_upper in ("SUNRISE", "SUNSET"):
                    val = _excel_serial_to_time(val)
                elif param_upper in ("HUMIDITY",):
                    val = _format_humidity(val)
            # Fix temperature unit across all rows
            if param_upper in ("TEMP", "TEMPERATURE"):
                val = _format_temperature(val)
            result[field_name][param] = val

    # Extract summary title and body (fix temperature units in text)
    title_row = label_rows.get("FORECAST AND ADVISORY TITLE", [])
    if len(title_row) > 1:
        result["summaryTitle"] = _format_temperature(str(title_row[1]).strip())

    body_row = label_rows.get("FORECAST AND ADVISORY BODY", [])
    if len(body_row) > 1:
        result["summaryBody"] = _format_temperature(str(body_row[1]).strip())

    # Extract SMS text (fix temperature units in text)
    sms_row = label_rows.get("SMS/TEXT", label_rows.get("SMS", []))
    if len(sms_row) > 1:
        result["smsText"] = _format_temperature(str(sms_row[1]).strip())

    return result


def _build_agromet_preview(sheets: list[dict], metadata: dict) -> dict:
    activities = []
    warnings: list[str] = []
    parsed_activities: list[dict] = []

    for sheet in sheets:
        rows = sheet["rawRows"]
        parsed = _parse_agromet_sheet(rows, sheet["name"])
        parsed_activities.append(parsed)

        activities.append({
            "sheetName": sheet["name"],
            "activity": sheet["name"],
            "weekLabel": parsed["metadata"].get("week", metadata.get("week") or ""),
        })

    # Use metadata from first sheet if available
    first_parsed = parsed_activities[0] if parsed_activities else {}
    first_meta = first_parsed.get("metadata", {})

    summary = metadata.get("description") or f"Parsed {len(activities)} advisory activities from {len(sheets)} worksheet(s)."
    return {
        "title": metadata.get("title") or "Agromet Advisory",
        "description": metadata.get("description") or "",
        "regionCode": metadata.get("regionCode") or first_meta.get("region", ""),
        "districtCode": metadata.get("districtCode") or first_meta.get("district", ""),
        "commodityCode": metadata.get("commodityCode") or "",
        "crop": metadata.get("commodityName") or metadata.get("commodityCode") or first_meta.get("crop", ""),
        "weatherForecast": {
            "source": "spreadsheet",
            "parameters": first_parsed.get("weatherParameters", []),
        },
        "summary": summary,
        "sheets": [_sheet_summary(sheet) for sheet in sheets],
        "activities": activities,
        "parsedActivities": parsed_activities,
        "advisories": [{"sheetName": a["activity"], "text": a.get("summaryBody", "")} for a in parsed_activities],
        "warnings": warnings,
        "errors": [],
        "totalRecords": sum(sheet["totalRows"] for sheet in sheets),
    }


def _build_poultry_preview(sheets: list[dict], metadata: dict) -> dict:
    management_metrics = {}
    activities = []
    advisories = []
    warnings: list[str] = []

    for sheet in sheets:
        rows = sheet["rawRows"]
        management_metrics.update(_extract_key_value_rows(rows, ["parameter", "recommended value"]))
        sheet_activities = _extract_named_items(rows, ["production stage", "stage", "phase"], sheet["name"])
        if not sheet_activities:
            sheet_activities = [{"sheetName": sheet["name"], "activity": sheet["name"], "weekLabel": metadata.get("week") or ""}]
            warnings.append(f'No explicit production-stage rows found in sheet "{sheet["name"]}". Using the sheet name as the stage label.')
        activities.extend(sheet_activities)
        advisories.extend(_extract_section_items(rows, ["recommendation", "recommended action", "action required"], sheet["name"]))

    summary = metadata.get("description") or f"Parsed {len(activities)} poultry management stages from {len(sheets)} worksheet(s)."
    return {
        "title": metadata.get("title") or "Poultry Advisory",
        "description": metadata.get("description") or "",
        "regionCode": metadata.get("regionCode") or "",
        "districtCode": metadata.get("districtCode") or "",
        "poultryTypeCode": metadata.get("poultryTypeCode") or "",
        "breedCode": metadata.get("breedCode") or "",
        "managementMetrics": management_metrics,
        "summary": summary,
        "sheets": [_sheet_summary(sheet) for sheet in sheets],
        "activities": activities,
        "advisories": advisories or [{"sheetName": sheets[0]["name"] if sheets else "", "text": summary}],
        "warnings": warnings,
        "errors": [],
        "totalRecords": sum(sheet["totalRows"] for sheet in sheets),
    }


def _build_crop_calendar_preview(sheets: list[dict], metadata: dict) -> dict:
    activities = []
    warnings: list[str] = []
    total_weeks = 0
    seasons = []
    for sheet in sheets:
        sheet_result = _extract_calendar_sheet_activities(sheet["rawRows"], sheet["name"], sheet.get("colors"))
        total_weeks = max(total_weeks, sheet_result["totalWeeks"])
        seasons.append({"name": sheet["name"], "activities": len(sheet_result["activities"])})
        if not sheet_result["activities"]:
            warnings.append(f'No active calendar rows found in sheet "{sheet["name"]}".')
        activities.extend(sheet_result["activities"])

    return {
        "title": metadata.get("title") or f'{metadata.get("crop") or "Crop"} Calendar',
        "description": metadata.get("description") or "",
        "regionCode": metadata.get("region") or metadata.get("regionCode") or "",
        "districtCode": metadata.get("district") or metadata.get("districtCode") or "",
        "crop": metadata.get("crop") or metadata.get("commodity") or "",
        "calendarType": "seasonal",
        "totalWeeks": total_weeks or 24,
        "activities": activities,
        "sampleActivities": [item["activityName"] for item in activities[:6]],
        "sheets": [_sheet_summary(sheet) for sheet in sheets],
        "seasons": seasons,
        "warnings": warnings,
        "errors": [],
    }


def _build_poultry_calendar_preview(sheets: list[dict], metadata: dict) -> dict:
    activities = []
    warnings: list[str] = []
    total_weeks = 0
    for sheet in sheets:
        sheet_result = _extract_calendar_sheet_activities(sheet["rawRows"], sheet["name"], sheet.get("colors"))
        total_weeks = max(total_weeks, sheet_result["totalWeeks"])
        if not sheet_result["activities"]:
            warnings.append(f'No active production rows found in sheet "{sheet["name"]}".')
        activities.extend(sheet_result["activities"])

    return {
        "title": metadata.get("title") or f'{metadata.get("poultryType") or "Poultry"} Calendar',
        "description": metadata.get("description") or "",
        "regionCode": metadata.get("region") or metadata.get("regionCode") or "",
        "districtCode": metadata.get("district") or metadata.get("districtCode") or "",
        "crop": metadata.get("poultryType") or metadata.get("commodity") or "poultry",
        "breedType": metadata.get("breedCode") or metadata.get("breedType") or "",
        "calendarType": "cycle",
        "totalWeeks": total_weeks or 8,
        "activities": activities,
        "sampleActivities": [item["activityName"] for item in activities[:6]],
        "sheets": [_sheet_summary(sheet) for sheet in sheets],
        "warnings": warnings,
        "errors": [],
    }


def _extract_calendar_sheet_activities(rows: list[list[str]], sheet_name: str, colors: list[list[str | None]] | None = None) -> dict:
    if not rows:
        return {"activities": [], "totalWeeks": 0}

    activity_col = None
    first_timeline_col = None
    total_timeline_cols = 0
    header_row_index = None
    for idx, row in enumerate(rows[:6]):
        normalized = [str(cell).strip().lower() for cell in row]
        for col, cell in enumerate(normalized):
            if "stage of activity" in cell or "production stage" in cell or cell == "activity":
                activity_col = col
                header_row_index = idx
            if cell.startswith("jan") or cell.startswith("week 1") or cell.startswith("wk1") or cell == "week 1-13":
                first_timeline_col = col if first_timeline_col is None else min(first_timeline_col, col)
        if activity_col is not None and first_timeline_col is not None:
            break

    if activity_col is None:
        activity_col = 1 if len(rows[0]) > 1 else 0
    if first_timeline_col is None:
        first_timeline_col = min(activity_col + 1, len(rows[0]) - 1 if rows[0] else 0)

    header_row_index = header_row_index if header_row_index is not None else 0
    max_cols = max((len(row) for row in rows), default=0)
    total_timeline_cols = max(0, max_cols - first_timeline_col)
    data_rows = rows[header_row_index + 1 :]
    activities = []

    for row_index, row in enumerate(data_rows):
        values = [str(cell).strip() for cell in row]
        abs_row_index = row_index + header_row_index + 1
        color_row = colors[abs_row_index] if colors and abs_row_index < len(colors) else None
        if not any(values) and not (color_row and any(color_row)):
            continue
        if activity_col >= len(values):
            continue
        name = values[activity_col].strip()
        if not name or name.lower() in {"s/n", "calendar date"} or name.startswith("[") or name.isdigit():
            continue
        active_indices = []
        activity_colors: set[str] = set()
        for col in range(first_timeline_col, len(values)):
            has_text = bool(values[col])
            has_color = bool(color_row and col < len(color_row) and color_row[col])
            if has_text or has_color:
                active_indices.append(col - first_timeline_col)
                if has_color:
                    activity_colors.add(color_row[col])
        if not active_indices:
            continue
        # Use the largest contiguous block to avoid stray decorative cells
        best_start, best_end = _largest_contiguous_block(active_indices)
        start_week = best_start + 1
        end_week = best_end + 1
        activities.append(
            {
                "sheetName": sheet_name,
                "activityCode": _slugify(name),
                "activityName": name,
                "startWeek": start_week,
                "endWeek": end_week,
                "productionWeek": start_week,
                "metadata": {
                    "sheetName": sheet_name,
                    "rowIndex": abs_row_index,
                    "colors": sorted(activity_colors) if activity_colors else [],
                },
            }
        )
    return {"activities": activities, "totalWeeks": total_timeline_cols}


def _largest_contiguous_block(indices: list[int]) -> tuple[int, int]:
    """Return (start, end) of the longest contiguous run in a sorted list of indices."""
    if not indices:
        return (0, 0)
    sorted_idx = sorted(indices)
    best_start = best_end = sorted_idx[0]
    cur_start = cur_end = sorted_idx[0]
    best_len = 1
    for i in range(1, len(sorted_idx)):
        if sorted_idx[i] == cur_end + 1:
            cur_end = sorted_idx[i]
        else:
            cur_start = cur_end = sorted_idx[i]
        cur_len = cur_end - cur_start + 1
        if cur_len > best_len:
            best_len = cur_len
            best_start = cur_start
            best_end = cur_end
    return (best_start, best_end)


def _slugify(value: str) -> str:
    text = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    return "-".join(part for part in text.split("-") if part) or "activity"


def _extract_weather_metrics(rows: list[list[str]]) -> dict:
    metrics = {}
    for row in rows:
        if len(row) < 2:
            continue
        label = str(row[0]).strip("[] ").lower()
        if label == "forecast":
            continue
        if any(keyword in label for keyword in ("rainfall", "temp", "humidity", "soil moisture", "wind", "solar")):
            value = next((str(cell).strip() for cell in row[1:] if str(cell).strip()), "")
            if value:
                metrics[label] = value
    return metrics


def _extract_named_items(rows: list[list[str]], header_candidates: list[str], sheet_name: str) -> list[dict]:
    results: list[dict] = []
    header_index = _find_header_index(rows, header_candidates)
    if header_index is None:
        return results
    header_row = [str(cell).strip().lower() for cell in rows[header_index]]
    data_rows = rows[header_index + 1 :]
    name_col = next((index for index, cell in enumerate(header_row) if any(candidate in cell for candidate in header_candidates)), 0)
    time_col = next((index for index, cell in enumerate(header_row) if any(token in cell for token in ("week", "timing", "month", "date"))), None)

    for row in data_rows:
        values = [str(cell).strip() for cell in row]
        if not any(values):
            break
        name = values[name_col] if name_col < len(values) else ""
        if not name or name.startswith("["):
            continue
        results.append(
            {
                "sheetName": sheet_name,
                "activity": name,
                "weekLabel": values[time_col] if time_col is not None and time_col < len(values) else "",
            }
        )
    return results


def _extract_section_items(rows: list[list[str]], header_candidates: list[str], sheet_name: str) -> list[dict]:
    results: list[dict] = []
    header_index = _find_header_index(rows, header_candidates)
    if header_index is None:
        return results
    header_row = [str(cell).strip().lower() for cell in rows[header_index]]
    data_rows = rows[header_index + 1 :]
    text_col = next((index for index, cell in enumerate(header_row) if any(candidate in cell for candidate in header_candidates)), 0)
    for row in data_rows:
        values = [str(cell).strip() for cell in row]
        if not any(values):
            break
        text = values[text_col] if text_col < len(values) else ""
        if not text or text.startswith("["):
            continue
        results.append({"sheetName": sheet_name, "text": text})
    return results


def _extract_key_value_rows(rows: list[list[str]], header_candidates: list[str]) -> dict:
    metrics = {}
    header_index = _find_header_index(rows, header_candidates)
    if header_index is None:
        return metrics
    header_row = [str(cell).strip().lower() for cell in rows[header_index]]
    key_col = next((index for index, cell in enumerate(header_row) if "parameter" in cell or "indicator" in cell), 0)
    value_col = next((index for index, cell in enumerate(header_row) if "recommended" in cell or "normal" in cell or "value" in cell), 1)
    for row in rows[header_index + 1 :]:
        values = [str(cell).strip() for cell in row]
        if not any(values):
            break
        key = values[key_col] if key_col < len(values) else ""
        value = values[value_col] if value_col < len(values) else ""
        if key and value and not key.startswith("["):
            metrics[key] = value
    return metrics


def _find_header_index(rows: list[list[str]], header_candidates: list[str]) -> int | None:
    for index, row in enumerate(rows):
        normalized = " | ".join(str(cell).strip().lower() for cell in row if str(cell).strip())
        if normalized and all(candidate in normalized for candidate in header_candidates[:1]):
            return index
    return None
