# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Schema models for Google Sheets."""

from reliquary_automationbench._upstream.schema.google_sheets.base import (
    GoogleSheetsRecord,
    GoogleSheetsState,
    generate_google_sheets_id,
)
from reliquary_automationbench._upstream.schema.google_sheets.row import Row
from reliquary_automationbench._upstream.schema.google_sheets.spreadsheet import Spreadsheet
from reliquary_automationbench._upstream.schema.google_sheets.worksheet import Worksheet

# Resolve forward references now that all models are loaded
GoogleSheetsState.model_rebuild()
Spreadsheet.model_rebuild()
Worksheet.model_rebuild()

__all__ = [
    "GoogleSheetsRecord",
    "GoogleSheetsState",
    "Row",
    "Spreadsheet",
    "Worksheet",
    "generate_google_sheets_id",
]
