# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Schema definitions for Gmail data models."""

from reliquary_automationbench._upstream.schema.gmail.base import GmailRecord, GmailState, generate_gmail_id
from reliquary_automationbench._upstream.schema.gmail.attachment import Attachment
from reliquary_automationbench._upstream.schema.gmail.draft import Draft
from reliquary_automationbench._upstream.schema.gmail.label import Label, get_default_labels
from reliquary_automationbench._upstream.schema.gmail.message import Message

# Rebuild forward references after all imports
GmailState.model_rebuild()

__all__ = [
    "GmailRecord",
    "GmailState",
    "generate_gmail_id",
    "Attachment",
    "Message",
    "Label",
    "Draft",
    "get_default_labels",
]
