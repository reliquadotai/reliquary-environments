# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""DocuSign schema model definitions used by AutomationBench."""

from reliquary_automationbench._upstream.schema.docusign.base import (
    DocuSignRecord,
    DocuSignState,
    generate_docusign_id,
)
from reliquary_automationbench._upstream.schema.docusign.envelope import (
    CarbonCopy,
    Envelope,
    EnvelopeDocument,
    Sender,
    Signer,
)
from reliquary_automationbench._upstream.schema.docusign.template import (
    Template,
    TemplateDocument,
    TemplateOwner,
    TemplateRecipient,
)
from reliquary_automationbench._upstream.schema.docusign.workspace import (
    Workspace,
    WorkspaceDocument,
    WorkspaceMember,
)

# Trigger model rebuild so forward references in DocuSignState are resolved
DocuSignState.model_rebuild()

__all__ = [
    "CarbonCopy",
    "DocuSignRecord",
    "DocuSignState",
    "generate_docusign_id",
    "Envelope",
    "EnvelopeDocument",
    "Sender",
    "Signer",
    "Template",
    "TemplateDocument",
    "TemplateOwner",
    "TemplateRecipient",
    "Workspace",
    "WorkspaceDocument",
    "WorkspaceMember",
]
